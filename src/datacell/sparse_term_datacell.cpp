
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sparse_term_datacell.h"

#include "utils/util_functions.h"
#include "vsag/allocator.h"
namespace vsag {

void
SparseTermDataCell::Query(float* global_dists, const SparseTermComputerPtr& computer) const {
    while (computer->HasNextTerm()) {
        auto it = computer->NextTermIter();
        auto term = computer->GetTerm(it);
        if (computer->HasNextTerm()) {
            auto next_it = it + 1;
            auto next_term = computer->GetTerm(next_it);
            if (next_term < term_ids_.size() && term_ids_[next_term]) {
                __builtin_prefetch(term_ids_[next_term]->data(), 0, 3);
                __builtin_prefetch(term_datas_[next_term]->data(), 0, 3);
            }
        }
        if (term >= term_sizes_.size() || term_sizes_[term] == 0) {
            continue;
        }

        auto term_size = static_cast<uint32_t>(static_cast<float>(term_sizes_[term]) *
                                               computer->term_retain_ratio_);

        if (use_quantization_) {
            computer->ScanForAccumulate(
                it, term_ids_[term]->data(), term_datas_[term]->data(), term_size, global_dists);
        } else {
            computer->ScanForAccumulate(it,
                                        term_ids_[term]->data(),
                                        reinterpret_cast<const float*>(term_datas_[term]->data()),
                                        term_size,
                                        global_dists);
        }
    }
    computer->ResetTerm();
}

template <InnerSearchMode mode, InnerSearchType type>
void
SparseTermDataCell::insert_candidate_into_heap(uint32_t id,
                                               float& dist,
                                               float& cur_heap_top,
                                               MaxHeap& heap,
                                               uint32_t offset_id,
                                               float radius,
                                               const FilterPtr& filter) const {
    if constexpr (type == InnerSearchType::WITH_FILTER) {
#if __cplusplus >= 202002L
        if (dist > cur_heap_top or not filter->CheckValid(id + offset_id)) [[likely]] {
#else
        if (__builtin_expect(dist > cur_heap_top or not filter->CheckValid(id + offset_id), 1)) {
#endif
            dist = 0;
            return;
        }
    } else {
#if __cplusplus >= 202002L
        if (dist > cur_heap_top) [[likely]] {
#else
        if (__builtin_expect(dist > cur_heap_top, 1)) {
#endif
            dist = 0;
            return;
        }
    }
    heap.emplace(dist, id + offset_id);
    if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
        heap.pop();
        cur_heap_top = heap.top().first;
    }
    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        cur_heap_top = radius - 1;
    }
    dist = 0;
}

template <InnerSearchType type>
bool
SparseTermDataCell::fill_heap_initial(uint32_t id,
                                      float& dist,
                                      float& cur_heap_top,
                                      MaxHeap& heap,
                                      uint32_t offset_id,
                                      uint32_t n_candidate,
                                      const FilterPtr& filter) const {
    if (dist < 0) {
        if constexpr (type == InnerSearchType::WITH_FILTER) {
            if (not filter->CheckValid(id + offset_id)) {
                dist = 0;
                return false;
            }
        }
        heap.emplace(dist, id + offset_id);
        cur_heap_top = heap.top().first;
        dist = 0;
        return heap.size() == n_candidate;
    }
    return false;
}

template <InnerSearchMode mode, InnerSearchType type>
void
SparseTermDataCell::InsertHeapByTermLists(float* dists,
                                          const SparseTermComputerPtr& computer,
                                          MaxHeap& heap,
                                          const InnerSearchParam& param,
                                          uint32_t offset_id) const {
    uint32_t id = 0;
    float cur_heap_top = std::numeric_limits<float>::max();
    auto n_candidate = param.ef;
    auto radius = param.radius;
    auto filter = param.is_inner_id_allowed;

    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        // note that radius = 1 - ip -> radius - 1 = 0 - ip
        // the dist in heap is equal to 0 - ip
        // thus, we need to compare dist with radius - 1
        cur_heap_top = radius - 1;
    }

    while (computer->HasNextTerm()) {
        auto it = computer->NextTermIter();
        auto term = computer->GetTerm(it);
        if (term >= term_ids_.size()) {
            continue;
        }

        uint32_t i = 0;
        auto term_size = static_cast<uint32_t>(static_cast<float>(term_sizes_[term]) *
                                               computer->term_retain_ratio_);
        auto& one_term_ids = *term_ids_[term];
        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            if (heap.size() < n_candidate) {
                for (; i < term_size; i++) {
                    id = one_term_ids[i];
                    if (fill_heap_initial<type>(
                            id, dists[id], cur_heap_top, heap, offset_id, n_candidate, filter)) {
                        i++;
                        break;
                    }
                }
            }
        }

        for (; i < term_size; i++) {
            id = one_term_ids[i];
            insert_candidate_into_heap<mode, type>(
                id, dists[id], cur_heap_top, heap, offset_id, radius, filter);
        }
    }
    computer->ResetTerm();
}

template <InnerSearchMode mode, InnerSearchType type>
void
SparseTermDataCell::InsertHeapByDists(float* dists,
                                      uint32_t dists_size,
                                      MaxHeap& heap,
                                      const InnerSearchParam& param,
                                      uint32_t offset_id) const {
    float cur_heap_top = std::numeric_limits<float>::max();
    auto n_candidate = param.ef;
    auto radius = param.radius;
    auto filter = param.is_inner_id_allowed;

    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        cur_heap_top = radius - 1;
    }

    uint32_t id = 0;
    if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
        if (heap.size() < n_candidate) {
            for (; id < total_count_; id++) {
                if (fill_heap_initial<type>(
                        id, dists[id], cur_heap_top, heap, offset_id, n_candidate, filter)) {
                    id++;
                    break;
                }
            }
        }
    }

    for (; id < total_count_; id++) {
        insert_candidate_into_heap<mode, type>(
            id, dists[id], cur_heap_top, heap, offset_id, radius, filter);
    }
}

void
SparseTermDataCell::DocPrune(Vector<std::pair<uint32_t, float>>& sorted_base) const {
    // use this function when inserting
    if (sorted_base.size() <= 1 || doc_retain_ratio_ == 1) {
        return;
    }
    float total_mass = 0.0F;
    for (const auto& pair : sorted_base) {
        total_mass += pair.second;
    }

    float part_mass = total_mass * doc_retain_ratio_;
    float temp_mass = 0.0F;
    int pruned_doc_len = 0;

    while (temp_mass < part_mass) {
        temp_mass += sorted_base[pruned_doc_len++].second;
    }

    sorted_base.resize(pruned_doc_len);
}

void
SparseTermDataCell::InsertVector(const SparseVector& sparse_base, uint16_t base_id) {
    // resize term
    uint32_t max_term_id = 0;
    for (auto i = 0; i < sparse_base.len_; i++) {
        auto term_id = sparse_base.ids_[i];
        max_term_id = std::max(max_term_id, term_id);
    }
    if (max_term_id > term_id_limit_) {
        throw std::runtime_error(
            fmt::format("max term id of sparse vector {} is greater than term id limit {}",
                        max_term_id,
                        term_id_limit_));
    }
    ResizeTermList(max_term_id + 1);

    Vector<std::pair<uint32_t, float>> sorted_base(allocator_);
    sort_sparse_vector(sparse_base, sorted_base);

    // doc prune
    DocPrune(sorted_base);

    // insert vector
    for (auto& item : sorted_base) {
        auto term = item.first;
        auto val = item.second;

        if (term_sizes_[term] == 0) {  // create term until needed
            term_ids_[term] = std::make_unique<Vector<uint16_t>>(allocator_);
            term_datas_[term] = std::make_unique<Vector<uint8_t>>(allocator_);
        }

        term_ids_[term]->push_back(base_id);

        auto& data_vec = *term_datas_[term];
        if (use_quantization_) {
            uint8_t buffer;
            Encode(val, &buffer);
            data_vec.push_back(buffer);
        } else {
            auto old_size = data_vec.size();
            data_vec.resize(old_size + sizeof(float));
            *reinterpret_cast<float*>(data_vec.data() + old_size) = val;
        }

        term_sizes_[term] += 1;
    }
    total_count_++;
}

void
SparseTermDataCell::ResizeTermList(InnerIdType new_term_capacity) {
    if (new_term_capacity <= term_capacity_) {
        return;
    }
    Vector<std::unique_ptr<Vector<uint16_t>>> new_ids(new_term_capacity, allocator_);
    Vector<std::unique_ptr<Vector<uint8_t>>> new_datas(new_term_capacity, allocator_);
    Vector<uint32_t> new_sizes(new_term_capacity, 0, allocator_);

    std::move(term_ids_.begin(), term_ids_.end(), new_ids.begin());
    std::move(term_datas_.begin(), term_datas_.end(), new_datas.begin());
    std::copy(term_sizes_.begin(), term_sizes_.end(), new_sizes.begin());

    term_ids_.swap(new_ids);
    term_datas_.swap(new_datas);
    term_sizes_.swap(new_sizes);
    term_capacity_ = new_term_capacity;
}

float
SparseTermDataCell::CalcDistanceByInnerId(const SparseTermComputerPtr& computer, uint16_t base_id) {
    float ip = 0;
    Vector<float> temp_data(allocator_);
    while (computer->HasNextTerm()) {
        auto it = computer->NextTermIter();
        auto term = computer->GetTerm(it);
        if (computer->HasNextTerm()) {
            auto next_it = it + 1;
            auto next_term = computer->GetTerm(next_it);
            if (next_term >= term_ids_.size()) {
                continue;
            }
            __builtin_prefetch(term_ids_[next_term]->data(), 0, 3);
            __builtin_prefetch(term_datas_[next_term]->data(), 0, 3);
        }
        // Fix: Check term_sizes_[term] == 0 to avoid null pointer dereference
        if (term >= term_ids_.size() || term_sizes_[term] == 0) {
            continue;
        }

        // Dequantize
        const float* vals = nullptr;
        auto size = term_sizes_[term];
        if (use_quantization_) {
            temp_data.resize(size);
            Decode(term_datas_[term]->data(), size, temp_data.data());
            vals = temp_data.data();
        } else {
            vals = reinterpret_cast<const float*>(term_datas_[term]->data());
        }

        computer->ScanForCalculateDist(
            it, term_ids_[term]->data(), vals, term_sizes_[term], base_id, &ip);
    }
    computer->ResetTerm();
    return 1 + ip;
}

int64_t
SparseTermDataCell::GetMemoryUsage() const {
    auto memory = sizeof(SparseTermDataCell);
    memory += term_ids_.size() * sizeof(std::unique_ptr<Vector<uint16_t>>);
    memory += term_datas_.size() * sizeof(std::unique_ptr<Vector<uint8_t>>);
    for (const auto& ptr : term_ids_) {
        if (ptr != nullptr) {
            memory += ptr->size() * sizeof(uint16_t);
        }
    }
    for (const auto& ptr : term_datas_) {
        if (ptr != nullptr) {
            memory += ptr->size() * sizeof(uint8_t);
        }
    }
    memory += sizeof(QuantizationParams);
    memory += term_sizes_.size() * sizeof(uint32_t);
    return static_cast<int64_t>(memory);
}

void
SparseTermDataCell::GetSparseVector(uint32_t base_id,
                                    SparseVector* data,
                                    Allocator* specified_allocator) {
    Allocator* allocator = specified_allocator != nullptr ? specified_allocator : allocator_;

    Vector<uint32_t> ids(allocator);
    Vector<float> vals(allocator);

    for (auto term = 0; term < term_ids_.size(); term++) {
        if (term_sizes_[term] == 0) {
            continue;
        }
        auto& one_term_ids = *term_ids_[term];
        for (auto i = 0; i < term_sizes_[term]; i++) {
            if (one_term_ids[i] == base_id) {
                ids.push_back(term);
                float v;
                if (use_quantization_) {
                    Decode(term_datas_[term]->data() + i, 1, &v);
                } else {
                    v = reinterpret_cast<float*>(term_datas_[term]->data())[i];
                }
                vals.push_back(v);
            }
        }
    }

    data->len_ = ids.size();
    data->ids_ = static_cast<uint32_t*>(allocator->Allocate(sizeof(uint32_t) * data->len_));
    data->vals_ = static_cast<float*>(allocator->Allocate(sizeof(float) * data->len_));

    memcpy(data->ids_, ids.data(), data->len_ * sizeof(uint32_t));
    memcpy(data->vals_, vals.data(), data->len_ * sizeof(float));
}

template <typename T, typename U>
void
convert(const Vector<T>& input, Vector<U>& output) {
    output.clear();
    output.reserve(input.size());
    for (const auto& value : input) {
        output.push_back(static_cast<U>(value));
    }
}

void
SparseTermDataCell::Serialize(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, term_capacity_);
    Vector<float> empty_data(allocator_);
    Vector<uint32_t> empty_ids(allocator_);
    Vector<float> buffer_data(allocator_);
    Vector<uint32_t> buffer_ids(allocator_);
    for (auto i = 0; i < term_capacity_; i++) {
        if (term_sizes_[i] != 0) {
            convert(*term_ids_[i], buffer_ids);
            StreamWriter::WriteVector(writer, buffer_ids);
            auto buffer_size =
                align_up(static_cast<int64_t>(term_datas_[i]->size()), sizeof(float)) /
                sizeof(float);
            buffer_data.resize(buffer_size);
            std::memcpy(buffer_data.data(),
                        term_datas_[i]->data(),
                        sizeof(uint8_t) * term_datas_[i]->size());
            StreamWriter::WriteVector(writer, buffer_data);
        } else {
            StreamWriter::WriteVector(writer, empty_ids);
            StreamWriter::WriteVector(writer, empty_data);
        }
    }
    StreamWriter::WriteVector(writer, term_sizes_);
}

void
SparseTermDataCell::Deserialize(StreamReader& reader) {
    uint32_t term_capacity;
    StreamReader::ReadObj(reader, term_capacity);
    ResizeTermList(term_capacity);
    Vector<uint32_t> ids_buffer(allocator_);
    Vector<float> data_buffer(allocator_);
    for (auto i = 0; i < term_capacity; i++) {
        StreamReader::ReadVector(reader, ids_buffer);
        StreamReader::ReadVector(reader, data_buffer);
        if (not ids_buffer.empty()) {
            term_ids_[i] = std::make_unique<Vector<uint16_t>>(allocator_);
            term_datas_[i] =
                std::make_unique<Vector<uint8_t>>(sizeof(float) * data_buffer.size(), allocator_);
            std::memcpy(
                term_datas_[i]->data(), data_buffer.data(), sizeof(float) * data_buffer.size());
            convert(ids_buffer, *term_ids_[i]);
            if (use_quantization_) {
                term_datas_[i]->resize(term_ids_[i]->size());
            }
        }
    }
    StreamReader::ReadVector(reader, term_sizes_);
}

void
SparseTermDataCell::Encode(float val, uint8_t* dst) const {
    float x = (val - quantization_params_->min_val) / quantization_params_->diff * 255.0F;
    *dst = static_cast<uint8_t>(std::clamp(x, 0.0F, 255.0F));
}

void
SparseTermDataCell::Decode(const uint8_t* src, size_t size, float* dst) const {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = static_cast<float>(src[i]) / 255.0F * quantization_params_->diff +
                 quantization_params_->min_val;
    }
}

template void
SparseTermDataCell::InsertHeapByTermLists<InnerSearchMode::KNN_SEARCH, InnerSearchType::PURE>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeapByTermLists<InnerSearchMode::KNN_SEARCH,
                                          InnerSearchType::WITH_FILTER>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeapByTermLists<InnerSearchMode::RANGE_SEARCH, InnerSearchType::PURE>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeapByTermLists<InnerSearchMode::RANGE_SEARCH,
                                          InnerSearchType::WITH_FILTER>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeapByDists<InnerSearchMode::KNN_SEARCH, InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeapByDists<InnerSearchMode::KNN_SEARCH, InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH, InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH, InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

}  // namespace vsag
