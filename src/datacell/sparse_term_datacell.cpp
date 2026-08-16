
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

#include <algorithm>
#include <cstring>
#include <numeric>
#include <type_traits>

#include "simd/fp16_simd.h"
#include "utils/util_functions.h"
#include "vsag/allocator.h"
#include "vsag_exception.h"
namespace vsag {

uint32_t
SparseTermDataCell::GetSparseValueCodeSize(SparseValueQuantizationType type) {
    switch (type) {
        case SparseValueQuantizationType::FP32:
            return sizeof(float);
        case SparseValueQuantizationType::SQ8:
            return sizeof(uint8_t);
        case SparseValueQuantizationType::FP16:
            return sizeof(uint16_t);
        default:
            CHECK_ARGUMENT(false, "unknown sparse value quantization type");
    }
    return sizeof(float);
}

void
SparseTermDataCell::SortPostingListByValue(uint16_t* ids,
                                           uint8_t* data,
                                           uint32_t posting_count,
                                           SparseValueQuantizationType quantization_type,
                                           Vector<uint32_t>& order,
                                           Vector<uint16_t>& sorted_ids,
                                           Vector<uint8_t>& sorted_data) {
    if (posting_count == 0) {
        return;
    }

    order.resize(posting_count);
    std::iota(order.begin(), order.end(), 0);
    if (posting_count == 1) {
        return;
    }

    const auto sort_by_code = [&order, ids, posting_count, &sorted_ids, &sorted_data](auto* codes) {
        using CodeType = std::remove_pointer_t<decltype(codes)>;
        const auto compare = [codes, ids](uint32_t left, uint32_t right) {
            if (codes[left] != codes[right]) {
                return codes[left] > codes[right];
            }
            return ids[left] < ids[right];
        };
        if (std::is_sorted(order.begin(), order.end(), compare)) {
            return;
        }
        std::sort(order.begin(), order.end(), compare);

        sorted_ids.resize(posting_count);
        sorted_data.resize(static_cast<uint64_t>(posting_count) * sizeof(CodeType));
        auto* sorted_codes = reinterpret_cast<CodeType*>(sorted_data.data());
        for (uint32_t i = 0; i < posting_count; ++i) {
            const auto source = order[i];
            sorted_ids[i] = ids[source];
            sorted_codes[i] = codes[source];
        }
        std::copy(sorted_ids.begin(), sorted_ids.end(), ids);
        std::copy(sorted_codes, sorted_codes + posting_count, codes);
    };
    switch (quantization_type) {
        case SparseValueQuantizationType::SQ8:
            sort_by_code(data);
            break;
        case SparseValueQuantizationType::FP16:
            sort_by_code(reinterpret_cast<uint16_t*>(data));
            break;
        case SparseValueQuantizationType::FP32:
            sort_by_code(reinterpret_cast<float*>(data));
            break;
        default:
            CHECK_ARGUMENT(false, "unknown sparse value quantization type");
    }
}

void
SparseTermDataCell::Query(float* global_dists, const SparseTermComputerPtr& computer) const {
    query_impl(global_dists, computer, nullptr);
}

uint64_t
SparseTermDataCell::Query(float* global_dists,
                          const SparseTermComputerPtr& computer,
                          SparseEvaluationTracker& evaluation_tracker) const {
    evaluation_tracker.BeginWindow();
    query_impl(global_dists, computer, &evaluation_tracker);
    return evaluation_tracker.Count();
}

void
SparseTermDataCell::query_impl(float* global_dists,
                               const SparseTermComputerPtr& computer,
                               SparseEvaluationTracker* evaluation_tracker) const {
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

        const auto posting_count = term_sizes_[term];
        const auto term_count = computer->GetTermScanCount(posting_count);
        const auto* term_ids = term_ids_[term]->data();
        const auto* term_data = term_datas_[term]->data();
        if (evaluation_tracker != nullptr) {
            evaluation_tracker->Mark(term_ids, term_count);
        }

        if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
            computer->ScanForAccumulateSQ8(it, term_ids, term_data, term_count, global_dists);
        } else if (sparse_value_quant_type_ == SparseValueQuantizationType::FP16) {
            computer->ScanForAccumulateFP16Bytes(it, term_ids, term_data, term_count, global_dists);
        } else {
            computer->ScanForAccumulateFloatBytes(
                it, term_ids, term_data, term_count, global_dists);
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
                                               uint32_t n_candidate,
                                               float radius,
                                               const FilterPtr& filter,
                                               const std::optional<float>& threshold,
                                               bool enable_reorder) const {
    if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
        if (threshold.has_value() and
            (not std::isfinite(dist) or (not enable_reorder and 1.0F + dist > threshold.value()))) {
            dist = 0.0F;
            return;
        }
    }
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
        if (heap.size() > n_candidate) {
            heap.pop();
        }
        cur_heap_top =
            heap.size() == n_candidate ? heap.top().first : std::numeric_limits<float>::max();
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
                                      const FilterPtr& filter,
                                      const std::optional<float>& threshold,
                                      bool enable_reorder) const {
    if (threshold.has_value() and
        (not std::isfinite(dist) or (not enable_reorder and 1.0F + dist > threshold.value()))) {
        dist = 0.0F;
        return false;
    }
    if (dist < 0) {
        if constexpr (type == InnerSearchType::WITH_FILTER) {
            const bool valid = filter->CheckValid(id + offset_id);
            if (not valid) {
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
bool
SparseTermDataCell::InsertHeapByTermLists(float* dists,
                                          const SparseTermComputerPtr& computer,
                                          MaxHeap& heap,
                                          const InnerSearchParam& param,
                                          uint32_t offset_id,
                                          const uint64_t* filter_callback_limit) const {
    uint32_t id = 0;
    float cur_heap_top = std::numeric_limits<float>::max();
    auto n_candidate = param.ef;
    auto radius = param.radius;
    auto filter = param.is_inner_id_allowed;

    if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
        if (heap.size() == n_candidate) {
            cur_heap_top = heap.top().first;
        }
    }

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

        const auto posting_count = term_sizes_[term];
        const auto term_count = computer->GetTermScanCount(posting_count);
        uint32_t i = 0;
        const auto term_end = term_count;
        auto& one_term_ids = *term_ids_[term];
        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            if (heap.size() < n_candidate) {
                for (; i < term_end; i++) {
                    id = one_term_ids[i];
                    const bool heap_filled = fill_heap_initial<type>(id,
                                                                     dists[id],
                                                                     cur_heap_top,
                                                                     heap,
                                                                     offset_id,
                                                                     n_candidate,
                                                                     filter,
                                                                     param.distance_threshold,
                                                                     param.enable_reorder);
                    if constexpr (type == InnerSearchType::WITH_FILTER) {
                        if (filter_callback_limit != nullptr and *filter_callback_limit == 0) {
                            computer->ResetTerm();
                            return true;
                        }
                    }
                    if (heap_filled) {
                        i++;
                        break;
                    }
                }
            }
        }

        for (; i < term_end; i++) {
            id = one_term_ids[i];
            insert_candidate_into_heap<mode, type>(id,
                                                   dists[id],
                                                   cur_heap_top,
                                                   heap,
                                                   offset_id,
                                                   n_candidate,
                                                   radius,
                                                   filter,
                                                   param.distance_threshold,
                                                   param.enable_reorder);
            if constexpr (type == InnerSearchType::WITH_FILTER) {
                if (filter_callback_limit != nullptr and *filter_callback_limit == 0) {
                    computer->ResetTerm();
                    return true;
                }
            }
        }
    }
    computer->ResetTerm();
    return false;
}

template <InnerSearchMode mode, InnerSearchType type>
bool
SparseTermDataCell::InsertHeapByDists(float* dists,
                                      uint32_t dists_size,
                                      MaxHeap& heap,
                                      const InnerSearchParam& param,
                                      uint32_t offset_id,
                                      const uint64_t* filter_callback_limit) const {
    float cur_heap_top = std::numeric_limits<float>::max();
    auto n_candidate = param.ef;
    auto radius = param.radius;
    auto filter = param.is_inner_id_allowed;

    if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
        if (heap.size() == n_candidate) {
            cur_heap_top = heap.top().first;
        }
    }

    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        cur_heap_top = radius - 1;
    }

    uint32_t id = 0;
    if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
        if (heap.size() < n_candidate) {
            for (; id < total_count_; id++) {
                const bool heap_filled = fill_heap_initial<type>(id,
                                                                 dists[id],
                                                                 cur_heap_top,
                                                                 heap,
                                                                 offset_id,
                                                                 n_candidate,
                                                                 filter,
                                                                 param.distance_threshold,
                                                                 param.enable_reorder);
                if constexpr (type == InnerSearchType::WITH_FILTER) {
                    if (filter_callback_limit != nullptr and *filter_callback_limit == 0) {
                        return true;
                    }
                }
                if (heap_filled) {
                    id++;
                    break;
                }
            }
        }
    }

    for (; id < total_count_; id++) {
        insert_candidate_into_heap<mode, type>(id,
                                               dists[id],
                                               cur_heap_top,
                                               heap,
                                               offset_id,
                                               n_candidate,
                                               radius,
                                               filter,
                                               param.distance_threshold,
                                               param.enable_reorder);
        if constexpr (type == InnerSearchType::WITH_FILTER) {
            if (filter_callback_limit != nullptr and *filter_callback_limit == 0) {
                return true;
            }
        }
    }
    return false;
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

    while (temp_mass < part_mass && pruned_doc_len < static_cast<int>(sorted_base.size())) {
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
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
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
        if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
            uint8_t buffer;
            Encode(val, &buffer);
            data_vec.push_back(buffer);
        } else if (sparse_value_quant_type_ == SparseValueQuantizationType::FP16) {
            auto old_size = data_vec.size();
            data_vec.resize(old_size + sizeof(uint16_t));
            auto fp16_value = generic::FloatToFP16(val);
            std::memcpy(data_vec.data() + old_size, &fp16_value, sizeof(fp16_value));
        } else {
            auto old_size = data_vec.size();
            data_vec.resize(old_size + sizeof(float));
            std::memcpy(data_vec.data() + old_size, &val, sizeof(val));
        }

        term_sizes_[term] += 1;
    }
    total_count_++;
}

void
SparseTermDataCell::SortByValue() {
    Vector<uint32_t> order(allocator_);
    Vector<uint16_t> sorted_ids(allocator_);
    Vector<uint8_t> sorted_data(allocator_);

    for (uint32_t term = 0; term < term_sizes_.size(); ++term) {
        const auto term_size = term_sizes_[term];
        if (term_size == 0) {
            continue;
        }

        auto& ids = *term_ids_[term];
        auto& data = *term_datas_[term];
        SortPostingListByValue(ids.data(),
                               data.data(),
                               term_size,
                               sparse_value_quant_type_,
                               order,
                               sorted_ids,
                               sorted_data);
    }
}

void
SparseTermDataCell::ResizeTermList(InnerIdType new_term_capacity) {
    if (new_term_capacity <= term_capacity_) {
        return;
    }
    InnerIdType new_capacity = term_capacity_ == 0 ? new_term_capacity : term_capacity_;
    while (new_capacity < new_term_capacity) {
        if (new_capacity > std::numeric_limits<InnerIdType>::max() / 2) {
            new_capacity = new_term_capacity;
            break;
        }
        new_capacity *= 2;
    }
    Vector<std::unique_ptr<Vector<uint16_t>>> new_ids(new_capacity, allocator_);
    Vector<std::unique_ptr<Vector<uint8_t>>> new_datas(new_capacity, allocator_);
    Vector<uint32_t> new_sizes(new_capacity, 0, allocator_);

    std::move(term_ids_.begin(), term_ids_.end(), new_ids.begin());
    std::move(term_datas_.begin(), term_datas_.end(), new_datas.begin());
    std::copy(term_sizes_.begin(), term_sizes_.end(), new_sizes.begin());

    term_ids_.swap(new_ids);
    term_datas_.swap(new_datas);
    term_sizes_.swap(new_sizes);
    term_capacity_ = new_capacity;
}

void
SparseTermDataCell::Compact() {
    uint32_t compact_term_capacity = 0;
    const uint64_t compactable_capacity = std::min(
        std::min(static_cast<uint64_t>(term_capacity_), static_cast<uint64_t>(term_sizes_.size())),
        std::min(static_cast<uint64_t>(term_ids_.size()),
                 static_cast<uint64_t>(term_datas_.size())));
    for (uint64_t i = 0; i < compactable_capacity; ++i) {
        if (term_sizes_[i] != 0) {
            compact_term_capacity = static_cast<uint32_t>(i + 1);
        }
    }

    Vector<std::unique_ptr<Vector<uint16_t>>> compact_ids(compact_term_capacity, allocator_);
    Vector<std::unique_ptr<Vector<uint8_t>>> compact_datas(compact_term_capacity, allocator_);
    Vector<uint32_t> compact_sizes(compact_term_capacity, 0, allocator_);
    for (uint32_t i = 0; i < compact_term_capacity; ++i) {
        compact_sizes[i] = term_sizes_[i];
        if (term_sizes_[i] != 0) {
            CHECK_ARGUMENT(term_ids_[i] != nullptr && term_datas_[i] != nullptr,
                           "non-empty sparse term has null posting data");
            compact_ids[i] = std::make_unique<Vector<uint16_t>>(
                term_ids_[i]->begin(), term_ids_[i]->end(), allocator_);
            compact_datas[i] = std::make_unique<Vector<uint8_t>>(
                term_datas_[i]->begin(), term_datas_[i]->end(), allocator_);
        }
    }

    term_ids_.swap(compact_ids);
    term_datas_.swap(compact_datas);
    term_sizes_.swap(compact_sizes);
    term_capacity_ = compact_term_capacity;
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
            if (next_term < term_ids_.size() && term_sizes_[next_term] != 0) {
                __builtin_prefetch(term_ids_[next_term]->data(), 0, 3);
                __builtin_prefetch(term_datas_[next_term]->data(), 0, 3);
            }
        }
        // Fix: Check term_sizes_[term] == 0 to avoid null pointer dereference
        if (term >= term_ids_.size() || term_sizes_[term] == 0) {
            continue;
        }

        auto size = term_sizes_[term];
        if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
            temp_data.resize(size);
            Decode(term_datas_[term]->data(), size, temp_data.data());
            computer->ScanForCalculateDist(
                it, term_ids_[term]->data(), temp_data.data(), term_sizes_[term], base_id, &ip);
        } else if (sparse_value_quant_type_ == SparseValueQuantizationType::FP16) {
            computer->ScanForCalculateDistFP16Bytes(it,
                                                    term_ids_[term]->data(),
                                                    term_datas_[term]->data(),
                                                    term_sizes_[term],
                                                    base_id,
                                                    &ip);
        } else {
            computer->ScanForCalculateDistFloatBytes(it,
                                                     term_ids_[term]->data(),
                                                     term_datas_[term]->data(),
                                                     term_sizes_[term],
                                                     base_id,
                                                     &ip);
        }
    }
    computer->ResetTerm();
    return 1 + ip;
}

uint64_t
SparseTermDataCell::GetMemoryUsage() const {
    auto memory = sizeof(SparseTermDataCell);
    memory += term_ids_.capacity() * sizeof(std::unique_ptr<Vector<uint16_t>>);
    memory += term_datas_.capacity() * sizeof(std::unique_ptr<Vector<uint8_t>>);
    for (const auto& ptr : term_ids_) {
        if (ptr != nullptr) {
            memory += sizeof(Vector<uint16_t>);
            memory += ptr->capacity() * sizeof(uint16_t);
        }
    }
    for (const auto& ptr : term_datas_) {
        if (ptr != nullptr) {
            memory += sizeof(Vector<uint8_t>);
            memory += ptr->capacity() * sizeof(uint8_t);
        }
    }
    memory += sizeof(QuantizationParams);
    memory += term_sizes_.capacity() * sizeof(uint32_t);
    return static_cast<uint64_t>(memory);
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
                if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
                    Decode(term_datas_[term]->data() + i, 1, &v);
                } else if (sparse_value_quant_type_ == SparseValueQuantizationType::FP16) {
                    uint16_t fp16_value = 0;
                    std::memcpy(
                        &fp16_value,
                        term_datas_[term]->data() + static_cast<uint64_t>(i) * sizeof(fp16_value),
                        sizeof(fp16_value));
                    v = generic::FP16ToFloat(fp16_value);
                } else {
                    std::memcpy(&v,
                                term_datas_[term]->data() + static_cast<uint64_t>(i) * sizeof(v),
                                sizeof(v));
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
SparseTermDataCell::Deserialize(StreamReader& reader, bool postings_sorted) {
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
            if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
                term_datas_[i]->resize(term_ids_[i]->size());
            } else if (sparse_value_quant_type_ == SparseValueQuantizationType::FP16) {
                term_datas_[i]->resize(term_ids_[i]->size() * sizeof(uint16_t));
            }
        }
    }
    StreamReader::ReadVector(reader, term_sizes_);
    for (uint64_t i = 0; i < term_ids_.size(); ++i) {
        if (i >= term_sizes_.size() || term_sizes_[i] == 0) {
            term_ids_[i].reset();
            term_datas_[i].reset();
        }
    }

    if (not postings_sorted) {
        SortByValue();
    }
    Compact();

    // Restore total_count_ from compacted deserialized data (not serialized for compatibility).
    total_count_ = 0;
    for (const auto& term_id_vec : term_ids_) {
        if (term_id_vec != nullptr) {
            for (uint16_t id : *term_id_vec) {
                total_count_ = std::max(total_count_, static_cast<int64_t>(id) + 1);
            }
        }
    }
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

template bool
SparseTermDataCell::InsertHeapByTermLists<InnerSearchMode::KNN_SEARCH, InnerSearchType::PURE>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_limit) const;

template bool
SparseTermDataCell::InsertHeapByTermLists<InnerSearchMode::KNN_SEARCH,
                                          InnerSearchType::WITH_FILTER>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_limit) const;

template bool
SparseTermDataCell::InsertHeapByTermLists<InnerSearchMode::RANGE_SEARCH, InnerSearchType::PURE>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_limit) const;

template bool
SparseTermDataCell::InsertHeapByTermLists<InnerSearchMode::RANGE_SEARCH,
                                          InnerSearchType::WITH_FILTER>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_limit) const;

template bool
SparseTermDataCell::InsertHeapByDists<InnerSearchMode::KNN_SEARCH, InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_limit) const;

template bool
SparseTermDataCell::InsertHeapByDists<InnerSearchMode::KNN_SEARCH, InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_limit) const;

template bool
SparseTermDataCell::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH, InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_limit) const;

template bool
SparseTermDataCell::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH, InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_limit) const;

}  // namespace vsag
