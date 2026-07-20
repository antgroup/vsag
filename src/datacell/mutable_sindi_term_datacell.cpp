
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

#include "mutable_sindi_term_datacell.h"

#include <algorithm>
#include <cstring>

#include "datacell/sindi_datacell_utils.h"
#include "simd/fp16_simd.h"
#include "utils/util_functions.h"
#include "vsag/allocator.h"
#include "vsag_exception.h"
namespace vsag {

void
MutableSindiTermDataCell::Finalize() {
    for (auto& window : windows_) {
        this->Compact(window);
    }
}

void
MutableSindiTermDataCell::SerializeWindows(StreamWriter& writer) const {
    for (const auto& window : windows_) {
        this->SerializeWindow(writer, window);
    }
}

void
MutableSindiTermDataCell::DeserializeWindows(StreamReader& reader, uint32_t window_count) {
    Vector<MutableSINDIWindow> loaded(allocator_);
    loaded.reserve(window_count);
    for (uint32_t i = 0; i < window_count; ++i) {
        loaded.emplace_back(allocator_);
        this->DeserializeWindow(reader, loaded.back());
    }
    windows_.swap(loaded);
    total_count_ = 0;
    for (uint32_t i = 0; i < windows_.size(); ++i) {
        uint32_t window_count = 0;
        for (const auto& ids : windows_[i].term_ids_) {
            if (ids != nullptr && not ids->empty()) {
                window_count = std::max(window_count, static_cast<uint32_t>(ids->back()) + 1);
            }
        }
        total_count_ =
            std::max(total_count_, static_cast<int64_t>(i) * window_size_ + window_count);
    }
}

void
MutableSindiTermDataCell::SerializeTermLayout(StreamWriter& writer,
                                              uint32_t term_dict_count) const {
    const auto layout = sindi_datacell_utils::BuildTermLayout(
        term_dict_count, GetWindowCount(), GetTermValueCodeSize(), this->CollectTermPostings());
    sindi_datacell_utils::SerializeTermLayout(writer, layout, GetTermValueCodeSize());
}

uint32_t
MutableSindiTermDataCell::GetTermValueCodeSize() const {
    return sindi_datacell_utils::GetValueCodeSize(sparse_value_quant_type_);
}

SindiTermPostingView
MutableSindiTermDataCell::GetTermPostingView(uint32_t term_id, uint32_t window_id) const {
    if (window_id >= windows_.size()) {
        return {};
    }
    const auto& window = windows_[window_id];
    if (term_id >= window.term_sizes_.size() || window.term_sizes_[term_id] == 0) {
        return {};
    }
    return {window.term_ids_[term_id]->data(),
            window.term_datas_[term_id]->data(),
            window.term_sizes_[term_id]};
}

std::vector<sindi_datacell_utils::TermPostingRecord>
MutableSindiTermDataCell::CollectTermPostings() const {
    std::vector<sindi_datacell_utils::TermPostingRecord> postings;
    for (uint32_t window_id = 0; window_id < windows_.size(); ++window_id) {
        const auto& window = windows_[window_id];
        for (uint32_t term_id = 0; term_id < window.term_capacity_; ++term_id) {
            if (window.term_sizes_[term_id] == 0) {
                continue;
            }
            postings.push_back({term_id, window_id, this->GetTermPostingView(term_id, window_id)});
        }
    }
    return postings;
}

void
MutableSindiTermDataCell::DeserializeTermLayout(StreamReader& reader,
                                                uint32_t window_count,
                                                uint64_t total_count) {
    const auto term_dict = sindi_datacell_utils::DeserializeTermDictionary(reader, term_id_limit_);
    uint64_t term_payload_size = 0;
    StreamReader::ReadObj(reader, term_payload_size);
    CHECK_ARGUMENT(reader.GetCursor() <= reader.Length() &&
                       term_payload_size <= reader.Length() - reader.GetCursor(),
                   "SINDI V2 term payload exceeds stream length");
    const auto payload_start = reader.GetCursor();
    const auto value_code_size = this->GetTermValueCodeSize();
    sindi_datacell_utils::ValidateTermDict(term_dict, term_payload_size);

    Vector<MutableSINDIWindow> loaded(allocator_);
    loaded.reserve(window_count);
    for (uint32_t window = 0; window < window_count; ++window) {
        loaded.emplace_back(allocator_);
    }
    for (uint32_t term = 0; term < term_dict.size(); ++term) {
        const auto& entry = term_dict[term];
        if (entry.posting_count == 0) {
            continue;
        }
        auto buffer = sindi_datacell_utils::ReadTermPayload(reader,
                                                            payload_start,
                                                            term_payload_size,
                                                            entry,
                                                            window_count,
                                                            window_size_,
                                                            total_count,
                                                            value_code_size,
                                                            allocator_);

        for (uint32_t window = 0; window < window_count; ++window) {
            const auto [begin, count] = buffer.GetPostingRange(window);
            if (count == 0) {
                continue;
            }
            auto& target = loaded[window];
            this->ResizeTermList(target, term + 1);
            target.term_ids_[term] = std::make_unique<Vector<uint16_t>>(allocator_);
            target.term_datas_[term] = std::make_unique<Vector<uint8_t>>(allocator_);
            target.term_ids_[term]->insert(target.term_ids_[term]->end(),
                                           buffer.ids.begin() + begin,
                                           buffer.ids.begin() + begin + count);
            const auto value_begin = static_cast<uint64_t>(begin) * value_code_size;
            const auto value_end = static_cast<uint64_t>(begin + count) * value_code_size;
            target.term_datas_[term]->insert(target.term_datas_[term]->end(),
                                             buffer.values.begin() + value_begin,
                                             buffer.values.begin() + value_end);
            target.term_sizes_[term] = count;
        }
    }
    for (uint32_t window = 0; window < window_count; ++window) {
        this->Compact(loaded[window]);
    }
    windows_.swap(loaded);
    total_count_ = static_cast<int64_t>(total_count);
    reader.Seek(payload_start + term_payload_size);
}

const MutableSINDIWindow&
MutableSindiTermDataCell::GetWindow(uint32_t window_id) const {
    CHECK_ARGUMENT(window_id < windows_.size(), "mutable SINDI window id out of range");
    return windows_[window_id];
}

MutableSINDIWindow&
MutableSindiTermDataCell::GetWindow(uint32_t window_id) {
    CHECK_ARGUMENT(window_id < windows_.size(), "mutable SINDI window id out of range");
    return windows_[window_id];
}

uint32_t
MutableSindiTermDataCell::GetTermDictCount() const {
    uint32_t term_dict_count = 0;
    for (const auto& window : windows_) {
        for (uint32_t term = window.term_capacity_; term > term_dict_count; --term) {
            if (window.term_sizes_[term - 1] != 0) {
                term_dict_count = term;
                break;
            }
        }
    }
    return term_dict_count;
}

void
MutableSindiTermDataCell::QueryWindow(float* dists,
                                      uint32_t window_id,
                                      const SparseTermComputerPtr& computer,
                                      bool use_term_lists_heap_insert,
                                      const QueryTermBuffers& query_term_buffers) const {
    CHECK_ARGUMENT(window_id < windows_.size(), "mutable SINDI window id out of range");
    const auto& window = windows_[window_id];
    while (computer->HasNextTerm()) {
        auto it = computer->NextTermIter();
        auto term = computer->GetTerm(it);
        if (computer->HasNextTerm()) {
            auto next_it = it + 1;
            auto next_term = computer->GetTerm(next_it);
            if (next_term < window.term_ids_.size() && window.term_ids_[next_term]) {
                __builtin_prefetch(window.term_ids_[next_term]->data(), 0, 3);
                __builtin_prefetch(window.term_datas_[next_term]->data(), 0, 3);
            }
        }
        if (term >= window.term_sizes_.size() || window.term_sizes_[term] == 0) {
            continue;
        }

        auto term_size = static_cast<uint32_t>(static_cast<float>(window.term_sizes_[term]) *
                                               computer->term_retain_ratio_);

        if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
            computer->ScanForAccumulateSQ8(it,
                                           window.term_ids_[term]->data(),
                                           window.term_datas_[term]->data(),
                                           term_size,
                                           dists);
        } else if (sparse_value_quant_type_ == SparseValueQuantizationType::FP16) {
            computer->ScanForAccumulateFP16Bytes(it,
                                                 window.term_ids_[term]->data(),
                                                 window.term_datas_[term]->data(),
                                                 term_size,
                                                 dists);
        } else {
            computer->ScanForAccumulateFloatBytes(it,
                                                  window.term_ids_[term]->data(),
                                                  window.term_datas_[term]->data(),
                                                  term_size,
                                                  dists);
        }
    }
    computer->ResetTerm();
}

void
MutableSindiTermDataCell::InsertHeapByWindow(float* dists,
                                             uint32_t window_id,
                                             const SparseTermComputerPtr& computer,
                                             MaxHeap& heap,
                                             const InnerSearchParam& param,
                                             uint32_t offset_id,
                                             InnerSearchMode mode,
                                             bool with_filter,
                                             const QueryTermBuffers& query_term_buffers) const {
    CHECK_ARGUMENT(window_id < windows_.size(), "mutable SINDI window id out of range");
    const auto& window = windows_[window_id];
    if (mode == InnerSearchMode::KNN_SEARCH) {
        if (with_filter) {
            this->InsertHeapByTermLists<InnerSearchMode::KNN_SEARCH, InnerSearchType::WITH_FILTER>(
                window, dists, computer, heap, param, offset_id);
        } else {
            this->InsertHeapByTermLists<InnerSearchMode::KNN_SEARCH, InnerSearchType::PURE>(
                window, dists, computer, heap, param, offset_id);
        }
    } else if (with_filter) {
        this->InsertHeapByTermLists<InnerSearchMode::RANGE_SEARCH, InnerSearchType::WITH_FILTER>(
            window, dists, computer, heap, param, offset_id);
    } else {
        this->InsertHeapByTermLists<InnerSearchMode::RANGE_SEARCH, InnerSearchType::PURE>(
            window, dists, computer, heap, param, offset_id);
    }
}

void
MutableSindiTermDataCell::InsertHeapByDists(float* dists,
                                            uint32_t dists_size,
                                            MaxHeap& heap,
                                            const InnerSearchParam& param,
                                            uint32_t offset_id,
                                            InnerSearchMode mode,
                                            bool with_filter) const {
    const auto window_id = offset_id / window_size_;
    CHECK_ARGUMENT(window_id < windows_.size(), "mutable SINDI window id out of range");
    if (mode == InnerSearchMode::KNN_SEARCH) {
        if (with_filter) {
            this->InsertHeapByDists<InnerSearchMode::KNN_SEARCH, InnerSearchType::WITH_FILTER>(
                dists, dists_size, heap, param, offset_id);
        } else {
            this->InsertHeapByDists<InnerSearchMode::KNN_SEARCH, InnerSearchType::PURE>(
                dists, dists_size, heap, param, offset_id);
        }
    } else if (with_filter) {
        this->InsertHeapByDists<InnerSearchMode::RANGE_SEARCH, InnerSearchType::WITH_FILTER>(
            dists, dists_size, heap, param, offset_id);
    } else {
        this->InsertHeapByDists<InnerSearchMode::RANGE_SEARCH, InnerSearchType::PURE>(
            dists, dists_size, heap, param, offset_id);
    }
}

template <InnerSearchMode mode, InnerSearchType type>
void
MutableSindiTermDataCell::insert_candidate_into_heap(uint32_t id,
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
MutableSindiTermDataCell::fill_heap_initial(uint32_t id,
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
MutableSindiTermDataCell::InsertHeapByTermLists(float* dists,
                                                const SparseTermComputerPtr& computer,
                                                MaxHeap& heap,
                                                const InnerSearchParam& param,
                                                uint32_t offset_id) const {
    this->InsertHeapByTermLists<mode, type>(
        this->GetWindow(0), dists, computer, heap, param, offset_id);
}

template <InnerSearchMode mode, InnerSearchType type>
void
MutableSindiTermDataCell::InsertHeapByTermLists(const MutableSINDIWindow& window,
                                                float* dists,
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
        if (term >= window.term_ids_.size() || window.term_sizes_[term] == 0) {
            continue;
        }

        uint32_t i = 0;
        auto term_size = static_cast<uint32_t>(static_cast<float>(window.term_sizes_[term]) *
                                               computer->term_retain_ratio_);
        auto& one_term_ids = *window.term_ids_[term];
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
MutableSindiTermDataCell::InsertHeapByDists(float* dists,
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
            for (; id < dists_size; id++) {
                if (fill_heap_initial<type>(
                        id, dists[id], cur_heap_top, heap, offset_id, n_candidate, filter)) {
                    id++;
                    break;
                }
            }
        }
    }

    for (; id < dists_size; id++) {
        insert_candidate_into_heap<mode, type>(
            id, dists[id], cur_heap_top, heap, offset_id, radius, filter);
    }
}

void
MutableSindiTermDataCell::DocPrune(Vector<std::pair<uint32_t, float>>& sorted_base) const {
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
MutableSindiTermDataCell::InsertVector(const SparseVector& sparse_base, uint32_t doc_id) {
    CHECK_ARGUMENT(window_size_ > 0 && window_size_ <= std::numeric_limits<uint16_t>::max() + 1U,
                   "mutable SINDI window size is invalid");
    const auto window_id = doc_id / window_size_;
    while (windows_.size() <= window_id) {
        windows_.emplace_back(allocator_);
    }
    auto* window = &windows_[window_id];
    const auto window_local_id = static_cast<uint16_t>(doc_id % window_size_);

    // Validate all input terms before pruning so invalid input is not silently accepted.
    for (auto i = 0; i < sparse_base.len_; i++) {
        if (sparse_base.ids_[i] > term_id_limit_) {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
                fmt::format("term id of sparse vector {} is greater than term id limit {}",
                            sparse_base.ids_[i],
                            term_id_limit_));
        }
    }

    Vector<std::pair<uint32_t, float>> sorted_base(allocator_);
    sort_sparse_vector(sparse_base, sorted_base);

    // doc prune
    DocPrune(sorted_base);

    if (not sorted_base.empty()) {
        const auto max_term = std::max_element(
            sorted_base.begin(), sorted_base.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.first < rhs.first;
            });
        this->ResizeTermList(*window, max_term->first + 1);
    }

    // insert vector
    for (auto& item : sorted_base) {
        auto term = item.first;
        auto val = item.second;

        if (window->term_sizes_[term] == 0) {
            window->term_ids_[term] = std::make_unique<Vector<uint16_t>>(allocator_);
            window->term_datas_[term] = std::make_unique<Vector<uint8_t>>(allocator_);
        }

        window->term_ids_[term]->push_back(window_local_id);

        auto& data_vec = *window->term_datas_[term];
        const auto old_size = data_vec.size();
        data_vec.resize(old_size + this->GetTermValueCodeSize());
        sindi_datacell_utils::EncodeValue(
            val, sparse_value_quant_type_, quantization_params_.get(), data_vec.data() + old_size);

        window->term_sizes_[term] += 1;
    }
    total_count_ = std::max(total_count_, static_cast<int64_t>(doc_id) + 1);
}

void
MutableSindiTermDataCell::ResizeTermList(InnerIdType new_term_capacity) {
    this->ResizeTermList(this->GetWindow(0), new_term_capacity);
}

void
MutableSindiTermDataCell::ResizeTermList(MutableSINDIWindow& window,
                                         InnerIdType new_term_capacity) {
    if (new_term_capacity <= window.term_capacity_) {
        return;
    }
    InnerIdType new_capacity =
        window.term_capacity_ == 0 ? new_term_capacity : window.term_capacity_;
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

    std::move(window.term_ids_.begin(), window.term_ids_.end(), new_ids.begin());
    std::move(window.term_datas_.begin(), window.term_datas_.end(), new_datas.begin());
    std::copy(window.term_sizes_.begin(), window.term_sizes_.end(), new_sizes.begin());

    window.term_ids_.swap(new_ids);
    window.term_datas_.swap(new_datas);
    window.term_sizes_.swap(new_sizes);
    window.term_capacity_ = new_capacity;
}

void
MutableSindiTermDataCell::Compact() {
    this->Compact(this->GetWindow(0));
}

void
MutableSindiTermDataCell::Compact(MutableSINDIWindow& window) {
    uint32_t compact_term_capacity = 0;
    const uint64_t compactable_capacity =
        std::min(std::min(static_cast<uint64_t>(window.term_capacity_),
                          static_cast<uint64_t>(window.term_sizes_.size())),
                 std::min(static_cast<uint64_t>(window.term_ids_.size()),
                          static_cast<uint64_t>(window.term_datas_.size())));
    for (uint64_t i = 0; i < compactable_capacity; ++i) {
        if (window.term_sizes_[i] != 0) {
            compact_term_capacity = static_cast<uint32_t>(i + 1);
        }
    }

    Vector<std::unique_ptr<Vector<uint16_t>>> compact_ids(compact_term_capacity, allocator_);
    Vector<std::unique_ptr<Vector<uint8_t>>> compact_datas(compact_term_capacity, allocator_);
    Vector<uint32_t> compact_sizes(compact_term_capacity, 0, allocator_);
    for (uint32_t i = 0; i < compact_term_capacity; ++i) {
        compact_sizes[i] = window.term_sizes_[i];
        if (window.term_sizes_[i] != 0) {
            CHECK_ARGUMENT(window.term_ids_[i] != nullptr && window.term_datas_[i] != nullptr,
                           "non-empty sparse term has null posting data");
            compact_ids[i] = std::make_unique<Vector<uint16_t>>(
                window.term_ids_[i]->begin(), window.term_ids_[i]->end(), allocator_);
            compact_datas[i] = std::make_unique<Vector<uint8_t>>(
                window.term_datas_[i]->begin(), window.term_datas_[i]->end(), allocator_);
        }
    }

    window.term_ids_.swap(compact_ids);
    window.term_datas_.swap(compact_datas);
    window.term_sizes_.swap(compact_sizes);
    window.term_capacity_ = compact_term_capacity;
}

float
MutableSindiTermDataCell::CalcDistanceByInnerId(const SparseTermComputerPtr& computer,
                                                uint32_t base_id,
                                                const QueryTermBuffers& query_term_buffers) const {
    const auto window_id = base_id / window_size_;
    CHECK_ARGUMENT(window_id < windows_.size(), "mutable SINDI inner id out of range");
    return this->CalcDistanceByInnerId(
        windows_[window_id], computer, static_cast<uint16_t>(base_id % window_size_));
}

float
MutableSindiTermDataCell::CalcDistanceByInnerId(const SparseTermComputerPtr& computer,
                                                uint16_t base_id) const {
    return this->CalcDistanceByInnerId(this->GetWindow(0), computer, base_id);
}

float
MutableSindiTermDataCell::CalcDistanceByInnerId(const MutableSINDIWindow& window,
                                                const SparseTermComputerPtr& computer,
                                                uint16_t base_id) const {
    float ip = 0;
    Vector<float> temp_data(allocator_);
    while (computer->HasNextTerm()) {
        auto it = computer->NextTermIter();
        auto term = computer->GetTerm(it);
        if (computer->HasNextTerm()) {
            auto next_it = it + 1;
            auto next_term = computer->GetTerm(next_it);
            if (next_term < window.term_ids_.size() && window.term_sizes_[next_term] != 0) {
                __builtin_prefetch(window.term_ids_[next_term]->data(), 0, 3);
                __builtin_prefetch(window.term_datas_[next_term]->data(), 0, 3);
            }
        }
        if (term >= window.term_ids_.size() || window.term_sizes_[term] == 0) {
            continue;
        }

        auto size = window.term_sizes_[term];
        if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
            temp_data.resize(size);
            Decode(window.term_datas_[term]->data(), size, temp_data.data());
            computer->ScanForCalculateDist(it,
                                           window.term_ids_[term]->data(),
                                           temp_data.data(),
                                           window.term_sizes_[term],
                                           base_id,
                                           &ip);
        } else if (sparse_value_quant_type_ == SparseValueQuantizationType::FP16) {
            computer->ScanForCalculateDistFP16Bytes(it,
                                                    window.term_ids_[term]->data(),
                                                    window.term_datas_[term]->data(),
                                                    window.term_sizes_[term],
                                                    base_id,
                                                    &ip);
        } else {
            computer->ScanForCalculateDistFloatBytes(it,
                                                     window.term_ids_[term]->data(),
                                                     window.term_datas_[term]->data(),
                                                     window.term_sizes_[term],
                                                     base_id,
                                                     &ip);
        }
    }
    computer->ResetTerm();
    return 1 + ip;
}

uint64_t
MutableSindiTermDataCell::GetMemoryUsage() const {
    auto memory = sizeof(MutableSindiTermDataCell);
    memory += windows_.capacity() * sizeof(MutableSINDIWindow);
    for (const auto& window : windows_) {
        memory += this->GetWindowMemoryUsage(window);
    }
    memory += sizeof(QuantizationParams);
    return static_cast<uint64_t>(memory);
}

uint64_t
MutableSindiTermDataCell::GetWindowMemoryUsage(const MutableSINDIWindow& window) const {
    uint64_t memory = 0;
    memory += window.term_ids_.capacity() * sizeof(std::unique_ptr<Vector<uint16_t>>);
    memory += window.term_datas_.capacity() * sizeof(std::unique_ptr<Vector<uint8_t>>);
    for (const auto& ptr : window.term_ids_) {
        if (ptr != nullptr) {
            memory += sizeof(Vector<uint16_t>);
            memory += ptr->capacity() * sizeof(uint16_t);
        }
    }
    for (const auto& ptr : window.term_datas_) {
        if (ptr != nullptr) {
            memory += sizeof(Vector<uint8_t>);
            memory += ptr->capacity() * sizeof(uint8_t);
        }
    }
    memory += window.term_sizes_.capacity() * sizeof(uint32_t);
    return memory;
}

void
MutableSindiTermDataCell::GetSparseVector(uint32_t base_id,
                                          SparseVector* data,
                                          Allocator* specified_allocator) const {
    const auto window_id = base_id / window_size_;
    CHECK_ARGUMENT(window_id < windows_.size(), "mutable SINDI inner id out of range");
    this->GetSparseVector(windows_[window_id], base_id % window_size_, data, specified_allocator);
}

void
MutableSindiTermDataCell::GetSparseVector(const MutableSINDIWindow& window,
                                          uint32_t base_id,
                                          SparseVector* data,
                                          Allocator* specified_allocator) const {
    Allocator* allocator = specified_allocator != nullptr ? specified_allocator : allocator_;

    Vector<uint32_t> ids(allocator);
    Vector<float> vals(allocator);

    for (uint32_t term = 0; term < window.term_ids_.size(); ++term) {
        if (window.term_sizes_[term] == 0) {
            continue;
        }
        auto& one_term_ids = *window.term_ids_[term];
        for (uint32_t i = 0; i < window.term_sizes_[term]; ++i) {
            if (one_term_ids[i] == base_id) {
                ids.push_back(term);
                float v;
                if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
                    Decode(window.term_datas_[term]->data() + i, 1, &v);
                } else if (sparse_value_quant_type_ == SparseValueQuantizationType::FP16) {
                    uint16_t fp16_value = 0;
                    std::memcpy(&fp16_value,
                                window.term_datas_[term]->data() +
                                    static_cast<uint64_t>(i) * sizeof(fp16_value),
                                sizeof(fp16_value));
                    v = generic::FP16ToFloat(fp16_value);
                } else {
                    std::memcpy(
                        &v,
                        window.term_datas_[term]->data() + static_cast<uint64_t>(i) * sizeof(v),
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
MutableSindiTermDataCell::SerializeWindow(StreamWriter& writer,
                                          const MutableSINDIWindow& window) const {
    StreamWriter::WriteObj(writer, window.term_capacity_);
    Vector<float> empty_data(allocator_);
    Vector<uint32_t> empty_ids(allocator_);
    Vector<float> buffer_data(allocator_);
    Vector<uint32_t> buffer_ids(allocator_);
    for (uint32_t i = 0; i < window.term_capacity_; ++i) {
        if (window.term_sizes_[i] != 0) {
            convert(*window.term_ids_[i], buffer_ids);
            StreamWriter::WriteVector(writer, buffer_ids);
            auto buffer_size =
                align_up(static_cast<int64_t>(window.term_datas_[i]->size()), sizeof(float)) /
                sizeof(float);
            buffer_data.resize(buffer_size);
            std::memcpy(buffer_data.data(),
                        window.term_datas_[i]->data(),
                        sizeof(uint8_t) * window.term_datas_[i]->size());
            StreamWriter::WriteVector(writer, buffer_data);
        } else {
            StreamWriter::WriteVector(writer, empty_ids);
            StreamWriter::WriteVector(writer, empty_data);
        }
    }
    StreamWriter::WriteVector(writer, window.term_sizes_);
}

void
MutableSindiTermDataCell::DeserializeWindow(StreamReader& reader, MutableSINDIWindow& window) {
    window = MutableSINDIWindow(allocator_);
    uint32_t term_capacity;
    StreamReader::ReadObj(reader, term_capacity);
    this->ResizeTermList(window, term_capacity);
    Vector<uint32_t> ids_buffer(allocator_);
    Vector<float> data_buffer(allocator_);
    for (uint32_t i = 0; i < term_capacity; ++i) {
        StreamReader::ReadVector(reader, ids_buffer);
        StreamReader::ReadVector(reader, data_buffer);
        if (not ids_buffer.empty()) {
            window.term_ids_[i] = std::make_unique<Vector<uint16_t>>(allocator_);
            window.term_datas_[i] =
                std::make_unique<Vector<uint8_t>>(sizeof(float) * data_buffer.size(), allocator_);
            std::memcpy(window.term_datas_[i]->data(),
                        data_buffer.data(),
                        sizeof(float) * data_buffer.size());
            convert(ids_buffer, *window.term_ids_[i]);
            if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
                window.term_datas_[i]->resize(window.term_ids_[i]->size());
            } else if (sparse_value_quant_type_ == SparseValueQuantizationType::FP16) {
                window.term_datas_[i]->resize(window.term_ids_[i]->size() * sizeof(uint16_t));
            }
        }
    }
    StreamReader::ReadVector(reader, window.term_sizes_);
    for (uint64_t i = 0; i < window.term_ids_.size(); ++i) {
        if (i >= window.term_sizes_.size() || window.term_sizes_[i] == 0) {
            window.term_ids_[i].reset();
            window.term_datas_[i].reset();
        }
    }

    this->Compact(window);
}

void
MutableSindiTermDataCell::Encode(float val, uint8_t* dst) const {
    sindi_datacell_utils::EncodeValue(
        val, sparse_value_quant_type_, quantization_params_.get(), dst);
}

void
MutableSindiTermDataCell::Decode(const uint8_t* src, uint64_t size, float* dst) const {
    const auto value_code_size = this->GetTermValueCodeSize();
    for (uint64_t i = 0; i < size; ++i) {
        dst[i] = sindi_datacell_utils::DecodeValue(
            src + i * value_code_size, sparse_value_quant_type_, quantization_params_.get());
    }
}

template void
MutableSindiTermDataCell::InsertHeapByTermLists<InnerSearchMode::KNN_SEARCH, InnerSearchType::PURE>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
MutableSindiTermDataCell::InsertHeapByTermLists<InnerSearchMode::KNN_SEARCH,
                                                InnerSearchType::WITH_FILTER>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
MutableSindiTermDataCell::InsertHeapByTermLists<InnerSearchMode::RANGE_SEARCH,
                                                InnerSearchType::PURE>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
MutableSindiTermDataCell::InsertHeapByTermLists<InnerSearchMode::RANGE_SEARCH,
                                                InnerSearchType::WITH_FILTER>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
MutableSindiTermDataCell::InsertHeapByDists<InnerSearchMode::KNN_SEARCH, InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
MutableSindiTermDataCell::InsertHeapByDists<InnerSearchMode::KNN_SEARCH,
                                            InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
MutableSindiTermDataCell::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH, InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
MutableSindiTermDataCell::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH,
                                            InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

}  // namespace vsag
