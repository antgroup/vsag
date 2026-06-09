
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

#pragma once

namespace vsag {

template <typename IOTmpl>
template <InnerSearchMode mode, InnerSearchType type>
void
DiskSparseTermListDataCell<IOTmpl>::insert_candidate_into_heap(uint32_t id,
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

template <typename IOTmpl>
template <InnerSearchType type>
bool
DiskSparseTermListDataCell<IOTmpl>::fill_heap_initial(uint32_t id,
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

template <typename IOTmpl>
template <InnerSearchMode mode, InnerSearchType type>
void
DiskSparseTermListDataCell<IOTmpl>::InsertHeapByWindow(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers) const {
    std::shared_lock lock(term_buffers_mutex_);
    uint32_t id = 0;
    float cur_heap_top = std::numeric_limits<float>::max();
    auto n_candidate = param.ef;
    auto radius = param.radius;
    auto filter = param.is_inner_id_allowed;

    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        cur_heap_top = radius - 1;
    }

    while (computer->HasNextTerm()) {
        auto it = computer->NextTermIter();
        auto term = computer->GetTerm(it);
        auto* tb = this->GetTermBuffer(term, query_term_buffers);
        if (tb == nullptr) {
            continue;
        }
        if (window_id >= window_count_) {
            continue;
        }
        uint32_t start = tb->window_offsets[window_id];
        uint32_t end = tb->window_offsets[window_id + 1];
        uint32_t term_size =
            static_cast<uint32_t>(static_cast<float>(end - start) * computer->term_retain_ratio_);

        auto& one_term_ids = tb->ids;
        uint32_t i = start;
        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            if (heap.size() < n_candidate) {
                for (; i < start + term_size; i++) {
                    id = one_term_ids[i];
                    if (fill_heap_initial<type>(
                            id, dists[id], cur_heap_top, heap, offset_id, n_candidate, filter)) {
                        i++;
                        break;
                    }
                }
            }
        }

        for (; i < start + term_size; i++) {
            id = one_term_ids[i];
            insert_candidate_into_heap<mode, type>(
                id, dists[id], cur_heap_top, heap, offset_id, radius, filter);
        }
    }
    computer->ResetTerm();
}

template <typename IOTmpl>
template <InnerSearchMode mode, InnerSearchType type>
void
DiskSparseTermListDataCell<IOTmpl>::InsertHeapByDists(float* dists,
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

}  // namespace vsag
