
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

#include "standard_heap.h"

#include <algorithm>

namespace vsag {
template <bool max_heap, bool fixed_size>
StandardHeap<max_heap, fixed_size>::StandardHeap(Allocator* allocator, int64_t max_size)
    : DistanceHeap(allocator, max_size), queue_(allocator) {
    // Search heaps are usually created empty and grow to a small bounded
    // frontier. Callers that know the expected frontier size (e.g. ef) pass
    // it as max_size so a single reservation covers the whole query; the
    // clamp only guards against pathological hints, and fixed_size=false
    // heaps still grow without bound beyond it.
    constexpr uint64_t kDefaultInitialReserve = 64;
    constexpr uint64_t kMaxInitialReserve = 512;
    uint64_t initial_reserve = kDefaultInitialReserve;
    if (max_size > 0) {
        initial_reserve = std::clamp<uint64_t>(
            static_cast<uint64_t>(max_size), kDefaultInitialReserve, kMaxInitialReserve);
    }
    queue_.reserve(initial_reserve);
}

template <bool max_heap, bool fixed_size>
void
StandardHeap<max_heap, fixed_size>::Push(float dist, InnerIdType id) {
    if constexpr (fixed_size) {
        if (this->queue_.size() == this->max_size_) {
            const bool worse =
                max_heap ? (dist > this->queue_.front().first)
                         : (dist < this->queue_.front().first);
            if (worse) {
                return;
            }
        }
    }
    this->queue_.emplace_back(dist, id);
    this->sift_up(this->queue_.size() - 1);

    if constexpr (fixed_size) {
        if (this->queue_.size() > max_size_) {
            this->Pop();
        }
    }
}

template <bool max_heap, bool fixed_size>
void
StandardHeap<max_heap, fixed_size>::Pop() {
    const auto size = this->queue_.size();
    if (size <= 1) {
        this->queue_.pop_back();
        return;
    }
    this->queue_.front() = this->queue_.back();
    this->queue_.pop_back();
    this->sift_down(0, size - 1);
}

template class StandardHeap<true, true>;
template class StandardHeap<true, false>;
template class StandardHeap<false, true>;
template class StandardHeap<false, false>;

}  // namespace vsag
