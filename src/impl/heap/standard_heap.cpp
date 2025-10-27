
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

namespace vsag {
template <bool max_heap, bool fixed_size>
StandardHeap<max_heap, fixed_size>::StandardHeap(Allocator* allocator, int64_t max_size)
    : DistanceHeap(allocator, max_size), queue_(allocator) {
}

template <bool max_heap, bool fixed_size>
void
StandardHeap<max_heap, fixed_size>::Push(float dist, InnerIdType id) {
    if constexpr (fixed_size) {
        if (this->queue_.size() == this->max_size_) {
            if constexpr (max_heap) {
                if (dist > this->queue_.front().first) {
                    return;
                }
            } else {
                if (dist < this->queue_.front().first) {
                    return;
                }
            }
        }
    }
    DistanceRecord record{dist, id};
    this->queue_.emplace_back(std::move(record));
    if constexpr (max_heap) {
        std::push_heap(this->queue_.begin(), this->queue_.end(), CompareMax());
    } else {
        std::push_heap(this->queue_.begin(), this->queue_.end(), CompareMin());
    }

    if constexpr (fixed_size) {
        if (this->queue_.size() > max_size_) {
            this->Pop();
        }
    }
}

template class StandardHeap<true, true>;
template class StandardHeap<true, false>;
template class StandardHeap<false, true>;
template class StandardHeap<false, false>;

}  // namespace vsag