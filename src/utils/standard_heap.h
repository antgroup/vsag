
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

#include <queue>

#include "distance_heap.h"

namespace vsag {
template <bool max_heap = true, bool fixed_size = true>
class StandardHeap : public DistanceHeap {
public:
    using QueueMax =
        std::priority_queue<DistanceRecord, Vector<std::pair<float, InnerIdType>>, CompareMax>;

    using QueueMin =
        std::priority_queue<DistanceRecord, Vector<std::pair<float, InnerIdType>>, CompareMin>;

public:
    StandardHeap(Allocator* allocator, int64_t max_size);

    void
    Push(float dist, InnerIdType id) override;

    [[nodiscard]] const DistanceRecord&
    Top() const override {
        return this->queue_.top();
    }

    void
    Pop() override {
        this->queue_.pop();
    }

    [[nodiscard]] uint64_t
    Size() const override {
        return this->queue_.size();
    }

    [[nodiscard]] bool
    Empty() const override {
        return this->queue_.size() == 0;
    }

private:
    typename std::conditional<max_heap, QueueMax, QueueMin>::type queue_;
};

template <bool max_heap, bool fixed_size>
StandardHeap<max_heap, fixed_size>::StandardHeap(Allocator* allocator, int64_t max_size)
    : DistanceHeap(allocator, max_size), queue_(allocator) {
}

template <bool max_heap, bool fixed_size>
void
StandardHeap<max_heap, fixed_size>::Push(float dist, InnerIdType id) {
    if constexpr (fixed_size) {
        if (this->queue_.size() < max_size_ or (dist < this->queue_.top().first) == max_heap) {
            queue_.emplace(dist, id);
            if (this->queue_.size() > this->max_size_) {
                this->queue_.pop();
            }
        }
    } else {
        queue_.emplace(dist, id);
    }
}

}  // namespace vsag
