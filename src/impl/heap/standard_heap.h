
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
    explicit StandardHeap(Allocator* allocator, int64_t max_size);

    ~StandardHeap() override = default;

    void
    Push(float dist, InnerIdType id) override;

    [[nodiscard]] const DistanceRecord&
    Top() const override {
        return this->queue_.front();
    }

    void
    Pop() override {
        if constexpr (max_heap) {
            std::pop_heap(queue_.begin(), queue_.end(), CompareMax());
        } else {
            std::pop_heap(queue_.begin(), queue_.end(), CompareMin());
        }
        queue_.pop_back();
    }

    [[nodiscard]] uint64_t
    Size() const override {
        return this->queue_.size();
    }

    [[nodiscard]] bool
    Empty() const override {
        return this->queue_.size() == 0;
    }

    [[nodiscard]] const DistanceRecord*
    GetData() const override {
        return this->queue_.data();
    }

private:
    Vector<DistanceRecord> queue_;
};
}  // namespace vsag
