
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
    Pop() override;

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
    // Hand-rolled sift operations: hoisting the moved record into a register
    // and writing it once at its final slot beats the generic iterator-based
    // std::push_heap / std::pop_heap on the per-neighbor hot path.
    static constexpr bool kIsMaxHeap = max_heap;

    static bool
    higher_than(const DistanceRecord& a, const DistanceRecord& b) {
        if constexpr (kIsMaxHeap) {
            return CompareMax()(a, b);
        } else {
            return CompareMin()(a, b);
        }
    }

    void
    sift_up(uint64_t idx) {
        const auto value = this->queue_[idx];
        while (idx > 0) {
            const auto parent = (idx - 1) / 2;
            if (!higher_than(value, this->queue_[parent])) {
                break;
            }
            this->queue_[idx] = this->queue_[parent];
            idx = parent;
        }
        this->queue_[idx] = value;
    }

    void
    sift_down(uint64_t idx, uint64_t size) {
        const auto value = this->queue_[idx];
        while (true) {
            auto child = 2 * idx + 1;
            if (child >= size) {
                break;
            }
            if (child + 1 < size && higher_than(this->queue_[child + 1], this->queue_[child])) {
                ++child;
            }
            if (!higher_than(this->queue_[child], value)) {
                break;
            }
            this->queue_[idx] = this->queue_[child];
            idx = child;
        }
        this->queue_[idx] = value;
    }

    Vector<DistanceRecord> queue_;
};
}  // namespace vsag
