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

/**
 * @file standard_heap.h
 * @brief Standard heap implementation using std::heap operations.
 */

#pragma once

#include <queue>

#include "distance_heap.h"

namespace vsag {

/**
 * @brief Standard heap implementation using STL heap operations.
 *
 * This template class provides a heap implementation using std::make_heap,
 * std::push_heap, and std::pop_heap operations on a vector container.
 *
 * @tparam max_heap If true, creates a max-heap; otherwise creates a min-heap.
 * @tparam fixed_size If true, maintains a fixed maximum size.
 */
template <bool max_heap = true, bool fixed_size = true>
class StandardHeap : public DistanceHeap {
public:
    /**
     * @brief Constructs a StandardHeap with an allocator and maximum size.
     *
     * @param allocator Pointer to the allocator for memory management.
     * @param max_size Maximum number of elements in the heap.
     */
    explicit StandardHeap(Allocator* allocator, int64_t max_size);

    /**
     * @brief Default destructor.
     */
    ~StandardHeap() override = default;

    /**
     * @brief Pushes a distance and ID pair onto the heap.
     *
     * @param dist The distance value.
     * @param id The internal ID.
     */
    void
    Push(float dist, InnerIdType id) override;

    /**
     * @brief Gets the top element of the heap.
     *
     * @return Const reference to the top distance record.
     */
    [[nodiscard]] const DistanceRecord&
    Top() const override {
        return this->queue_.front();
    }

    /**
     * @brief Removes the top element from the heap.
     */
    void
    Pop() override {
        if constexpr (max_heap) {
            std::pop_heap(queue_.begin(), queue_.end(), CompareMax());
        } else {
            std::pop_heap(queue_.begin(), queue_.end(), CompareMin());
        }
        queue_.pop_back();
    }

    /**
     * @brief Gets the current number of elements in the heap.
     *
     * @return Number of elements in the heap.
     */
    [[nodiscard]] uint64_t
    Size() const override {
        return this->queue_.size();
    }

    /**
     * @brief Checks if the heap is empty.
     *
     * @return true if the heap is empty, false otherwise.
     */
    [[nodiscard]] bool
    Empty() const override {
        return this->queue_.size() == 0;
    }

    /**
     * @brief Gets the underlying data array.
     *
     * @return Const pointer to the data array.
     */
    [[nodiscard]] const DistanceRecord*
    GetData() const override {
        return this->queue_.data();
    }

private:
    /// Internal vector storing heap elements.
    Vector<DistanceRecord> queue_;
};
}  // namespace vsag