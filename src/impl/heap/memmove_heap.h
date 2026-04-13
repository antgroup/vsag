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
 * @file memmove_heap.h
 * @brief Heap implementation using memory movement for ordered insertion.
 */

#pragma once

#include "distance_heap.h"

namespace vsag {

/**
 * @brief Heap implementation using memory movement for ordered insertion.
 *
 * This template class provides a heap implementation that maintains elements
 * in sorted order using memory movement during insertion. This can be more
 * efficient for small heaps where binary heap overhead is significant.
 *
 * @tparam max_heap If true, creates a max-heap; otherwise creates a min-heap.
 * @tparam fixed_size If true, maintains a fixed maximum size.
 */
template <bool max_heap = true, bool fixed_size = true>
class MemmoveHeap : public DistanceHeap {
public:
    /**
     * @brief Constructs a MemmoveHeap with an allocator and maximum size.
     *
     * @param allocator Pointer to the allocator for memory management.
     * @param max_size Maximum number of elements in the heap.
     */
    explicit MemmoveHeap(Allocator* allocator, int64_t max_size);

    /**
     * @brief Default destructor.
     */
    ~MemmoveHeap() override = default;

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
        return this->ordered_buffer_[cur_size_ - 1];
    }

    /**
     * @brief Removes the top element from the heap.
     */
    void
    Pop() override {
        cur_size_--;
    }

    /**
     * @brief Gets the current number of elements in the heap.
     *
     * @return Number of elements in the heap.
     */
    [[nodiscard]] uint64_t
    Size() const override {
        return this->cur_size_;
    }

    /**
     * @brief Checks if the heap is empty.
     *
     * @return true if the heap is empty, false otherwise.
     */
    [[nodiscard]] bool
    Empty() const override {
        return this->cur_size_ == 0;
    }

    /**
     * @brief Gets the underlying data array.
     *
     * @return Const pointer to the data array.
     */
    [[nodiscard]] const DistanceRecord*
    GetData() const override {
        return this->ordered_buffer_.data();
    }

private:
    /// Buffer storing elements in sorted order.
    Vector<DistanceRecord> ordered_buffer_;

    /// Current number of elements in the heap.
    int64_t cur_size_{0};
};

}  // namespace vsag