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
 * @file distance_heap.h
 * @brief Base class for distance-based heap implementations.
 */

#pragma once

#include <type_traits>
#include <utility>

#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

DEFINE_POINTER2(DistHeap, DistanceHeap);

/**
 * @brief Abstract base class for distance-based heap data structures.
 *
 * This class provides a common interface for heaps that store distance-record
 * pairs, supporting both max-heap and min-heap configurations.
 */
class DistanceHeap {
public:
    /// Type alias for a distance-record pair (distance, internal ID).
    using DistanceRecord = std::pair<float, InnerIdType>;

    /**
     * @brief Gets the implementation as a specific heap type.
     *
     * @tparam HeapImpl The concrete heap implementation type.
     * @return Reference to the heap implementation.
     */
    template <class HeapImpl>
    HeapImpl&
    GetImpl() {
        return static_cast<HeapImpl>(*this);
    }

    /**
     * @brief Comparator for max-heap ordering.
     *
     * Orders elements so that larger distances have higher priority.
     */
    struct CompareMax {
        /**
         * @brief Compares two distance records for max-heap ordering.
         *
         * @param a First distance record.
         * @param b Second distance record.
         * @return true if a should come before b in max-heap order.
         */
        constexpr bool
        operator()(DistanceRecord const& a, DistanceRecord const& b) const noexcept {
            return a.first < b.first;
        }
    };

    /**
     * @brief Comparator for min-heap ordering.
     *
     * Orders elements so that smaller distances have higher priority.
     */
    struct CompareMin {
        /**
         * @brief Compares two distance records for min-heap ordering.
         *
         * @param a First distance record.
         * @param b Second distance record.
         * @return true if a should come before b in min-heap order.
         */
        constexpr bool
        operator()(DistanceRecord const& a, DistanceRecord const& b) const noexcept {
            return a.first > b.first;
        }
    };

public:
    /**
     * @brief Factory method to create a heap instance based on parameters.
     *
     * @tparam max_heap If true, creates a max-heap; otherwise creates a min-heap.
     * @tparam fixed_size If true, creates a fixed-size heap.
     * @param allocator Pointer to the allocator for memory management.
     * @param max_size Maximum number of elements in the heap.
     * @return Smart pointer to the created heap instance.
     */
    template <bool max_heap, bool fixed_size>
    static DistHeapPtr
    MakeInstanceBySize(Allocator* allocator, int64_t max_size);

public:
    /**
     * @brief Constructs a DistanceHeap with an allocator.
     *
     * @param allocator Pointer to the allocator for memory management.
     */
    explicit DistanceHeap(Allocator* allocator);

    /**
     * @brief Constructs a DistanceHeap with an allocator and maximum size.
     *
     * @param allocator Pointer to the allocator for memory management.
     * @param max_size Maximum number of elements in the heap.
     */
    explicit DistanceHeap(Allocator* allocator, int64_t max_size);

    /**
     * @brief Virtual destructor.
     */
    virtual ~DistanceHeap() = default;

    /**
     * @brief Pushes a distance record onto the heap.
     *
     * @param record The distance record to push.
     */
    virtual void
    Push(const DistanceRecord& record);

    /**
     * @brief Pushes a distance and ID pair onto the heap.
     *
     * @param dist The distance value.
     * @param id The internal ID.
     */
    virtual void
    Push(float dist, InnerIdType id) = 0;

    /**
     * @brief Gets the top element of the heap.
     *
     * @return Const reference to the top distance record.
     */
    [[nodiscard]] virtual const DistanceRecord&
    Top() const = 0;

    /**
     * @brief Removes the top element from the heap.
     */
    virtual void
    Pop() = 0;

    /**
     * @brief Gets the current number of elements in the heap.
     *
     * @return Number of elements in the heap.
     */
    [[nodiscard]] virtual uint64_t
    Size() const = 0;

    /**
     * @brief Checks if the heap is empty.
     *
     * @return true if the heap is empty, false otherwise.
     */
    [[nodiscard]] virtual bool
    Empty() const = 0;

    /**
     * @brief Gets the underlying data array.
     *
     * @return Const pointer to the data array.
     */
    [[nodiscard]] virtual const DistanceRecord*
    GetData() const = 0;

    /**
     * @brief Merges another heap into this heap.
     *
     * @param other The heap to merge.
     */
    virtual void
    Merge(const DistanceHeap& other);

protected:
    /// Pointer to the allocator for memory management.
    Allocator* allocator_{nullptr};
    /// Maximum number of elements in the heap (-1 for unlimited).
    int64_t max_size_{-1};
};
}  // namespace vsag