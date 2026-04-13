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

#include "resource_object.h"
#include "resource_object_pool.h"
#include "typing.h"
#include "utils/pointer_define.h"
#include "utils/prefetch.h"

namespace vsag {
class Allocator;

DEFINE_POINTER(VisitedList);

/**
 * @file visited_list.h
 * @brief Visited list for tracking visited nodes in graph-based algorithms.
 *
 * This file provides an efficient visited list implementation used in graph-based
 * search algorithms (e.g., HNSW) to track which nodes have been visited during traversal.
 * The implementation uses a tag-based approach for O(1) reset operations.
 */

/**
 * @class VisitedList
 * @brief Efficient visited list for graph traversal using tag-based reset.
 *
 * This class provides a memory-efficient way to track visited nodes in graph algorithms.
 * Instead of clearing the entire array on reset, it uses an incrementing tag value,
 * making reset operations O(1) instead of O(n).
 */
class VisitedList : public ResourceObject {
public:
    /// Type used for visited list entries and tag values
    using VisitedListType = uint16_t;

public:
    /**
     * @brief Constructs a visited list with the specified capacity.
     * @param max_size Maximum number of elements that can be tracked.
     * @param allocator Allocator for memory management.
     */
    explicit VisitedList(InnerIdType max_size, Allocator* allocator);

    /**
     * @brief Destructor that releases allocated memory.
     */
    ~VisitedList() override;

    /**
     * @brief Marks an element as visited.
     * @param id The element identifier to mark as visited.
     */
    void
    Set(const InnerIdType& id) {
        this->list_[id] = this->tag_;
    }

    /**
     * @brief Checks if an element has been visited.
     * @param id The element identifier to check.
     * @return True if the element has been visited, false otherwise.
     */
    [[nodiscard]] bool
    Get(const InnerIdType& id) {
        return this->list_[id] == this->tag_;
    }

    /**
     * @brief Prefetches the cache line containing the element.
     * @param id The element identifier to prefetch.
     */
    void
    Prefetch(const InnerIdType& id) {
        PrefetchLines(this->list_ + id, 64);
    }

    /**
     * @brief Resets the visited list for reuse.
     *
     * Increments the tag value instead of clearing the array,
     * making this operation O(1).
     */
    void
    Reset() override;

    /**
     * @brief Returns the memory usage of this object.
     * @return Memory usage in bytes.
     */
    int64_t
    GetMemoryUsage() const override {
        return sizeof(VisitedList) + sizeof(VisitedListType) * this->max_size_;
    }

private:
    /// Allocator for memory management
    Allocator* const allocator_{nullptr};

    /// Array storing tag values for each element
    VisitedListType* list_{nullptr};

    /// Current tag value for comparison
    VisitedListType tag_{1};

    /// Maximum capacity of the visited list
    const InnerIdType max_size_{0};
};

/// Pool for reusing VisitedList objects
using VisitedListPool = ResourceObjectPool<VisitedList>;
}  // namespace vsag