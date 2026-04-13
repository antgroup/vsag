
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

#include "typing.h"
#include "utils/visited_list.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/iterator_context.h"

namespace vsag {

/**
 * @brief Context for iterator-based filtered search operations.
 *
 * This class manages the state and data structures needed for performing
 * filtered search using an iterator pattern, maintaining a discard heap
 * for candidates and tracking visited nodes.
 */
class IteratorFilterContext : public IteratorContext {
public:
    IteratorFilterContext() : is_first_used_(true){};
    ~IteratorFilterContext();

    /**
     * @brief Initializes the context with the specified parameters.
     *
     * @param max_size Maximum number of elements in the visited list.
     * @param ef_search Search beam width parameter.
     * @param allocator Allocator for memory management.
     * @return tl::expected<void, Error> Success or error on initialization.
     */
    tl::expected<void, Error>
    init(InnerIdType max_size, int64_t ef_search, Allocator* allocator);

    /**
     * @brief Adds a node to the discard heap for later retrieval.
     *
     * @param dis Distance to the query point.
     * @param inner_id Internal ID of the node.
     */
    void
    AddDiscardNode(float dis, uint32_t inner_id);

    /**
     * @brief Gets the ID of the top element in the discard heap.
     *
     * @return uint32_t Internal ID of the top candidate.
     */
    uint32_t
    GetTopID();

    /**
     * @brief Gets the distance of the top element in the discard heap.
     *
     * @return float Distance value of the top candidate.
     */
    float
    GetTopDist();

    /**
     * @brief Removes the top element from the discard heap.
     */
    void
    PopDiscard();

    /**
     * @brief Checks if the discard heap is empty.
     *
     * @return true if empty, false otherwise.
     */
    bool
    Empty();

    /**
     * @brief Checks if this is the first use of the iterator.
     *
     * @return true if first use, false otherwise.
     */
    bool
    IsFirstUsed() const;

    /**
     * @brief Marks the iterator as no longer being first use.
     */
    void
    SetOFFFirstUsed();

    /**
     * @brief Marks a point as visited in the visited list.
     *
     * @param inner_id Internal ID to mark as visited.
     */
    void
    SetPoint(uint32_t inner_id);

    /**
     * @brief Checks if a point has been visited.
     *
     * @param inner_id Internal ID to check.
     * @return true if visited, false otherwise.
     */
    bool
    CheckPoint(uint32_t inner_id);

    /**
     * @brief Gets the number of elements in the discard heap.
     *
     * @return int64_t Number of discarded candidates.
     */
    int64_t
    GetDiscardElementNum();

private:
    static constexpr uint32_t BITS_PER_BYTE = 8;
    static constexpr uint32_t BYTE_POS_MASK = 3;  ///< 2^3 for byte position calculation
    static constexpr uint32_t BIT_POS_MASK = BITS_PER_BYTE - 1;

    /**
     * @brief Calculates byte position from bit position.
     *
     * @param pos Bit position.
     * @return uint32_t Byte position in the visited list.
     */
    static inline uint32_t
    byte_pos(uint32_t pos) {
        return pos >> BYTE_POS_MASK;
    }

    /**
     * @brief Calculates bit position within a byte.
     *
     * @param pos Full position.
     * @return uint32_t Bit position within the byte.
     */
    static inline uint32_t
    bit_pos(uint32_t pos) {
        return pos & BIT_POS_MASK;
    }

private:
    int64_t ef_search_{-1};             ///< Search beam width parameter
    bool is_first_used_{true};          ///< Flag indicating first iterator use
    uint32_t max_size_{0};              ///< Maximum size of the visited list
    Allocator* allocator_{nullptr};     ///< Allocator for memory management
    uint8_t* list_{nullptr};            ///< Bit array for tracking visited nodes
    std::unique_ptr<MaxHeap> discard_;  ///< Heap of discarded candidates
};

};  // namespace vsag
