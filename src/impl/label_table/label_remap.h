
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

/**
 * @file label_remap.h
 * @brief Reverse mapping table from Label to Internal ID
 *
 * Provides efficient reverse mapping from external labels to internal IDs, supporting:
 * - Fast label lookup via hash tables
 * - Multiple hash table implementations (ROBIN, PG)
 * - Batch traversal operations
 */

#include <memory>

#include "common.h"
#include "typing.h"

namespace vsag {

/**
 * @class LabelRemap
 * @brief Reverse mapping from Label to InternalId
 *
 * Uses hash tables for efficient label to internal ID lookup,
 * supporting two hash table implementations:
 * - ROBIN: Uses tsl::robin_map, suitable for general scenarios
 * - PG: Uses tsl::robin_pg_map, supports growth factor optimization
 */
class LabelRemap {
public:
    /**
     * @brief Construct a new LabelRemap object
     * @param allocator Memory allocator
     * @param remap_type Hash table type, default is PG
     */
    explicit LabelRemap(Allocator* allocator, LabelRemapType remap_type = LabelRemapType::PG);

    /**
     * @brief Clear all mappings
     * @note This operation does not release memory, use Reset() to release memory
     */
    void
    Clear();

    /**
     * @brief Reset the hash table and release memory
     * @note Creates a new empty hash table instance
     */
    void
    Reset();

    /**
     * @brief Reserve capacity for the hash table
     * @param size The number of elements to reserve space for
     * @note Useful for performance optimization when the expected size is known
     */
    void
    Reserve(uint64_t size);

    /**
     * @brief Get the number of mappings in the table
     * @return The number of label-ID pairs stored
     */
    uint64_t
    Size() const;

    /**
     * @brief Insert or update a mapping
     * @param label External label, must be unique
     * @param inner_id Internal ID, must be valid
     * @throws std::bad_alloc on memory allocation failure
     * @note If label exists, updates the corresponding internal ID
     * @complexity Average O(1), worst O(n)
     */
    void
    InsertOrAssign(LabelType label, InnerIdType inner_id);

    /**
     * @brief Insert a new mapping (does not update existing)
     * @param label External label
     * @param inner_id Internal ID
     * @throws std::bad_alloc on memory allocation failure
     * @note If label exists, the operation is ignored
     * @complexity Average O(1), worst O(n)
     */
    void
    Emplace(LabelType label, InnerIdType inner_id);

    /**
     * @brief Remove a mapping by label
     * @param label The label to remove
     * @return True if the label was found and removed, false otherwise
     * @complexity Average O(1), worst O(n)
     */
    bool
    Erase(LabelType label);

    /**
     * @brief Find the internal ID for a given label
     * @param label The label to search for
     * @param inner_id Output parameter for the found internal ID
     * @return True if the label was found, false otherwise
     * @complexity Average O(1), worst O(n)
     */
    bool
    Find(LabelType label, InnerIdType& inner_id) const;

    /**
     * @brief Get the estimated memory usage of the hash table
     * @return Estimated memory usage in bytes
     * @note This includes hash table overhead (buckets, pointers, load factor)
     *       The estimate uses a multiplier to account for container overhead
     */
    int64_t
    GetMemoryUsage() const {
        return Size() * (sizeof(LabelType) + sizeof(InnerIdType)) * 2;
    }

    /**
     * @brief Iterate over all label-ID pairs
     * @tparam Func Function type, should accept (LabelType, InnerIdType) parameters
     * @param func The function to call for each label-ID pair
     * @note The iteration order is not guaranteed
     */
    template <typename Func>
    void
    ForEach(Func&& func) const {
        if (pg_map_ != nullptr) {
            for (const auto& [label, inner_id] : *pg_map_) {
                func(label, inner_id);
            }
            return;
        }
        for (const auto& [label, inner_id] : *robin_map_) {
            func(label, inner_id);
        }
    }

    /**
     * @brief Get the current hash table type
     * @return The hash table type (ROBIN or PG)
     */
    [[nodiscard]] LabelRemapType
    GetType() const {
        return remap_type_;
    }

private:
    Allocator* allocator_;                                             ///< Memory allocator pointer
    LabelRemapType remap_type_;                                        ///< Current hash table type
    std::unique_ptr<UnorderedMap<LabelType, InnerIdType>> robin_map_;  ///< ROBIN hash table
    std::unique_ptr<PGUnorderedMap<LabelType, InnerIdType>> pg_map_;   ///< PG hash table
};

}  // namespace vsag
