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
 * @file dense_duplicate_tracker.h
 * @brief Dense duplicate tracker implementation for tracking duplicate vector IDs.
 *
 * This file provides the DenseDuplicateTracker class which implements the
 * DuplicateInterface for efficiently tracking duplicate vectors using
 * a dense array-based storage approach.
 */

#pragma once

#include <shared_mutex>

#include "duplicate_interface.h"
#include "impl/allocator/allocator_wrapper.h"
#include "typing.h"

namespace vsag {

/**
 * @brief Dense duplicate tracker for tracking duplicate vector IDs.
 *
 * This class implements DuplicateInterface and provides functionality for:
 * - Tracking which vectors are duplicates of each other
 * - Using dense array-based storage for efficient memory access
 * - Supporting serialization and deserialization
 */
class DenseDuplicateTracker : public DuplicateInterface {
public:
    /**
     * @brief Constructs a DenseDuplicateTracker.
     * @param allocator The allocator for memory management.
     */
    explicit DenseDuplicateTracker(Allocator* allocator);

    /**
     * @brief Sets a duplicate relationship between two IDs.
     * @param group_id The group ID (primary vector).
     * @param duplicate_id The duplicate ID (duplicate vector).
     */
    void
    SetDuplicateId(InnerIdType group_id, InnerIdType duplicate_id) override;

    /**
     * @brief Gets all duplicate IDs for a given ID.
     * @param id The ID to query.
     * @return Vector of duplicate IDs.
     */
    auto
    GetDuplicateIds(InnerIdType id) const -> std::vector<InnerIdType> override;

    /**
     * @brief Gets the group ID for a given ID.
     * @param id The ID to query.
     * @return The group ID (primary vector ID).
     */
    [[nodiscard]] auto
    GetGroupId(InnerIdType id) const -> InnerIdType override;

    /**
     * @brief Serializes the tracker to a stream.
     * @param writer The stream writer for output.
     */
    void
    Serialize(StreamWriter& writer) const override;

    /**
     * @brief Deserializes the tracker from a stream.
     * @param reader The stream reader for input.
     */
    void
    Deserialize(StreamReader& reader) override;

    /**
     * @brief Deserializes the tracker from legacy format.
     * @param reader The stream reader for input.
     * @param total_size The total size of the data.
     */
    void
    DeserializeFromLegacyFormat(StreamReader& reader, size_t total_size) override;

    /**
     * @brief Resizes the tracker to accommodate more entries.
     * @param new_size The new size.
     */
    void
    Resize(InnerIdType new_size) override;

private:
    /// Allocator for memory management
    Allocator* allocator_;
    /// Array storing group ID for each entry (maps ID to its group)
    Vector<InnerIdType> duplicate_ids_;
    /// Mutex for thread-safe access
    mutable std::shared_mutex mutex_;
    /// Count of duplicate entries
    size_t duplicate_count_{0};
    /// Flag indicating if deserialization has occurred
    bool has_deserialized_{false};
};

}  // namespace vsag
