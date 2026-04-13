/**
 * @file duplicate_interface.h
 * @brief Duplicate interface for tracking duplicate vector IDs.
 *
 * This file defines the abstract interface for tracking and managing
 * duplicate vector IDs in the index.
 */

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

#include <memory>
#include <vector>

#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

DEFINE_POINTER(DuplicateInterface);

/**
 * @brief Abstract interface for tracking duplicate vector IDs.
 *
 * DuplicateInterface provides operations for managing groups of duplicate
 * vector IDs, allowing the index to track which vectors are duplicates
 * of each other.
 */
class DuplicateInterface {
public:
    virtual ~DuplicateInterface() = default;

    /**
     * @brief Sets a duplicate ID for a group.
     *
     * Associates a duplicate ID with a group, enabling tracking of
     * multiple vectors that represent the same data.
     *
     * @param group_id The group ID (representative ID).
     * @param duplicate_id The duplicate ID to associate with the group.
     */
    virtual void
    SetDuplicateId(InnerIdType group_id, InnerIdType duplicate_id) = 0;

    /**
     * @brief Gets all duplicate IDs for a given ID.
     *
     * @param id The internal ID to query.
     * @return Vector of all duplicate IDs associated with this ID's group.
     */
    [[nodiscard]] virtual auto
    GetDuplicateIds(InnerIdType id) const -> std::vector<InnerIdType> = 0;

    /**
     * @brief Gets the group ID for a given ID.
     *
     * Returns the representative ID of the group that contains the given ID.
     *
     * @param id The internal ID to query.
     * @return The group ID (representative ID).
     */
    [[nodiscard]] virtual auto
    GetGroupId(InnerIdType id) const -> InnerIdType = 0;

    /**
     * @brief Serializes the duplicate tracker to a stream writer.
     *
     * @param writer The stream writer for output.
     */
    virtual void
    Serialize(StreamWriter& writer) const = 0;

    /**
     * @brief Deserializes the duplicate tracker from a stream reader.
     *
     * @param reader The stream reader for input.
     */
    virtual void
    Deserialize(StreamReader& reader) = 0;

    /**
     * @brief Deserializes from legacy format.
     *
     * Used for backward compatibility with older serialization formats.
     *
     * @param reader The stream reader for input.
     * @param total_size Total size of the legacy data.
     */
    virtual void
    DeserializeFromLegacyFormat(StreamReader& reader, size_t total_size) = 0;

    /**
     * @brief Resizes the duplicate tracker to a new capacity.
     *
     * @param new_size The new capacity.
     */
    virtual void
    Resize(InnerIdType new_size) = 0;
};

using DuplicateTrackerPtr = std::shared_ptr<DuplicateInterface>;

}  // namespace vsag