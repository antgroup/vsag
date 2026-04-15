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

/// @file conjugate_graph.h
/// @brief Conjugate graph for result enhancement in vector similarity search.

#pragma once

#include <queue>

#include "impl/logger/logger.h"
#include "storage/footer.h"
#include "typing.h"
#include "vsag/index.h"

namespace vsag {

/// @brief Number of neighbors to consider during result enhancement.
static constexpr int64_t LOOK_AT_K = 20;

/// @brief Maximum degree for conjugate graph edges.
static constexpr int64_t MAXIMUM_DEGREE = 128;

/// @brief Conjugate graph for enhancing search results through tag-based neighbor relationships.
///
/// This class maintains a graph structure where nodes are identified by tag IDs.
/// It supports adding neighbor relationships between tags and using these relationships
/// to enhance search results by expanding the result set with related elements.
class ConjugateGraph {
public:
    /// @brief Constructs a conjugate graph with the specified allocator.
    /// @param allocator Allocator for memory management.
    explicit ConjugateGraph(Allocator* allocator);

    /// @brief Adds a neighbor relationship from one tag to another.
    /// @param from_tag_id Source tag ID.
    /// @param to_tag_id Target tag ID.
    /// @return True if the neighbor was added successfully, Error on failure.
    tl::expected<bool, Error>
    AddNeighbor(int64_t from_tag_id, int64_t to_tag_id);

    /// @brief Enhances search results using the conjugate graph relationships.
    /// @param results Priority queue of (distance, label) pairs to be enhanced.
    /// @param distance_of_tag Function to compute distance for a given tag ID.
    /// @return Number of new elements added to results, Error on failure.
    tl::expected<uint32_t, Error>
    EnhanceResult(std::priority_queue<std::pair<float, LabelType>>& results,
                  const std::function<float(int64_t)>& distance_of_tag) const;

    /// @brief Updates a tag ID to a new value.
    /// @param old_tag_id Old tag ID to be replaced.
    /// @param new_tag_id New tag ID to replace with.
    /// @return True if update was successful, Error on failure.
    tl::expected<bool, Error>
    UpdateId(int64_t old_tag_id, int64_t new_tag_id);

public:
    /// @brief Serializes the conjugate graph to binary format.
    /// @return Binary representation of the graph, Error on failure.
    tl::expected<Binary, Error>
    Serialize() const;

    /// @brief Serializes the conjugate graph to an output stream.
    /// @param out_stream Output stream to write the serialized data.
    /// @return void on success, Error on failure.
    tl::expected<void, Error>
    Serialize(std::ostream& out_stream) const;

    /// @brief Deserializes the conjugate graph from binary data.
    /// @param binary Binary data containing the serialized graph.
    /// @return void on success, Error on failure.
    tl::expected<void, Error>
    Deserialize(const Binary& binary);

    /// @brief Deserializes the conjugate graph from a stream reader.
    /// @param in_stream Stream reader containing the serialized data.
    /// @return void on success, Error on failure.
    tl::expected<void, Error>
    Deserialize(StreamReader& in_stream);

    /// @brief Gets the memory usage of the conjugate graph.
    /// @return Memory usage in bytes.
    uint64_t
    GetMemoryUsage() const;

private:
    /// @brief Gets the neighbor set for a given tag ID.
    /// @param from_tag_id Tag ID to query neighbors for.
    /// @return Shared pointer to set of neighbor tag IDs.
    std::shared_ptr<UnorderedSet<int64_t>>
    get_neighbors(int64_t from_tag_id) const;

    /// @brief Clears all data in the conjugate graph.
    void
    clear();

    /// @brief Checks if the conjugate graph is empty.
    /// @return True if the graph has no edges.
    bool
    is_empty() const;

private:
    /// Memory usage in bytes.
    uint32_t memory_usage_;

    /// Map from tag ID to set of neighbor tag IDs.
    UnorderedMap<int64_t, std::shared_ptr<UnorderedSet<int64_t>>> conjugate_graph_;

    /// Serialization footer for version management.
    SerializationFooter footer_;

    /// Allocator for memory management.
    Allocator* allocator_;
};

}  // namespace vsag