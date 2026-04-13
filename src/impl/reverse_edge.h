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

/// @file reverse_edge.h
/// @brief Reverse edge management for graph-based vector indexes.

#pragma once

#include <mutex>
#include <shared_mutex>

#include "typing.h"
#include "vsag/allocator.h"

namespace vsag {

/// @brief Reverse edge container for tracking incoming edges in graph-based indexes.
///
/// This class maintains reverse edge mappings that allow efficient lookup of
/// all nodes that point to a given node. This is useful for maintaining
/// consistency during edge updates and deletions.
class ReverseEdge {
public:
    /// @brief Constructs a reverse edge container with the specified allocator.
    /// @param allocator Allocator for memory management.
    explicit ReverseEdge(Allocator* allocator) : allocator_(allocator), reverse_edges_(allocator) {
    }

    /// @brief Adds a reverse edge from one node to another.
    /// @param from Source node ID.
    /// @param to Target node ID.
    void
    AddReverseEdge(InnerIdType from, InnerIdType to);

    /// @brief Removes a reverse edge from one node to another.
    /// @param from Source node ID.
    /// @param to Target node ID.
    void
    RemoveReverseEdge(InnerIdType from, InnerIdType to);

    /// @brief Gets all incoming neighbors for a given node.
    /// @param id Node ID to query.
    /// @param neighbors Output vector to store neighbor IDs.
    void
    GetIncomingNeighbors(InnerIdType id, Vector<InnerIdType>& neighbors) const;

    /// @brief Clears all incoming edges for a given node.
    /// @param id Node ID to clear edges for.
    void
    ClearIncomingNeighbors(InnerIdType id);

    /// @brief Clears all reverse edges in the container.
    void
    Clear();

    /// @brief Resizes the internal storage.
    /// @param new_size New size for storage.
    void
    Resize(InnerIdType new_size);

    /// @brief Gets the memory usage of the reverse edge container.
    /// @return Memory usage in bytes.
    int64_t
    GetMemoryUsage() const;

private:
    /// Allocator for memory management.
    Allocator* const allocator_;
    /// Map from node ID to list of nodes pointing to it.
    UnorderedMap<InnerIdType, std::unique_ptr<Vector<InnerIdType>>> reverse_edges_;
    /// Mutex for thread-safe access to reverse_edges_.
    mutable std::shared_mutex mutex_{};
};

}  // namespace vsag