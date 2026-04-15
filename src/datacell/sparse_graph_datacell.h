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
 * @file sparse_graph_datacell.h
 * @brief Sparse graph data cell for managing neighbor relationships in sparse vectors.
 */

#pragma once

#include <shared_mutex>

#include "graph_interface.h"
#include "io/memory_block_io.h"
#include "sparse_duplicate_tracker.h"
#include "sparse_graph_datacell_parameter.h"

namespace vsag {

/**
 * @brief Sparse graph data cell for managing neighbor relationships.
 *
 * This class implements a sparse graph structure for storing and managing
 * neighbor relationships in sparse vector indices. It supports dynamic insertion,
 * deletion, and recovery of neighbor lists with thread-safe operations.
 */
class SparseGraphDataCell : public GraphInterface {
public:
    using NeighborCountsType = uint32_t;

    /**
     * @brief Constructs a sparse graph data cell with graph interface parameter.
     *
     * @param graph_param Graph interface parameter
     * @param common_param Common index parameters
     */
    SparseGraphDataCell(const GraphInterfaceParamPtr& graph_param,
                        const IndexCommonParam& common_param);

    /**
     * @brief Constructs a sparse graph data cell with sparse graph parameter.
     *
     * @param graph_param Sparse graph data cell parameter
     * @param common_param Common index parameters
     */
    SparseGraphDataCell(const SparseGraphDatacellParamPtr& graph_param,
                        const IndexCommonParam& common_param);

    /**
     * @brief Constructs a sparse graph data cell with allocator.
     *
     * @param graph_param Sparse graph data cell parameter
     * @param allocator Memory allocator
     */
    SparseGraphDataCell(const SparseGraphDatacellParamPtr& graph_param, Allocator* allocator);

    /**
     * @brief Inserts neighbor list for a given ID.
     *
     * @param id Inner ID to insert neighbors for
     * @param neighbor_ids Vector of neighbor IDs
     */
    void
    InsertNeighborsById(InnerIdType id, const Vector<InnerIdType>& neighbor_ids) override;

    /**
     * @brief Deletes neighbor list for a given ID.
     *
     * @param id Inner ID to delete neighbors for
     */
    void
    DeleteNeighborsById(InnerIdType id) override;

    /**
     * @brief Recovers previously deleted neighbor list for a given ID.
     *
     * @param id Inner ID to recover neighbors for
     */
    void
    RecoverDeleteNeighborsById(vsag::InnerIdType id) override;

    /**
     * @brief Gets the number of neighbors for a given ID.
     *
     * @param id Inner ID to query
     * @return Number of neighbors
     */
    uint32_t
    GetNeighborSize(InnerIdType id) const override;

    /**
     * @brief Retrieves neighbor list for a given ID.
     *
     * @param id Inner ID to query
     * @param neighbor_ids Output vector to store neighbor IDs
     */
    void
    GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const override;

    /**
     * @brief Checks if an ID exists in the graph.
     *
     * @param id Inner ID to check
     * @return True if the ID exists
     */
    [[nodiscard]] bool
    CheckIdExists(InnerIdType id) const override;

    /**
     * @brief Resizes the graph to a new capacity.
     *
     * @param new_size New capacity size
     */
    void
    Resize(InnerIdType new_size) override;

    /**
     * @brief Prefetches neighbors of a base point with id.
     *
     * @param id Inner ID of the base point
     * @param neighbor_i Index of neighbor, 0 for neighbor size, 1 for first neighbor
     */
    void
    Prefetch(InnerIdType id, uint32_t neighbor_i) override {
        // TODO(LHT): implement
    }

    /**
     * @brief Serializes the graph to a stream.
     *
     * @param writer Stream writer for serialization
     */
    void
    Serialize(StreamWriter& writer) override;

    /**
     * @brief Deserializes the graph from a stream.
     *
     * @param reader Stream reader for deserialization
     */
    void
    Deserialize(StreamReader& reader) override;

    /**
     * @brief Merges another graph into this one.
     *
     * @param other Graph to merge from
     * @param bias ID bias for merging
     */
    void
    MergeOther(GraphInterfacePtr other, uint64_t bias) override;

    /**
     * @brief Gets all IDs in the graph.
     *
     * @return Vector of all inner IDs
     */
    Vector<InnerIdType>
    GetIds() const override;

    /**
     * @brief Gets the memory usage of the graph.
     *
     * @return Memory usage in bytes
     */
    int64_t
    GetMemoryUsage() const override;

    /**
     * @brief Creates a duplicate tracker for the graph.
     *
     * @return Pointer to the created duplicate tracker
     */
    DuplicateTrackerPtr
    CreateDuplicateTracker() override {
        return std::make_shared<SparseDuplicateTracker>(allocator_);
    }

private:
    uint32_t code_line_size_{0};           ///< Size of code line for serialization
    Allocator* const allocator_{nullptr};  ///< Memory allocator
    /// Map from ID to neighbor list
    UnorderedMap<InnerIdType, std::unique_ptr<Vector<InnerIdType>>> neighbors_;
    mutable std::shared_mutex neighbors_map_mutex_{};  ///< Mutex for thread-safe access

    bool is_support_delete_{true};                     ///< Flag indicating delete operation support
    uint32_t remove_flag_bit_{8};                      ///< Number of bits for remove flag
    uint32_t id_bit_{24};                              ///< Number of bits for ID
    uint32_t remove_flag_mask_{0x00ffffff};            ///< Mask for remove flag
    UnorderedMap<InnerIdType, uint8_t> node_version_;  ///< Node version tracking
};

}  // namespace vsag