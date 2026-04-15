
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
 * @file compressed_graph_datacell.h
 * @brief Compressed graph data cell implementation using Elias-Fano encoding.
 *
 * This file provides the CompressedGraphDataCell class which implements
 * the GraphInterface for storing graph neighbor information in a
 * space-efficient compressed format using Elias-Fano encoding.
 */

#pragma once

#include <shared_mutex>

#include "compressed_graph_datacell_parameter.h"
#include "graph_interface.h"
#include "impl/elias_fano_encoder.h"
#include "io/memory_block_io.h"
#include "sparse_duplicate_tracker.h"

namespace vsag {

/**
 * @brief Compressed graph data cell for storing neighbor relationships efficiently.
 *
 * This class implements GraphInterface and provides functionality for:
 * - Storing graph neighbor information in a compressed format
 * - Using Elias-Fano encoding for space-efficient storage
 * - Supporting standard graph operations with reduced memory footprint
 */
class CompressedGraphDataCell : public GraphInterface {
public:
    /**
     * @brief Constructs a CompressedGraphDataCell with graph interface parameters.
     * @param graph_param The graph interface parameters.
     * @param common_param The common index parameters.
     */
    explicit CompressedGraphDataCell(const GraphInterfaceParamPtr& graph_param,
                                     const IndexCommonParam& common_param);

    /**
     * @brief Constructs a CompressedGraphDataCell with specific parameters.
     * @param graph_param The compressed graph data cell parameters.
     * @param common_param The common index parameters.
     */
    explicit CompressedGraphDataCell(const CompressedGraphDatacellParamPtr& graph_param,
                                     const IndexCommonParam& common_param);

    ~CompressedGraphDataCell();

    /**
     * @brief Inserts neighbor IDs for a given node.
     * @param id The internal ID of the node.
     * @param neighbor_ids The vector of neighbor IDs to insert.
     */
    void
    InsertNeighborsById(InnerIdType id, const Vector<InnerIdType>& neighbor_ids) override;

    /**
     * @brief Gets the number of neighbors for a given node.
     * @param id The internal ID of the node.
     * @return The number of neighbors.
     */
    [[nodiscard]] uint32_t
    GetNeighborSize(InnerIdType id) const override;

    /**
     * @brief Retrieves the neighbor IDs for a given node.
     * @param id The internal ID of the node.
     * @param neighbor_ids Output vector to store the neighbor IDs.
     */
    void
    GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const override;

    /**
     * @brief Checks if a node with the given ID exists.
     * @param id The internal ID to check.
     * @return True if the node exists, false otherwise.
     */
    [[nodiscard]] bool
    CheckIdExists(InnerIdType id) const override;

    /**
     * @brief Resizes the graph to accommodate more nodes.
     * @param new_size The new capacity size.
     */
    void
    Resize(InnerIdType new_size) override;

    /**
     * @brief Prefetch operation (no-op for compressed format).
     * @param id The internal ID of the node.
     * @param neighbor_i The index of neighbor.
     */
    void
    Prefetch(InnerIdType id, uint32_t neighbor_i) override {
    }

    /**
     * @brief Serializes the graph data to a stream.
     * @param writer The stream writer for output.
     */
    void
    Serialize(StreamWriter& writer) override;

    /**
     * @brief Deserializes the graph data from a stream.
     * @param reader The stream reader for input.
     */
    void
    Deserialize(StreamReader& reader) override;

    /**
     * @brief Gets the memory usage of this data cell.
     * @return The memory usage in bytes.
     */
    int64_t
    GetMemoryUsage() const override;

    /**
     * @brief Gets all valid IDs in the graph.
     * @return A vector of all valid internal IDs.
     */
    Vector<InnerIdType>
    GetIds() const override;

    /**
     * @brief Creates a duplicate tracker instance.
     * @return A shared pointer to the sparse duplicate tracker.
     */
    DuplicateTrackerPtr
    CreateDuplicateTracker() override {
        return std::make_shared<SparseDuplicateTracker>(allocator_);
    }

private:
    /// Allocator for memory management
    Allocator* const allocator_{nullptr};

    /// Elias-Fano encoded neighbor sets for each node
    Vector<std::unique_ptr<EliasFanoEncoder>> neighbor_sets_;
};

}  // namespace vsag
