/**
 * @file graph_interface.h
 * @brief Graph interface for managing neighbor relationships in vector index.
 *
 * This file defines the abstract interface for graph-based data structures
 * used in approximate nearest neighbor search algorithms like HNSW.
 */

// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance the License.
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

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "duplicate_interface.h"
#include "graph_interface_parameter.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "io/io_parameter.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

class ReverseEdge;
DEFINE_POINTER(GraphInterface);

/**
 * @brief Abstract interface for graph-based neighbor management.
 *
 * GraphInterface provides the core operations for managing neighbor relationships
 * in graph-based vector indices. It supports operations like inserting, deleting,
 * and querying neighbors, as well as serialization and duplicate tracking.
 */
class GraphInterface {
public:
    GraphInterface() = default;

    virtual ~GraphInterface() = default;

    /**
     * @brief Creates a GraphInterface instance based on parameters.
     *
     * @param graph_param Graph interface configuration parameters.
     * @param common_param Common index parameters.
     * @return Shared pointer to the created GraphInterface instance.
     */
    static GraphInterfacePtr
    MakeInstance(const GraphInterfaceParamPtr& graph_param, const IndexCommonParam& common_param);

public:
    /**
     * @brief Inserts neighbor IDs for a given node.
     *
     * @param id The internal ID of the node.
     * @param neighbor_ids Vector of neighbor IDs to insert.
     */
    virtual void
    InsertNeighborsById(InnerIdType id, const Vector<InnerIdType>& neighbor_ids) = 0;

    /**
     * @brief Deletes all neighbors of a node.
     *
     * @param id The internal ID of the node whose neighbors should be deleted.
     */
    virtual void
    DeleteNeighborsById(InnerIdType id) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "DeleteNeighborsById is not implemented");
    }

    /**
     * @brief Recovers previously deleted neighbors of a node.
     *
     * @param id The internal ID of the node whose neighbors should be recovered.
     */
    virtual void
    RecoverDeleteNeighborsById(InnerIdType id) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "RecoverDeleteNeighborsById is not implemented");
    }

    /**
     * @brief Gets the number of neighbors for a given node.
     *
     * @param id The internal ID of the node.
     * @return Number of neighbors.
     */
    virtual uint32_t
    GetNeighborSize(InnerIdType id) const = 0;

    /**
     * @brief Retrieves all neighbor IDs for a given node.
     *
     * @param id The internal ID of the node.
     * @param neighbor_ids Output vector to store neighbor IDs.
     */
    virtual void
    GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const = 0;

    /**
     * @brief Checks if a node with the given ID exists.
     *
     * @param id The internal ID to check.
     * @return True if the node exists, false otherwise.
     */
    [[nodiscard]] virtual bool
    CheckIdExists(InnerIdType id) const = 0;

    /**
     * @brief Resizes the graph to accommodate a new capacity.
     *
     * @param new_size The new capacity for the graph.
     */
    virtual void
    Resize(InnerIdType new_size) = 0;

    /**
     * @brief Prefetches neighbor data for cache optimization.
     *
     * @param id The internal ID of the node.
     * @param neighbor_i Index of the neighbor to prefetch.
     */
    virtual void
    Prefetch(InnerIdType id, uint32_t neighbor_i) = 0;

    /**
     * @brief Merges another graph into this one.
     *
     * @param other The graph to merge.
     * @param bias ID offset for the merged graph nodes.
     */
    virtual void
    MergeOther(GraphInterfacePtr other, uint64_t bias) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "MergeOther in GraphInterface is not implemented");
    }

    /**
     * @brief Gets all valid node IDs in the graph.
     *
     * @return Vector of all valid internal IDs.
     */
    virtual Vector<InnerIdType>
    GetIds() const {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "GetIds in GraphInterface is not implemented");
    }

    /**
     * @brief Gets the memory usage of the graph.
     *
     * @return Memory usage in bytes.
     */
    virtual int64_t
    GetMemoryUsage() const {
        return 0;
    }

    /**
     * @brief Gets incoming neighbors for a node (reverse edges).
     *
     * @param id The internal ID of the node.
     * @param neighbors Output vector to store incoming neighbor IDs.
     */
    virtual void
    GetIncomingNeighbors(InnerIdType id, Vector<InnerIdType>& neighbors) const {
        neighbors.clear();
    }

public:
    /**
     * @brief Serializes the graph to a stream writer.
     *
     * @param writer The stream writer for output.
     */
    virtual void
    Serialize(StreamWriter& writer) {
        StreamWriter::WriteObj(writer, this->total_count_);
        StreamWriter::WriteObj(writer, this->max_capacity_);
        StreamWriter::WriteObj(writer, this->maximum_degree_);

        if (duplicate_tracker_) {
            duplicate_tracker_->Serialize(writer);
        }
    }

    /**
     * @brief Deserializes the graph from a stream reader.
     *
     * @param reader The stream reader for input.
     */
    virtual void
    Deserialize(StreamReader& reader) {
        StreamReader::ReadObj(reader, this->total_count_);
        StreamReader::ReadObj(reader, this->max_capacity_);
        StreamReader::ReadObj(reader, this->maximum_degree_);

        if (duplicate_tracker_) {
            duplicate_tracker_->Deserialize(reader);
        }
    }

    /**
     * @brief Calculates the serialized size of the graph.
     *
     * @return Size in bytes needed for serialization.
     */
    uint64_t
    CalcSerializeSize() {
        auto calSizeFunc = [](uint64_t cursor, uint64_t size, void* buf) { return; };
        WriteFuncStreamWriter writer(calSizeFunc, 0);
        this->Serialize(writer);
        return writer.cursor_;
    }

    /**
     * @brief Gets the total count of nodes in the graph.
     *
     * @return Total number of nodes.
     */
    [[nodiscard]] virtual InnerIdType
    TotalCount() const {
        return this->total_count_;
    }

    /**
     * @brief Gets the maximum degree (max neighbors per node).
     *
     * @return Maximum degree value.
     */
    [[nodiscard]] virtual InnerIdType
    MaximumDegree() const {
        return this->maximum_degree_;
    }

    /**
     * @brief Gets the maximum capacity of the graph.
     *
     * @return Maximum capacity value.
     */
    [[nodiscard]] virtual InnerIdType
    MaxCapacity() const {
        return this->max_capacity_;
    }

    /**
     * @brief Checks if the graph is stored in memory.
     *
     * @return True if in-memory, false otherwise.
     */
    [[nodiscard]] virtual bool
    InMemory() const {
        return true;
    }

    /**
     * @brief Sets the maximum degree for the graph.
     *
     * @param maximum_degree The new maximum degree value.
     */
    virtual void
    SetMaximumDegree(uint32_t maximum_degree) {
        this->maximum_degree_ = maximum_degree;
    }

    /**
     * @brief Sets the total count of nodes.
     *
     * @param total_count The new total count value.
     */
    virtual void
    SetTotalCount(InnerIdType total_count) {
        this->total_count_ = total_count;
    };

    /**
     * @brief Sets the maximum capacity of the graph.
     *
     * @param capacity The new capacity value.
     */
    virtual void
    SetMaxCapacity(InnerIdType capacity) {
        this->max_capacity_ = std::max(capacity, this->total_count_.load());
    };

    /**
     * @brief Initializes IO resources for disk-based storage.
     *
     * @param io_param IO configuration parameters.
     */
    virtual void
    InitIO(const IOParamPtr& io_param) {
    }

public:
    /**
     * @brief Creates a duplicate tracker instance.
     *
     * @return Shared pointer to the duplicate tracker, or nullptr if not supported.
     */
    virtual DuplicateTrackerPtr
    CreateDuplicateTracker() {
        return nullptr;
    }

    /**
     * @brief Initializes the duplicate tracker.
     */
    void
    InitDuplicateTracker() {
        duplicate_tracker_ = CreateDuplicateTracker();
        if (duplicate_tracker_ != nullptr) {
            duplicate_tracker_->Resize(max_capacity_);
        }
    }

    /**
     * @brief Gets the duplicate tracker.
     *
     * @return Shared pointer to the duplicate tracker.
     */
    DuplicateTrackerPtr
    GetDuplicateTracker() const {
        return duplicate_tracker_;
    }

    /**
     * @brief Sets the duplicate tracker.
     *
     * @param tracker The duplicate tracker to set.
     */
    void
    SetDuplicateTracker(DuplicateTrackerPtr tracker) {
        duplicate_tracker_ = tracker;
    }

    /**
     * @brief Sets a duplicate ID for a group.
     *
     * @param group_id The group ID.
     * @param duplicate_id The duplicate ID to associate.
     */
    void
    SetDuplicateId(InnerIdType group_id, InnerIdType duplicate_id) {
        if (duplicate_tracker_) {
            duplicate_tracker_->SetDuplicateId(group_id, duplicate_id);
        }
    }

    /**
     * @brief Gets all duplicate IDs for a given ID.
     *
     * @param id The internal ID to query.
     * @return Vector of duplicate IDs.
     */
    std::vector<InnerIdType>
    GetDuplicateIds(InnerIdType id) const {
        if (duplicate_tracker_) {
            return duplicate_tracker_->GetDuplicateIds(id);
        }
        return {};
    }

    /**
     * @brief Gets the group ID for a given ID.
     *
     * @param id The internal ID to query.
     * @return The group ID (returns id if no duplicate tracker).
     */
    [[nodiscard]] InnerIdType
    GetGroupId(InnerIdType id) const {
        if (duplicate_tracker_) {
            return duplicate_tracker_->GetGroupId(id);
        }
        return id;
    }

public:
    /// Maximum capacity of the graph
    InnerIdType max_capacity_{100};
    /// Maximum number of neighbors per node
    uint32_t maximum_degree_{0};

    /// Total count of nodes in the graph
    std::atomic<InnerIdType> total_count_{0};
    /// Allocator for memory management
    Allocator* allocator_{nullptr};

protected:
    /// Tracker for duplicate vector IDs
    DuplicateTrackerPtr duplicate_tracker_{nullptr};
};

}  // namespace vsag