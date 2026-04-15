
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
 * @file graph_datacell.h
 * @brief Graph data cell implementation for storing and managing neighbor relationships.
 *
 * This file provides the GraphDataCell class which implements the GraphInterface
 * for storing graph neighbor information using IO operations. The graph can be
 * built via nn-descent algorithm or incremental insertion.
 */

#pragma once

#include <limits>
#include <memory>
#include <vector>

#include "algorithm/hnswlib/hnswalg.h"
#include "common.h"
#include "dense_duplicate_tracker.h"
#include "graph_datacell_parameter.h"
#include "graph_interface.h"
#include "graph_interface_parameter.h"
#include "impl/reverse_edge.h"
#include "index_common_param.h"
#include "io/basic_io.h"
#include "vsag/constants.h"

namespace vsag {

template <typename IOTmpl>
class GraphDataCell;

/**
 * @brief Graph data cell for storing and managing neighbor relationships.
 *
 * This class implements the GraphInterface and provides functionality for:
 * - Building graphs via nn-descent or incremental insertion
 * - Adding neighbors with pruning support
 * - Retrieving neighbors efficiently
 * - Supporting delete and recovery operations
 *
 * @tparam IOTmpl The IO template type for storage operations.
 */
template <typename IOTmpl>
class GraphDataCell : public GraphInterface {
public:
    /**
     * @brief Constructs a GraphDataCell with graph interface parameters.
     * @param graph_param The graph interface parameters.
     * @param common_param The common index parameters.
     */
    explicit GraphDataCell(const GraphInterfaceParamPtr& graph_param,
                           const IndexCommonParam& common_param);

    /**
     * @brief Constructs a GraphDataCell with graph data cell parameters.
     * @param graph_param The graph data cell parameters.
     * @param common_param The common index parameters.
     */
    explicit GraphDataCell(const GraphDataCellParamPtr& graph_param,
                           const IndexCommonParam& common_param);

    /**
     * @brief Inserts neighbor IDs for a given node.
     * @param id The internal ID of the node.
     * @param neighbor_ids The vector of neighbor IDs to insert.
     */
    void
    InsertNeighborsById(InnerIdType id, const Vector<InnerIdType>& neighbor_ids) override;

    /**
     * @brief Marks neighbors of a node as deleted.
     * @param id The internal ID of the node to delete.
     */
    void
    DeleteNeighborsById(vsag::InnerIdType id) override;

    /**
     * @brief Recovers previously deleted neighbors of a node.
     * @param id The internal ID of the node to recover.
     */
    void
    RecoverDeleteNeighborsById(vsag::InnerIdType id) override;

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
    CheckIdExists(InnerIdType id) const override {
        return id < this->total_count_ && id < this->max_capacity_;
    }

    /**
     * @brief Resizes the graph to accommodate more nodes.
     * @param new_size The new capacity size.
     */
    void
    Resize(InnerIdType new_size) override;

    /**
     * @brief Sets the IO instance for storage operations.
     * @param io The shared pointer to the IO instance.
     */
    inline void
    SetIO(std::shared_ptr<BasicIO<IOTmpl>> io) {
        this->io_ = io;
    }

    /**
     * @brief Initializes the IO with the given parameters.
     * @param io_param The IO parameters.
     */
    virtual void
    InitIO(const IOParamPtr& io_param) override {
        this->io_->InitIO(io_param);
    }

    /**
     * @brief Prefetches neighbors of a base point for cache optimization.
     * @param id The internal ID of the base point.
     * @param neighbor_i The index of neighbor (0 for neighbor size, 1 for first neighbor).
     */
    void
    Prefetch(InnerIdType id, uint32_t neighbor_i) override {
        io_->Prefetch(static_cast<uint64_t>(id) * static_cast<uint64_t>(this->code_line_size_) +
                      sizeof(uint32_t) + neighbor_i * sizeof(InnerIdType));
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
     * @brief Checks if the graph data is stored in memory.
     * @return True if stored in memory, false otherwise.
     */
    bool
    InMemory() const override {
        return IOTmpl::InMemory;
    }

    /**
     * @brief Merges another graph interface into this one.
     * @param other The other graph interface to merge.
     * @param bias The ID bias for merging.
     */
    void
    MergeOther(GraphInterfacePtr other, uint64_t bias) override;

    /**
     * @brief Gets the memory usage of this data cell.
     * @return The memory usage in bytes.
     */
    int64_t
    GetMemoryUsage() const override {
        int64_t memory = sizeof(GraphDataCell) + node_versions_.size() * sizeof(uint8_t);
        if (IOTmpl::InMemory) {
            memory += io_->GetMemoryUsage();
        }
        if (reverse_edges_) {
            memory += reverse_edges_->GetMemoryUsage();
        }
        return memory;
    }

    /**
     * @brief Gets the incoming neighbors for a given node.
     * @param id The internal ID of the node.
     * @param neighbors Output vector to store the incoming neighbor IDs.
     */
    void
    GetIncomingNeighbors(InnerIdType id, Vector<InnerIdType>& neighbors) const override {
        if (reverse_edges_) {
            reverse_edges_->GetIncomingNeighbors(id, neighbors);
        } else {
            neighbors.clear();
        }
    }

    /**
     * @brief Creates a duplicate tracker instance.
     * @return A shared pointer to the duplicate tracker.
     */
    DuplicateTrackerPtr
    CreateDuplicateTracker() override {
        return std::make_shared<DenseDuplicateTracker>(allocator_);
    }

private:
    /// IO instance for storage operations
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};

    /// Reverse edge storage for incoming neighbor lookup
    std::unique_ptr<ReverseEdge> reverse_edges_{nullptr};

    /// Version tracking for each node (used for delete support)
    Vector<uint8_t> node_versions_;

    /// Flag indicating whether delete operations are supported
    bool is_support_delete_{true};
    /// Number of bits used for remove flag
    uint32_t remove_flag_bit_{8};
    /// Number of bits used for ID storage
    uint32_t id_bit_{24};
    /// Mask for extracting ID from combined value
    uint32_t remove_flag_mask_{0x00FFFFFF};

    /// Size of one code line (neighbors + count)
    uint32_t code_line_size_{0};
};

template <typename IOTmpl>
void
GraphDataCell<IOTmpl>::MergeOther(GraphInterfacePtr other, uint64_t bias) {
    auto other_graph = std::dynamic_pointer_cast<GraphDataCell<IOTmpl>>(other);
    if (!other_graph) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "GraphDataCell can only merge with GraphDataCell");
    }
    if (this->maximum_degree_ != other_graph->maximum_degree_) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("GraphDataCell maximum degree mismatch: {} vs {}",
                                        this->maximum_degree_,
                                        other_graph->maximum_degree_));
    }
    Vector<InnerIdType> neighbor_ids(allocator_);
    if (is_support_delete_) {
        for (int i = 0; i < other_graph->total_count_; ++i) {
            node_versions_[i + bias] = other_graph->node_versions_[i];
        }
    }
    for (int i = 0; i < other_graph->total_count_; ++i) {
        other_graph->GetNeighbors(i, neighbor_ids);
        for (auto& neighbor_id : neighbor_ids) {
            neighbor_id += bias;
        }
        this->InsertNeighborsById(i + bias, neighbor_ids);
    }
}

template <typename IOTmpl>
GraphDataCell<IOTmpl>::GraphDataCell(const GraphDataCellParamPtr& param,
                                     const IndexCommonParam& common_param)
    : node_versions_(common_param.allocator_.get()) {
    this->io_ = std::make_shared<IOTmpl>(param->io_parameter_, common_param);
    this->maximum_degree_ = param->max_degree_;
    this->max_capacity_ = param->init_max_capacity_;
    this->is_support_delete_ = param->support_remove_;
    this->remove_flag_bit_ = param->remove_flag_bit_;
    this->id_bit_ = sizeof(InnerIdType) * 8 - this->remove_flag_bit_;
    this->remove_flag_mask_ = (1 << this->id_bit_) - 1;
    this->code_line_size_ = this->maximum_degree_ * sizeof(InnerIdType) + sizeof(uint32_t);
    this->allocator_ = common_param.allocator_.get();
    if (this->is_support_delete_) {
        node_versions_.resize(max_capacity_);
    }
    if (param->use_reverse_edges_) {
        reverse_edges_ = std::make_unique<ReverseEdge>(this->allocator_);
    }

    if (param->support_duplicate_) {
        this->InitDuplicateTracker();
    }
}

template <typename IOTmpl>
GraphDataCell<IOTmpl>::GraphDataCell(const GraphInterfaceParamPtr& param,
                                     const IndexCommonParam& common_param)
    : GraphDataCell<IOTmpl>(std::dynamic_pointer_cast<GraphDataCellParameter>(param),
                            common_param) {
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl>::InsertNeighborsById(InnerIdType id,
                                           const Vector<InnerIdType>& neighbor_ids) {
    if (neighbor_ids.size() > this->maximum_degree_) {
        throw std::invalid_argument(fmt::format(
            "insert neighbors count {} more than {}", neighbor_ids.size(), this->maximum_degree_));
    }

    // Update reverse edges if enabled
    if (reverse_edges_) {
        Vector<InnerIdType> old_neighbors(allocator_);
        if (id < this->total_count_) {
            this->GetNeighbors(id, old_neighbors);
        }
        UnorderedSet<InnerIdType> old_set(allocator_);
        UnorderedSet<InnerIdType> new_set(allocator_);
        for (const auto& n : old_neighbors) {
            old_set.insert(n);
        }
        for (const auto& n : neighbor_ids) {
            new_set.insert(n);
        }
        for (const auto& old_n : old_neighbors) {
            if (new_set.find(old_n) == new_set.end()) {
                reverse_edges_->RemoveReverseEdge(id, old_n);
            }
        }
        for (const auto& new_n : neighbor_ids) {
            if (old_set.find(new_n) == old_set.end()) {
                reverse_edges_->AddReverseEdge(id, new_n);
            }
        }
    }

    InnerIdType current = total_count_.load();
    while (current < id + 1 && !total_count_.compare_exchange_weak(current, id + 1)) {
    }
    auto start = static_cast<uint64_t>(id) * static_cast<uint64_t>(this->code_line_size_);
    if (is_support_delete_) {
        uint32_t neighbor_count = std::min((uint32_t)(neighbor_ids.size()), this->maximum_degree_);
        this->io_->Write((uint8_t*)(&neighbor_count), sizeof(neighbor_count), start);
        start += sizeof(neighbor_count);
        Vector<InnerIdType> neighbor_ids_ptr(neighbor_ids.size(), 0, this->allocator_);
        for (int i = 0; i < neighbor_ids.size(); ++i) {
            auto neighbor_id = neighbor_ids[i];
            neighbor_ids_ptr[i] = neighbor_id | (node_versions_[neighbor_id] << id_bit_);
        }
        this->io_->Write((uint8_t*)(neighbor_ids_ptr.data()),
                         static_cast<uint64_t>(neighbor_count) * sizeof(InnerIdType),
                         start);
    } else {
        uint32_t neighbor_count = std::min((uint32_t)(neighbor_ids.size()), this->maximum_degree_);
        this->io_->Write((uint8_t*)(&neighbor_count), sizeof(neighbor_count), start);
        start += sizeof(neighbor_count);
        this->io_->Write((uint8_t*)(neighbor_ids.data()),
                         static_cast<uint64_t>(neighbor_count) * sizeof(InnerIdType),
                         start);
    }
}

template <typename IOTmpl>
uint32_t
GraphDataCell<IOTmpl>::GetNeighborSize(InnerIdType id) const {
    auto start = static_cast<uint64_t>(id) * static_cast<uint64_t>(this->code_line_size_);
    uint32_t neighbor_count = 0;
    this->io_->Read(sizeof(neighbor_count), start, (uint8_t*)(&neighbor_count));
    return neighbor_count;
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl>::GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const {
    auto start = static_cast<uint64_t>(id) * static_cast<uint64_t>(this->code_line_size_);
    uint32_t neighbor_count = 0;
    this->io_->Read(sizeof(neighbor_count), start, (uint8_t*)(&neighbor_count));
    if (is_support_delete_) {
        neighbor_count &= remove_flag_mask_;
        start += sizeof(neighbor_count);
        Vector<InnerIdType> shared_neighbor_ids(neighbor_count, this->allocator_);
        this->io_->Read(
            neighbor_count * sizeof(InnerIdType), start, (uint8_t*)(shared_neighbor_ids.data()));
        neighbor_ids.clear();
        neighbor_ids.reserve(neighbor_count);
        for (int i = 0; i < neighbor_count; ++i) {
            uint8_t neighbor_version = shared_neighbor_ids[i] >> id_bit_;
            InnerIdType neighbor_id = shared_neighbor_ids[i] & remove_flag_mask_;
            if (node_versions_[neighbor_id] == neighbor_version) {
                neighbor_ids.push_back(neighbor_id);
            }
        }
    } else {
        start += sizeof(neighbor_count);
        neighbor_ids.resize(neighbor_count);
        this->io_->Read(
            neighbor_ids.size() * sizeof(InnerIdType), start, (uint8_t*)(neighbor_ids.data()));
    }
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl>::Resize(InnerIdType new_size) {
    if (new_size < this->max_capacity_) {
        return;
    }
    if (is_support_delete_) {
        if (new_size > remove_flag_mask_) {
            // remove_flag_mask_ exactly matches the maximum size of the graph in dynamic mode.
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("the size of graph is limit ({})", remove_flag_mask_));
        }
        node_versions_.resize(new_size);
    }
    uint64_t io_size = static_cast<uint64_t>(new_size) * static_cast<uint64_t>(code_line_size_);
    this->io_->Resize(io_size);
    this->max_capacity_ = new_size;
    if (this->duplicate_tracker_ != nullptr) {
        this->duplicate_tracker_->Resize(new_size);
    }
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl>::Serialize(StreamWriter& writer) {
    GraphInterface::Serialize(writer);
    this->io_->Serialize(writer);
    StreamWriter::WriteObj(writer, this->code_line_size_);
    if (is_support_delete_) {
        StreamWriter::WriteVector(writer, node_versions_);
    }
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl>::Deserialize(StreamReader& reader) {
    GraphInterface::Deserialize(reader);
    this->io_->Deserialize(reader);
    StreamReader::ReadObj(reader, this->code_line_size_);
    if (is_support_delete_) {
        StreamReader::ReadVector(reader, node_versions_);
    }
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl>::DeleteNeighborsById(vsag::InnerIdType id) {
    if (is_support_delete_) {
        if (id <= max_capacity_) {
            if (node_versions_[id] + 1 == 0) {
                throw VsagException(
                    ErrorType::INTERNAL_ERROR,
                    "remove point too many times in GraphDatacell, please rebuild index");
            }
        } else {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("remove point {} not exist in GraphDatacell", id));
        }
        node_versions_[id]++;
    } else {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "disable delete in graph datacell");
    }
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl>::RecoverDeleteNeighborsById(vsag::InnerIdType id) {
    if (is_support_delete_) {
        if (id <= max_capacity_) {
            if (node_versions_[id] == 0) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "recover remove point too many times in GraphDatacell");
            }
        } else {
            throw VsagException(
                ErrorType::INTERNAL_ERROR,
                fmt::format("recover remove point {} not exist in GraphDatacell", id));
        }
        node_versions_[id]--;
    } else {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "disable delete in graph datacell");
    }
}

}  // namespace vsag
