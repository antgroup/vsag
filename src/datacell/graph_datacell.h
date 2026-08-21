
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

#include <limits>
#include <memory>
#include <vector>

#include "common.h"
#include "dense_duplicate_tracker.h"
#include "graph_datacell_parameter.h"
#include "graph_interface.h"
#include "graph_interface_parameter.h"
#include "impl/reverse_edge.h"
#include "index_common_param.h"
#include "io/common/basic_io.h"
#include "layout/fixed_layout.h"
#include "vsag/constants.h"

namespace vsag {

/**
 * built by nn-descent or incremental insertion
 * add neighbors and pruning
 * retrieve neighbors
 */
template <typename IOTmpl>
class GraphDataCell;

template <typename IOTmpl>
class GraphDataCell : public GraphInterface {
public:
    explicit GraphDataCell(const GraphInterfaceParamPtr& graph_param,
                           const IndexCommonParam& common_param);

    explicit GraphDataCell(const GraphDataCellParamPtr& graph_param,
                           const IndexCommonParam& common_param);

    void
    InsertNeighborsById(InnerIdType id, const Vector<InnerIdType>& neighbor_ids) override;

    void
    DeleteNeighborsById(vsag::InnerIdType id) override;

    void
    RecoverDeleteNeighborsById(vsag::InnerIdType id) override;

    [[nodiscard]] uint32_t
    GetNeighborSize(InnerIdType id) const override;

    void
    GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const override;

    [[nodiscard]] bool
    CheckIdExists(InnerIdType id) const override {
        return id < this->total_count_ && id < this->max_capacity_;
    }

    void
    Resize(InnerIdType new_size) override;

    inline void
    SetIO(std::shared_ptr<BasicIO<IOTmpl>> io) {
        this->layout_.SetIO(std::move(io));
    }

    virtual void
    InitIO(const IOParamPtr& io_param) override {
        this->layout_.InitIO(io_param);
    }

    /****
     * prefetch neighbors of a base point with id
     * @param id of base point
     * @param neighbor_i index of neighbor, 0 for neighbor size, 1 for first neighbor
     */
    void
    Prefetch(InnerIdType id, uint32_t neighbor_i) override {
        const uint64_t offset_in_record =
            NEIGHBORS_OFFSET + static_cast<uint64_t>(neighbor_i) * sizeof(InnerIdType);
        layout_.PrefetchAt(id, offset_in_record, 64);
    }

    void
    Serialize(StreamWriter& writer) override;

    [[nodiscard]] uint64_t
    GetIOSize() const override {
        // any io whose Write persists the bytes (in-memory or file-backed)
        // can be pre-allocated by ReserveIO and filled concurrently by
        // WriteRaw; only SkipDeserialize ios (e.g. reader_io, whose Write is
        // a no-op counter) opt out and fall back to the whole-component path
        if constexpr (not IOTmpl::SkipDeserialize) {
            return this->io_->size_;
        } else {
            return 0;
        }
    }

    uint64_t
    ReserveIO(StreamReader& reader) override {
        if constexpr (not IOTmpl::SkipDeserialize) {
            GraphInterface::Deserialize(reader);
            uint64_t io_size = 0;
            StreamReader::ReadObj(reader, io_size);
            this->io_->ResizeForOverwrite(io_size);
            return io_size;
        } else {
            return GraphInterface::ReserveIO(reader);
        }
    }

    void
    WriteRaw(const uint8_t* data, uint64_t size, uint64_t offset) override {
        if constexpr (not IOTmpl::SkipDeserialize) {
            this->io_->Write(data, size, offset);
        } else {
            GraphInterface::WriteRaw(data, size, offset);
        }
    }

    void
    DeserializeTail(StreamReader& reader) override {
        if constexpr (not IOTmpl::SkipDeserialize) {
            StreamReader::ReadObj(reader, this->code_line_size_);
            if (is_support_delete_) {
                StreamReader::ReadVector(reader, node_versions_);
            }
        } else {
            GraphInterface::DeserializeTail(reader);
        }
    }

    void
    Deserialize(StreamReader& reader) override;

    bool
    InMemory() const override {
        return IOTmpl::InMemory;
    }

    void
    MergeOther(GraphInterfacePtr other, uint64_t bias) override;

    uint64_t
    GetMemoryUsage() const override {
        uint64_t memory = sizeof(GraphDataCell) + node_versions_.size() * sizeof(uint8_t);
        memory += layout_.GetMemoryUsage();
        if (reverse_edges_) {
            memory += reverse_edges_->GetMemoryUsage();
        }
        return memory;
    }

    void
    GetIncomingNeighbors(InnerIdType id, Vector<InnerIdType>& neighbors) const override {
        if (reverse_edges_) {
            reverse_edges_->GetIncomingNeighbors(id, neighbors);
        } else {
            neighbors.clear();
        }
    }

    DuplicateTrackerPtr
    CreateDuplicateTracker() override {
        return std::make_shared<DenseDuplicateTracker>(allocator_);
    }

    void
    Move(InnerIdType from, InnerIdType to) override;

    void
    ShrinkToFit(InnerIdType capacity) override {
        this->layout_.Shrink(capacity);
        if (is_support_delete_) {
            node_versions_.resize(capacity);
        }
        this->max_capacity_ = capacity;
    }

protected:
    static constexpr uint64_t COUNT_OFFSET = 0;
    static constexpr uint64_t NEIGHBORS_OFFSET = sizeof(uint32_t);

    FixedLayout<IOTmpl> layout_{};

    Vector<uint8_t> node_versions_;

    bool is_support_delete_{true};
    uint32_t remove_flag_bit_{8};
    uint32_t id_bit_{24};
    uint32_t remove_flag_mask_{0x00FFFFFF};

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
    this->maximum_degree_ = param->max_degree_;
    this->max_capacity_ = param->init_max_capacity_;
    this->is_support_delete_ = param->support_remove_;
    this->remove_flag_bit_ = param->remove_flag_bit_;
    this->id_bit_ = sizeof(InnerIdType) * 8 - this->remove_flag_bit_;
    this->remove_flag_mask_ = (1 << this->id_bit_) - 1;
    this->code_line_size_ = this->maximum_degree_ * sizeof(InnerIdType) + sizeof(uint32_t);
    this->layout_.SetCodeSize(this->code_line_size_);
    this->layout_.SetIO(std::make_shared<IOTmpl>(param->io_parameter_, common_param));
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
    Vector<InnerIdType> old_neighbors(allocator_);
    if (reverse_edges_ && id < this->total_count_) {
        this->GetNeighbors(id, old_neighbors);
    }
    UpdateReverseEdges(id, old_neighbors, neighbor_ids);

    InnerIdType current = total_count_.load();
    while (current < id + 1 && !total_count_.compare_exchange_weak(current, id + 1)) {
    }
    if (is_support_delete_) {
        uint32_t neighbor_count = std::min((uint32_t)(neighbor_ids.size()), this->maximum_degree_);
        this->layout_.WriteAt(id,
                              COUNT_OFFSET,
                              reinterpret_cast<const uint8_t*>(&neighbor_count),
                              sizeof(neighbor_count));
        Vector<InnerIdType> neighbor_ids_ptr(neighbor_ids.size(), 0, this->allocator_);
        for (int i = 0; i < neighbor_ids.size(); ++i) {
            auto neighbor_id = neighbor_ids[i];
            neighbor_ids_ptr[i] = neighbor_id | (node_versions_[neighbor_id] << id_bit_);
        }
        this->layout_.WriteAt(id,
                              NEIGHBORS_OFFSET,
                              reinterpret_cast<const uint8_t*>(neighbor_ids_ptr.data()),
                              static_cast<uint64_t>(neighbor_count) * sizeof(InnerIdType));
    } else {
        uint32_t neighbor_count = std::min((uint32_t)(neighbor_ids.size()), this->maximum_degree_);
        this->layout_.WriteAt(id,
                              COUNT_OFFSET,
                              reinterpret_cast<const uint8_t*>(&neighbor_count),
                              sizeof(neighbor_count));
        this->layout_.WriteAt(id,
                              NEIGHBORS_OFFSET,
                              reinterpret_cast<const uint8_t*>(neighbor_ids.data()),
                              static_cast<uint64_t>(neighbor_count) * sizeof(InnerIdType));
    }
}

template <typename IOTmpl>
uint32_t
GraphDataCell<IOTmpl>::GetNeighborSize(InnerIdType id) const {
    uint32_t neighbor_count = 0;
    this->layout_.ReadAt(
        id, COUNT_OFFSET, sizeof(neighbor_count), reinterpret_cast<uint8_t*>(&neighbor_count));
    return neighbor_count;
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl>::GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const {
    uint32_t neighbor_count = 0;
    this->layout_.ReadAt(
        id, COUNT_OFFSET, sizeof(neighbor_count), reinterpret_cast<uint8_t*>(&neighbor_count));
    if (neighbor_count > this->maximum_degree_) {
        neighbor_ids.clear();
        return;
    }
    if (is_support_delete_) {
        neighbor_count &= remove_flag_mask_;
        Vector<InnerIdType> shared_neighbor_ids(neighbor_count, this->allocator_);
        this->layout_.ReadAt(id,
                             NEIGHBORS_OFFSET,
                             static_cast<uint64_t>(neighbor_count) * sizeof(InnerIdType),
                             reinterpret_cast<uint8_t*>(shared_neighbor_ids.data()));
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
        neighbor_ids.resize(neighbor_count);
        this->layout_.ReadAt(id,
                             NEIGHBORS_OFFSET,
                             static_cast<uint64_t>(neighbor_ids.size()) * sizeof(InnerIdType),
                             reinterpret_cast<uint8_t*>(neighbor_ids.data()));
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
    this->layout_.Resize(new_size);
    this->max_capacity_ = new_size;
    if (this->duplicate_tracker_ != nullptr) {
        this->duplicate_tracker_->Resize(new_size);
    }
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl>::Serialize(StreamWriter& writer) {
    GraphInterface::Serialize(writer);
    this->layout_.Serialize(writer);
    StreamWriter::WriteObj(writer, this->code_line_size_);
    if (is_support_delete_) {
        StreamWriter::WriteVector(writer, node_versions_);
    }
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl>::Deserialize(StreamReader& reader) {
    GraphInterface::Deserialize(reader);
    this->layout_.Deserialize(reader);
    StreamReader::ReadObj(reader, this->code_line_size_);
    this->layout_.SetCodeSize(this->code_line_size_);
    if (is_support_delete_) {
        StreamReader::ReadVector(reader, node_versions_);
    }
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl>::DeleteNeighborsById(vsag::InnerIdType id) {
    if (is_support_delete_) {
        if (id < max_capacity_) {
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
        if (id < max_capacity_) {
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

template <typename IOTmpl>
void
GraphDataCell<IOTmpl>::Move(InnerIdType from, InnerIdType to) {
    if (from == to) {
        return;
    }

    Vector<InnerIdType> reverse_neighbors(allocator_);
    this->GetIncomingNeighbors(from, reverse_neighbors);

    Vector<InnerIdType> neighbors(allocator_);
    this->InsertNeighborsById(to, neighbors);
    for (const auto& reverse_nb : reverse_neighbors) {
        this->GetNeighbors(reverse_nb, neighbors);
        Vector<InnerIdType> new_neighbors(allocator_);
        bool has_to = false;
        for (const auto& nb : neighbors) {
            if (nb != from) {
                new_neighbors.emplace_back(nb);
            }
            if (nb == to) {
                has_to = true;
            }
        }
        if (not has_to) {
            new_neighbors.emplace_back(to);
        }
        this->InsertNeighborsById(reverse_nb, new_neighbors);
        neighbors.clear();
    }

    Vector<InnerIdType> from_neighbors(allocator_);
    this->GetNeighbors(from, from_neighbors);
    this->InsertNeighborsById(to, from_neighbors);

    from_neighbors.clear();
    this->InsertNeighborsById(from, from_neighbors);

    if (is_support_delete_) {
        node_versions_[to] = node_versions_[from];
    }
}

}  // namespace vsag
