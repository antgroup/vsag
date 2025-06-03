
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
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <vector>

#include "algorithm/hnswlib/hnswalg.h"
#include "common.h"
#include "graph_datacell_parameter.h"
#include "graph_interface.h"
#include "graph_interface_parameter.h"
#include "index/index_common_param.h"
#include "io/basic_io.h"
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

    [[nodiscard]] uint32_t
    GetNeighborSize(InnerIdType id) const override;

    void
    GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const override;

    void
    Resize(InnerIdType new_size) override;

    inline void
    SetIO(std::shared_ptr<BasicIO<IOTmpl>> io) {
        this->io_ = io;
    }

    /****
     * prefetch neighbors of a base point with id
     * @param id of base point
     * @param neighbor_i index of neighbor, 0 for neighbor size, 1 for first neighbor
     */
    void
    Prefetch(InnerIdType id, uint32_t neighbor_i) override {
        io_->Prefetch(static_cast<uint64_t>(id) * static_cast<uint64_t>(this->code_line_size_) +
                      sizeof(uint32_t) + neighbor_i * sizeof(InnerIdType));
    }

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(StreamReader& reader) override;

    bool
    InMemory() const override {
        return this->io_->InMemory();
    }

private:
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};

    Vector<uint8_t> node_versions_;

    bool is_support_delete_{true};
    uint32_t remove_flag_bit_{8};
    uint32_t id_bit_{24};
    uint32_t remove_flag_mask_{0x00ffffff};

    uint32_t code_line_size_{0};
};

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
    this->max_capacity_ = new_size;
    uint64_t io_size = static_cast<uint64_t>(new_size) * static_cast<uint64_t>(code_line_size_);
    uint8_t end_flag =
        127;  // the value is meaningless, only to occupy the position for io allocate
    this->io_->Write(&end_flag, 1, io_size);
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

}  // namespace vsag
