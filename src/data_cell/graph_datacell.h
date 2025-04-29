
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

    uint32_t code_line_size_{0};
};

template <typename IOTmpl>
GraphDataCell<IOTmpl>::GraphDataCell(const GraphDataCellParamPtr& param,
                                     const IndexCommonParam& common_param) {
    this->io_ = std::make_shared<IOTmpl>(param->io_parameter_, common_param);
    this->maximum_degree_ = param->max_degree_;
    this->max_capacity_ = param->init_max_capacity_;
    this->code_line_size_ = this->maximum_degree_ * sizeof(InnerIdType) + sizeof(uint32_t);
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
    uint32_t neighbor_count = std::min((uint32_t)(neighbor_ids.size()), this->maximum_degree_);
    this->io_->Write((uint8_t*)(&neighbor_count), sizeof(neighbor_count), start);
    start += sizeof(neighbor_count);
    this->io_->Write((uint8_t*)(neighbor_ids.data()),
                     static_cast<uint64_t>(neighbor_count) * sizeof(InnerIdType),
                     start);
}

template <typename IOTmpl>
uint32_t
GraphDataCell<IOTmpl>::GetNeighborSize(InnerIdType id) const {
    auto start = static_cast<uint64_t>(id) * static_cast<uint64_t>(this->code_line_size_);
    uint32_t result = 0;
    this->io_->Read(sizeof(result), start, (uint8_t*)(&result));
    return result;
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl>::GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const {
    auto start = static_cast<uint64_t>(id) * static_cast<uint64_t>(this->code_line_size_);
    uint32_t neighbor_count = 0;
    this->io_->Read(sizeof(neighbor_count), start, (uint8_t*)(&neighbor_count));
    neighbor_ids.resize(neighbor_count);
    start += sizeof(neighbor_count);
    this->io_->Read(
        neighbor_ids.size() * sizeof(InnerIdType), start, (uint8_t*)(neighbor_ids.data()));
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl>::Resize(InnerIdType new_size) {
    if (new_size < this->max_capacity_) {
        return;
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
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl>::Deserialize(StreamReader& reader) {
    GraphInterface::Deserialize(reader);
    this->io_->Deserialize(reader);
    StreamReader::ReadObj(reader, this->code_line_size_);
}

}  // namespace vsag
