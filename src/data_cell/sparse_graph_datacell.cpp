
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

#include "sparse_graph_datacell.h"

#include "sparse_graph_datacell_parameter.h"

namespace vsag {

SparseGraphDataCell::SparseGraphDataCell(const SparseGraphDatacellParamPtr& graph_param,
                                         Allocator* allocator)
    : allocator_(allocator),
      neighbors_(allocator),
      node_version_(allocator),
      is_support_delete_(graph_param->support_delete_) {
    this->maximum_degree_ = graph_param->max_degree_;
    this->remove_flag_bit_ = graph_param->remove_flag_bit_;
    this->id_bit_ = sizeof(InnerIdType) * 8 - this->remove_flag_bit_;
    this->remove_flag_mask_ = (1 << this->id_bit_) - 1;
}

SparseGraphDataCell::SparseGraphDataCell(const SparseGraphDatacellParamPtr& graph_param,
                                         const IndexCommonParam& common_param)
    : SparseGraphDataCell(graph_param, common_param.allocator_.get()) {
}

SparseGraphDataCell::SparseGraphDataCell(const GraphInterfaceParamPtr& param,
                                         const IndexCommonParam& common_param)
    : SparseGraphDataCell(std::dynamic_pointer_cast<SparseGraphDatacellParameter>(param),
                          common_param) {
}

void
SparseGraphDataCell::InsertNeighborsById(InnerIdType id, const Vector<InnerIdType>& neighbor_ids) {
    if (neighbor_ids.size() > this->maximum_degree_) {
        throw std::invalid_argument(fmt::format(
            "insert neighbors count {} more than {}", neighbor_ids.size(), this->maximum_degree_));
    }
    auto size = std::min(this->maximum_degree_, (uint32_t)(neighbor_ids.size()));
    std::unique_lock<std::shared_mutex> wlock(this->neighbors_map_mutex_);
    this->max_capacity_ = std::max(this->max_capacity_, id + 1);
    auto iter = this->neighbors_.find(id);
    if (iter == this->neighbors_.end()) {
        iter =
            this->neighbors_.emplace(id, std::make_unique<Vector<InnerIdType>>(allocator_)).first;
        total_count_++;
        if (is_support_delete_) {
            node_version_[id] = 0;
        }
    }
    if (is_support_delete_) {
        iter->second->resize(size);
        for (int i = 0; i < size; ++i) {
            iter->second->at(i) = (neighbor_ids[i] | (node_version_[neighbor_ids[i]] << id_bit_));
        }
    } else {
        iter->second->assign(neighbor_ids.begin(), neighbor_ids.begin() + size);
    }
}

uint32_t
SparseGraphDataCell::GetNeighborSize(InnerIdType id) const {
    std::shared_lock<std::shared_mutex> rlock(this->neighbors_map_mutex_);
    auto iter = this->neighbors_.find(id);
    if (iter != this->neighbors_.end()) {
        return iter->second->size();
    }
    return 0;
}
void
SparseGraphDataCell::GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const {
    std::shared_lock<std::shared_mutex> rlock(this->neighbors_map_mutex_);
    auto iter = this->neighbors_.find(id);
    if (iter != this->neighbors_.end()) {
        const auto& ngbrs = iter->second;
        if (is_support_delete_) {
            neighbor_ids.clear();
            neighbor_ids.reserve(iter->second->size());
            for (unsigned int& neighbor_id : *ngbrs) {
                uint8_t cur_version = neighbor_id >> id_bit_;
                uint32_t real_id = neighbor_id & remove_flag_mask_;
                if (node_version_.at(real_id) == cur_version) {
                    neighbor_ids.push_back(real_id);
                }
            }
        } else {
            neighbor_ids.assign(iter->second->begin(), iter->second->end());
        }
    }
}
void
SparseGraphDataCell::Serialize(StreamWriter& writer) {
    GraphInterface::Serialize(writer);
    StreamWriter::WriteObj(writer, this->code_line_size_);
    auto size = this->neighbors_.size();
    StreamWriter::WriteObj(writer, size);
    for (auto& pair : this->neighbors_) {
        auto key = pair.first;
        StreamWriter::WriteObj(writer, key);
        StreamWriter::WriteVector(writer, *(pair.second));
    }
    if (is_support_delete_) {
        for (const auto& item : this->node_version_) {
            StreamWriter::WriteObj(writer, item.first);
            StreamWriter::WriteObj(writer, item.second);
        }
    }
}

void
SparseGraphDataCell::Deserialize(StreamReader& reader) {
    GraphInterface::Deserialize(reader);
    StreamReader::ReadObj(reader, this->code_line_size_);
    uint64_t size;
    StreamReader::ReadObj(reader, size);
    for (uint64_t i = 0; i < size; ++i) {
        InnerIdType key;
        StreamReader::ReadObj(reader, key);
        this->neighbors_[key] = std::make_unique<vsag::Vector<InnerIdType>>(allocator_);
        StreamReader::ReadVector(reader, *(this->neighbors_[key]));
    }
    this->total_count_ = size;
    if (is_support_delete_) {
        for (uint64_t i = 0; i < size; ++i) {
            InnerIdType key;
            StreamReader::ReadObj(reader, key);
            uint8_t value;
            StreamReader::ReadObj(reader, value);
            this->node_version_[key] = value;
        }
    }
}

void
SparseGraphDataCell::Resize(InnerIdType new_size){};

void
SparseGraphDataCell::DeleteNeighborsById(vsag::InnerIdType id) {
    if (is_support_delete_) {
        std::unique_lock<std::shared_mutex> wlock(this->neighbors_map_mutex_);
        auto iter = node_version_.find(id);
        if (iter != node_version_.end()) {
            if (iter->second + 1 == 0) {
                throw VsagException(
                    ErrorType::INTERNAL_ERROR,
                    "remove point too many times in SparseGraphDatacell, please rebuild index");
            }
            iter->second++;
        } else {
            throw VsagException(
                ErrorType::INTERNAL_ERROR,
                fmt::format("remove point {} not exist in SparseGraphDatacell", id));
        }
    } else {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "disable delete in sparse graph datacell");
    }
}

}  // namespace vsag
