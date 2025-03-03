
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

#include "graph_datacell_parameter.h"

namespace vsag {

SparseGraphDataCell::SparseGraphDataCell(Allocator* allocator, uint32_t max_degree)
    : allocator_(allocator), neighbors_(allocator_) {
    this->maximum_degree_ = max_degree;
}

SparseGraphDataCell::SparseGraphDataCell(const GraphInterfaceParamPtr& param,
                                         const IndexCommonParam& common_param)
    : SparseGraphDataCell(common_param.allocator_.get(),
                          std::dynamic_pointer_cast<GraphDataCellParameter>(param)->max_degree_) {
}

void
SparseGraphDataCell::InsertNeighborsById(InnerIdType id, const Vector<InnerIdType>& neighbor_ids) {
    if (neighbor_ids.size() > this->maximum_degree_) {
        logger::warn(fmt::format(
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
    }
    iter->second->assign(neighbor_ids.begin(), neighbor_ids.begin() + size);
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
        neighbor_ids.assign(iter->second->begin(), iter->second->end());
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
}
void
SparseGraphDataCell::Resize(InnerIdType new_size){};
}  // namespace vsag
