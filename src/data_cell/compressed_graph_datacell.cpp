
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

#include "compressed_graph_datacell.h"

#include "graph_datacell_parameter.h"

namespace vsag {

CompressedGraphDataCell::CompressedGraphDataCell(const GraphInterfaceParamPtr& graph_param,
                                                 const IndexCommonParam& common_param)
    : CompressedGraphDataCell(
          std::dynamic_pointer_cast<CompressedGraphDatacellParameter>(graph_param), common_param) {
}

CompressedGraphDataCell::CompressedGraphDataCell(const CompressedGraphDatacellParamPtr& graph_param,
                                                 const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()), neighbor_sets_(allocator_) {
    this->maximum_degree_ = graph_param->max_degree_;
    this->max_capacity_ = 0;
}

void
CompressedGraphDataCell::InsertNeighborsById(InnerIdType id,
                                             const Vector<InnerIdType>& neighbor_ids) {
    if (neighbor_ids.size() > this->maximum_degree_) {
        throw std::invalid_argument(fmt::format(
            "insert neighbors count {} more than {}", neighbor_ids.size(), this->maximum_degree_));
    }

    Vector<InnerIdType> tmp(neighbor_ids.begin(), neighbor_ids.end(), allocator_);
    std::sort(tmp.begin(), tmp.end());

    // TODO(deming): reset the size of neighbor_sets_[id] while tmp is empty
    if (not tmp.empty()) {
        if (neighbor_sets_[id] == nullptr) {
            neighbor_sets_[id] = std::make_unique<EliasFanoEncoder>(allocator_);
        }
        neighbor_sets_[id]->Encode(tmp, max_capacity_);
    }
}

uint32_t
CompressedGraphDataCell::GetNeighborSize(InnerIdType id) const {
    return (neighbor_sets_[id]) ? neighbor_sets_[id]->Size() : 0;
}

void
CompressedGraphDataCell::GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const {
    if (GetNeighborSize(id) > 0) {
        neighbor_sets_[id]->DecompressAll(neighbor_ids);
    }
}

void
CompressedGraphDataCell::Serialize(StreamWriter& writer) {
    GraphInterface::Serialize(writer);

    auto vertex_num = this->neighbor_sets_.size();
    StreamWriter::WriteObj(writer, vertex_num);
    for (InnerIdType id = 0; id < vertex_num; id++) {
        if (GetNeighborSize(id) == 0) {
            uint8_t zero = 0;
            StreamWriter::WriteObj(writer, zero);
        } else {
            const EliasFanoEncoder& encoder = *neighbor_sets_[id];
            StreamWriter::WriteObj(writer, encoder.num_elements);
            StreamWriter::WriteObj(writer, encoder.low_bits_width);
            StreamWriter::WriteObj(writer, encoder.low_bits_size);
            StreamWriter::WriteObj(writer, encoder.high_bits_size);
            for (size_t j = 0; j < encoder.low_bits_size + encoder.high_bits_size; j++) {
                StreamWriter::WriteObj(writer, encoder.bits[j]);
            }
        }
    }
}

void
CompressedGraphDataCell::Deserialize(StreamReader& reader) {
    GraphInterface::Deserialize(reader);
    uint64_t vertex_num;
    StreamReader::ReadObj(reader, vertex_num);
    Resize(vertex_num);
    for (uint64_t id = 0; id < vertex_num; ++id) {
        uint8_t num_elements = 0;
        StreamReader::ReadObj(reader, num_elements);
        if (num_elements > 0) {
            this->neighbor_sets_[id] = std::make_unique<EliasFanoEncoder>(allocator_);
            EliasFanoEncoder& encoder = *this->neighbor_sets_[id];
            encoder.num_elements = num_elements;
            StreamReader::ReadObj(reader, encoder.low_bits_width);
            StreamReader::ReadObj(reader, encoder.low_bits_size);
            StreamReader::ReadObj(reader, encoder.high_bits_size);

            encoder.bits = static_cast<uint64_t*>(allocator_->Allocate(
                (encoder.low_bits_size + encoder.high_bits_size) * sizeof(uint64_t)));
            for (size_t j = 0; j < encoder.low_bits_size + encoder.high_bits_size; j++) {
                StreamReader::ReadObj(reader, encoder.bits[j]);
            }
        }
    }
    this->total_count_ = vertex_num;
}

void
CompressedGraphDataCell::Resize(InnerIdType new_size) {
    if (new_size < this->max_capacity_) {
        return;
    }
    neighbor_sets_.resize(new_size);
    this->max_capacity_ = new_size;
    this->total_count_ = new_size;
}

}  // namespace vsag
