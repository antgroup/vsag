
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

#include <shared_mutex>

#include "graph_interface.h"
#include "io/memory_block_io.h"
#include "sparse_graph_datacell_parameter.h"

namespace vsag {

class SparseGraphDataCell : public GraphInterface {
public:
    using NeighborCountsType = uint32_t;

    SparseGraphDataCell(const GraphInterfaceParamPtr& graph_param,
                        const IndexCommonParam& common_param);
    SparseGraphDataCell(const SparseGraphDatacellParamPtr& graph_param,
                        const IndexCommonParam& common_param);
    SparseGraphDataCell(const SparseGraphDatacellParamPtr& graph_param, Allocator* allocator);

    void
    InsertNeighborsById(InnerIdType id, const Vector<InnerIdType>& neighbor_ids) override;

    void
    DeleteNeighborsById(InnerIdType id) override;

    uint32_t
    GetNeighborSize(InnerIdType id) const override;

    void
    GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const override;

    void
    Resize(InnerIdType new_size) override;

    /****
     * prefetch neighbors of a base point with id
     * @param id of base point
     * @param neighbor_i index of neighbor, 0 for neighbor size, 1 for first neighbor
     */
    void
    Prefetch(InnerIdType id, uint32_t neighbor_i) override {
        // TODO(LHT): implement
    }

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(StreamReader& reader) override;

private:
    uint32_t code_line_size_{0};
    Allocator* const allocator_{nullptr};
    UnorderedMap<InnerIdType, std::unique_ptr<Vector<InnerIdType>>> neighbors_;
    mutable std::shared_mutex neighbors_map_mutex_{};

    bool is_support_delete_{true};
    uint32_t remove_flag_bit_{8};
    uint32_t id_bit_{24};
    uint32_t remove_flag_mask_{0x00ffffff};
    UnorderedMap<InnerIdType, uint8_t> node_version_;
};

}  // namespace vsag
