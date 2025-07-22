
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

#include "compressed_graph_datacell_parameter.h"
#include "graph_interface.h"
#include "impl/elias_fano_encoder.h"
#include "io/memory_block_io.h"

namespace vsag {

class CompressedGraphDataCell : public GraphInterface {
public:
    explicit CompressedGraphDataCell(const GraphInterfaceParamPtr& graph_param,
                                     const IndexCommonParam& common_param);

    explicit CompressedGraphDataCell(const CompressedGraphDatacellParamPtr& graph_param,
                                     const IndexCommonParam& common_param);

    ~CompressedGraphDataCell();

    void
    InsertNeighborsById(InnerIdType id, const Vector<InnerIdType>& neighbor_ids) override;

    [[nodiscard]] uint32_t
    GetNeighborSize(InnerIdType id) const override;

    void
    GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const override;

    void
    Resize(InnerIdType new_size) override;

    void
    Prefetch(InnerIdType id, uint32_t neighbor_i) override {
        // Not supposed to use
    }

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(StreamReader& reader) override;

private:
    Allocator* const allocator_{nullptr};
    Vector<std::unique_ptr<EliasFanoEncoder>> neighbor_sets_;
};

}  // namespace vsag
