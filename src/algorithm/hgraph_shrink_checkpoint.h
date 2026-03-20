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

#include "hgraph_shrink_edges.h"

namespace vsag {

class HgraphShrinkCheckPoint {
public:
    constexpr static uint64_t BATCH_SIZE = 10;

    HgraphShrinkCheckPoint() = default;

    std::vector<InnerIdType> deleted_nodes_;  // for state PREPARE
    InnerIdType current_deleted_node_index_{0};

    HGraphShrinkEdges current_forward_edges_;  // for state COLLECT_FORWARD_EDGES
    InnerIdType current_forward_edge_index_{0};

    HGraphShrinkEdges current_reversion_edges_;  // for state COLLECT_REVERSE_EDGES
    InnerIdType current_reversion_edge_index_{0};

    InnerIdType process_reversion_edge_index_{0};
    InnerIdType process_forward_edge_index_{0};
};

}  // namespace vsag
