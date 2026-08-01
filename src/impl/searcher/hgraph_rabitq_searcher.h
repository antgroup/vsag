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

#include "datacell/flatten_interface.h"
#include "datacell/hgraph_rabitq_fused_datacell.h"
#include "impl/heap/distance_heap.h"
#include "impl/inner_search_param.h"
#include "index_common_param_fwd.h"
#include "query_context.h"
#include "utils/lock_strategy.h"
#include "utils/visited_list.h"

namespace vsag {

class HGraphRaBitQSearcher {
public:
    explicit HGraphRaBitQSearcher(const IndexCommonParam& common_param,
                                  MutexArrayPtr neighbors_mutex);

    void
    SetMutexArray(const MutexArrayPtr& neighbors_mutex) {
        neighbors_mutex_ = neighbors_mutex;
    }

    DistHeapPtr
    Search(const HGraphRaBitQFusedDataCellPtr& graph,
           const FlattenInterfacePtr& flatten,
           const VisitedListPtr& visited_list,
           const void* query,
           const InnerSearchParam& search_param,
           QueryContext* ctx,
           RaBitQCandidateVector* lower_bound_candidates) const;

    InnerIdType
    Route(const GraphInterfacePtr& route_graph,
          const HGraphRaBitQFusedDataCellPtr& fused_graph,
          const FlattenInterfacePtr& flatten,
          const ComputerInterfacePtr& computer,
          InnerIdType entry_point,
          bool enable_one_bit_search) const;

private:
    Allocator* allocator_{nullptr};
    MutexArrayPtr neighbors_mutex_{nullptr};
};

}  // namespace vsag
