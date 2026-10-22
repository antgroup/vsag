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

#include "parallel_searcher_impl.h"

namespace vsag {

DistHeapPtr
ParallelSearcher::SearchWithDistanceEarlyStop(
    const GraphInterfacePtr& graph,
    const FlattenInterfacePtr& flatten,
    const VisitedListPtr& vl,
    const void* query,
    const InnerSearchParam& inner_search_param,
    const LabelTablePtr& label_table,
    QueryContext* ctx,
    DistanceRecordVector* rabitq_lower_bound_candidates) const {
    CHECK_ARGUMENT(inner_search_param.search_mode == KNN_SEARCH,
                   "distance early stop supports KNN search only");
    if (ctx != nullptr and ctx->stats != nullptr) {
        ctx->stats->early_stop_armed.store(true, std::memory_order_relaxed);
    }
    return this->search_impl<KNN_SEARCH, true>(graph,
                                               flatten,
                                               vl,
                                               query,
                                               inner_search_param,
                                               label_table,
                                               ctx,
                                               rabitq_lower_bound_candidates);
}

}  // namespace vsag
