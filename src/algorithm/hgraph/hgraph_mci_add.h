// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <functional>
#include <shared_mutex>
#include <utility>
#include <vector>

#include "hgraph.h"

namespace vsag {

/// Lock-free planning and application of the incremental MCI update of one completed Add batch.
///
/// Constructed on the stack of HGraph::add_impl for the duration of one batch. The executor only
/// borrows the index: every worker, plan slot and frozen read view lives inside Run. Keeping this
/// machinery here rather than on HGraph leaves hgraph.h with a forward declaration plus one friend
/// line, and keeps the parallel path out of the header that every translation unit includes.
/// Helpers shared with the serial update (collect_mci_join_targets, apply_mci_join_targets and
/// build_local_mci_cliques) stay on HGraph because its serial paths call them too.
class MciIncrementalAddExecutor {
public:
    explicit MciIncrementalAddExecutor(HGraph& hgraph) : hgraph_(hgraph) {
    }

    MciIncrementalAddExecutor(const MciIncrementalAddExecutor&) = delete;
    MciIncrementalAddExecutor&
    operator=(const MciIncrementalAddExecutor&) = delete;

    void
    Run(const HGraph::AddBatch& batch, const DatasetPtr& data);

    /// Plan one row of an Add batch against a frozen read view. Caller owns view and plan and must
    /// not hold any CliqueDataCell write lock. plan.inner_id must already hold the row this plan
    /// describes: it is read here and reused by the merge, so the caller sets it once. Plans are
    /// not applied here.
    void
    plan_incremental_mci_clique(const std::shared_lock<std::shared_mutex>& view,
                                const Vector<InnerIdType>& knn_ids,
                                uint64_t visible_total,
                                uint64_t degree_target,
                                MciAddPlan& plan);

private:
    HGraph& hgraph_;
};

}  // namespace vsag
