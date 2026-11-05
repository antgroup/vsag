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

    /// Per-row MCI work precomputed by the lock-free planning pass.
    ///
    /// Every field is derived from one frozen CliqueDataCell read view plus this row's own KNN, so
    /// the workers that fill these plans never write shared state. The planning pass only records;
    /// the merge pass revalidates each decision against the live structure, so a plan that is
    /// older than the live structure can under-report work but never over-report it.
    struct MciAddPlan {
        explicit MciAddPlan(Allocator* allocator)
            : knn_ids(allocator), join_targets(allocator), new_cliques(allocator) {
        }

        void
        Reset() {
            this->inner_id = 0;
            this->visible_total = 0;
            this->degree_target = 0;
            this->live_neighbor_count = 0;
            this->memberships_empty = false;
            this->knn_empty = false;
            this->knn_ids.clear();
            this->join_targets.clear();
            this->new_cliques.clear();
        }

        InnerIdType inner_id{0};
        /// Insertion-prefix candidate bound (inner_id + 1) used while planning.
        uint64_t visible_total{0};
        /// Degree target and both flags below are relative to the planning view, which is a lower
        /// bound of the live structure.
        uint64_t degree_target{0};
        /// Neighbors the frozen view already gives this row, excluding the cliques the plan only
        /// projects. The merge skips a row once this reaches degree_target, because then the live
        /// structure is already covered and there is nothing left for the merge to write.
        uint64_t live_neighbor_count{0};
        bool memberships_empty{false};
        bool knn_empty{false};
        Vector<InnerIdType> knn_ids;
        /// Existing cliques worth joining as (intersection, clique_id), ordered by gain.
        Vector<std::pair<uint64_t, InnerIdType>> join_targets;
        /// Local MCE proposals, already mapped to global inner IDs and sorted by member ID.
        Vector<Vector<InnerIdType>> new_cliques;
    };

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

    /// Apply the first plan_count plans of one chunk in inner-ID order. Returns the number of rows
    /// that needed the serial repair fallback, which is logged by the caller. Caller holds
    /// mci_mutation_mutex_.
    uint64_t
    merge_incremental_mci_clique_plans(Vector<MciAddPlan>& plans,
                                       uint64_t plan_count,
                                       uint64_t total);

private:
    HGraph& hgraph_;
};

}  // namespace vsag
