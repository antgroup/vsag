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

#include <atomic>
#include <limits>
#include <shared_mutex>

#include "container_types.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

class StreamReader;
class StreamWriter;

DEFINE_POINTER(CliqueDataCell);

struct CliqueDataCellStats {
    bool has_index{false};
    uint64_t total_nodes{0};
    uint64_t covered_nodes{0};
    uint64_t base_clique_count{0};
    uint64_t delta_clique_count{0};
    uint64_t total_clique_count{0};
    uint64_t retired_clique_count{0};
    uint64_t inactive_node_count{0};
    uint64_t base_membership_count{0};
    uint64_t delta_extra_membership_count{0};
    uint64_t delta_clique_membership_count{0};
    uint64_t total_membership_count{0};
    uint64_t max_membership_per_node{0};
    uint64_t max_clique_size{0};
    double covered_node_ratio{0.0};
    double avg_membership_per_node{0.0};
    double avg_clique_size{0.0};
};

// One shared lock pins both CSR and delta storage for an entire query.
struct CliqueDataCellSearchView {
    std::shared_lock<std::shared_mutex> guard;
    const InnerIdType* p_maxc{nullptr};
    const InnerIdType* maxcs{nullptr};
    const InnerIdType* p_node_to_cid{nullptr};
    const InnerIdType* node_to_cids{nullptr};
    uint64_t total_clique_count{0};
    uint64_t base_clique_count{0};
    uint64_t base_node_count{0};
    uint64_t total_nodes{0};
    const Vector<Vector<InnerIdType>>* delta_cliques{nullptr};
    const Vector<Vector<InnerIdType>>* delta_extra{nullptr};
    const Vector<Vector<InnerIdType>>* delta_node_cids{nullptr};
    const uint8_t* inactive_nodes{nullptr};
    const uint8_t* retired_cliques{nullptr};

    [[nodiscard]] bool
    IsLiveNode(InnerIdType id) const {
        return id < total_nodes and inactive_nodes[id] == 0;
    }

    template <typename Visitor>
    void
    ForEachNodeClique(InnerIdType id, Visitor&& visit) const {
        if (not IsLiveNode(id)) {
            return;
        }
        // Base rows only cover [0, base_node_count); the delta row exists for every live slot in
        // [0, total_nodes), which TryGetSearchView validates before publishing the view.
        if (id < base_node_count) {
            for (auto offset = p_node_to_cid[id]; offset < p_node_to_cid[id + 1]; ++offset) {
                const auto cid = node_to_cids[offset];
                if (cid < total_clique_count and retired_cliques[cid] == 0 and not visit(cid)) {
                    return;
                }
            }
        }
        for (auto cid : (*delta_node_cids)[id]) {
            if (cid < total_clique_count and retired_cliques[cid] == 0 and not visit(cid)) {
                return;
            }
        }
    }

    template <typename Visitor>
    void
    ForEachMember(InnerIdType cid, Visitor&& visit) const {
        if (cid >= total_clique_count or retired_cliques[cid] != 0) {
            return;
        }
        if (cid < base_clique_count) {
            for (auto offset = p_maxc[cid]; offset < p_maxc[cid + 1]; ++offset) {
                if (IsLiveNode(maxcs[offset])) {
                    visit(maxcs[offset]);
                }
            }
            for (auto id : (*delta_extra)[cid]) {
                if (IsLiveNode(id)) {
                    visit(id);
                }
            }
        } else {
            for (auto id : (*delta_cliques)[cid - base_clique_count]) {
                if (IsLiveNode(id)) {
                    visit(id);
                }
            }
        }
    }
};

struct MCIDeleteSnapshot {
    // Node IDs belong to the slot space at PrepareDelete time, before FORCE_REMOVE compaction.
    // Translate repair_node_ids through old_to_new before accessing the compacted index.
    explicit MCIDeleteSnapshot(Allocator* allocator)
        : affected_clique_ids(allocator),
          retired_clique_ids(allocator),
          repair_node_ids(allocator) {
    }

    Vector<InnerIdType> affected_clique_ids;
    Vector<InnerIdType> retired_clique_ids;
    Vector<InnerIdType> repair_node_ids;
};

class CliqueDataCell {
public:
    explicit CliqueDataCell(Allocator* allocator);

    void
    Clear(uint64_t total);

    void
    Assign(Vector<InnerIdType>&& p_maxc,
           Vector<InnerIdType>&& maxcs,
           Vector<InnerIdType>&& p_node_to_cid,
           Vector<InnerIdType>&& node_to_cids,
           uint64_t total);

    // Merge live base/delta memberships into CSR without changing vector inner IDs.
    void
    Flush(uint64_t total);

    // Compact into CSR while remapping vector slots. The maximum InnerIdType removes a slot.
    // The caller must exclude mutations and publish only after vector moves and repair finish.
    void
    RemapNodes(const Vector<InnerIdType>& old_to_new, uint64_t total);

    Vector<InnerIdType>
    GetInactiveNodeIds() const;

    void
    MarkUnavailable();

    void
    MarkAvailable(uint64_t total);

    [[nodiscard]] bool
    HasCliqueIndex(uint64_t total) const;

    [[nodiscard]] uint64_t
    TotalLogicalCliqueCount() const;

    [[nodiscard]] uint64_t
    GetTotalNodes() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return delta_node_to_cids_.size();
    }

    void
    CollectNodeCliqueIds(InnerIdType node_id, Vector<InnerIdType>& clique_ids) const;

    void
    GetCliqueMembers(InnerIdType clique_id, Vector<InnerIdType>& members) const;

    [[nodiscard]] uint64_t
    GetCliqueMemberCount(InnerIdType clique_id) const;

    bool
    AppendNodeToClique(InnerIdType node_id,
                       InnerIdType clique_id,
                       uint64_t total,
                       uint64_t max_members);

    void
    AppendNewClique(const Vector<InnerIdType>& members, uint64_t total);

    [[nodiscard]] MCIDeleteSnapshot
    PrepareDelete(const Vector<InnerIdType>& node_ids,
                  uint64_t clique_size_threshold,
                  uint64_t node_mct_threshold) const;

    void
    CommitDelete(const Vector<InnerIdType>& node_ids,
                 const Vector<InnerIdType>& retired_clique_ids,
                 uint64_t total);

    void
    Serialize(StreamWriter& writer) const;

    /// If supplied, validate the caller's physical ID domain before publishing the loaded CSR.
    void
    Deserialize(StreamReader& reader,
                uint64_t format_version = 2,
                uint64_t expected_total = std::numeric_limits<uint64_t>::max());

    [[nodiscard]] uint64_t
    GetMemoryUsage() const;

    [[nodiscard]] bool
    TryGetSearchView(uint64_t total, CliqueDataCellSearchView& view) const;

    [[nodiscard]] CliqueDataCellStats
    CollectStats(uint64_t total) const;

    /// One live node slot: in range and not marked inactive. This is the exact condition
    /// PrepareDelete uses to accept a vector id as deleting.
    [[nodiscard]] bool
    IsNodeLiveUnlocked(InnerIdType node_id) const;

    /// Batch-scoped read access for callers that must read many cliques, such as the chunked Add
    /// planning pass and the parallel delete prepare. Such a caller takes the shared lock once with
    /// AcquireReadLock and then uses the *Unlocked readers below, paying one lock acquisition per
    /// chunk instead of one per query.
    ///
    /// Contract: the returned lock must stay alive for the whole read burst, no session may
    /// outlive the burst, and no mutating call may run (here or on another thread) until it is
    /// released. The view is only frozen against mutation: later appends are invisible, and the
    /// absence of a clique or membership in this view is not proof that it is absent live.
    [[nodiscard]] std::shared_lock<std::shared_mutex>
    AcquireReadLock() const;

    void
    CollectNodeCliqueIdsUnlocked(InnerIdType node_id, Vector<InnerIdType>& clique_ids) const;

    void
    GetCliqueMembersUnlocked(InnerIdType clique_id, Vector<InnerIdType>& members) const;

    [[nodiscard]] uint64_t
    GetCliqueMemberCountUnlocked(InnerIdType clique_id) const;

    [[nodiscard]] uint64_t
    TotalLogicalCliqueCountUnlocked() const;

    /// PrepareDelete with the same snapshot semantics, but the three read-only stages are spread
    /// over Min(thread_count, items) workers with per-worker output buffers that are merged, sorted
    /// and deduplicated between stages. Every stage reads the same pre-mutation state, so the
    /// returned sets are identical to the serial PrepareDelete.
    [[nodiscard]] MCIDeleteSnapshot
    PrepareDeleteParallel(const Vector<InnerIdType>& node_ids,
                          uint64_t clique_size_threshold,
                          uint64_t node_mct_threshold,
                          uint64_t thread_count) const;

private:
    void
    reset_delta_unlocked(uint64_t total);

    void
    ensure_delta_node_rows_unlocked(uint64_t total);

    // Caller holds mutex_. Preserve storage origin for allocation-free counts/statistics,
    // including while MCI is unpublished and the query view cannot be acquired.
    template <typename Visitor>
    void
    for_each_live_member_unlocked(InnerIdType clique_id, Visitor visit) const {
        if (is_clique_retired_unlocked(clique_id)) {
            return;
        }
        auto append_live = [&](auto begin, auto end, bool from_base) {
            for (auto iter = begin; iter != end; ++iter) {
                if (not is_node_inactive_unlocked(*iter)) {
                    visit(*iter, from_base);
                }
            }
        };
        if (clique_id < total_clique_count_) {
            append_live(
                maxcs_.begin() + p_maxc_[clique_id], maxcs_.begin() + p_maxc_[clique_id + 1], true);
            if (clique_id < delta_clique_extra_.size()) {
                const auto& extra = delta_clique_extra_[clique_id];
                append_live(extra.begin(), extra.end(), false);
            }
        } else {
            const auto delta_id = clique_id - total_clique_count_;
            if (delta_id < delta_cliques_.size()) {
                const auto& members = delta_cliques_[delta_id];
                append_live(members.begin(), members.end(), false);
            }
        }
    }

    [[nodiscard]] uint64_t
    total_logical_clique_count_unlocked() const;

    [[nodiscard]] bool
    is_node_inactive_unlocked(InnerIdType node_id) const;

    [[nodiscard]] bool
    is_clique_retired_unlocked(InnerIdType clique_id) const;

    void
    collect_node_clique_ids_unlocked(InnerIdType node_id, Vector<InnerIdType>& clique_ids) const;

    void
    get_clique_members_unlocked(InnerIdType clique_id, Vector<InnerIdType>& members) const;

    void
    validate(uint64_t total) const;

    void
    compact_unlocked(uint64_t total, const Vector<InnerIdType>* old_to_new);

private:
    Allocator* allocator_{nullptr};

    Vector<InnerIdType> p_maxc_;
    Vector<InnerIdType> maxcs_;
    Vector<InnerIdType> p_node_to_cid_;
    Vector<InnerIdType> node_to_cids_;
    uint64_t total_clique_count_{0};
    Vector<Vector<InnerIdType>> delta_cliques_;
    Vector<Vector<InnerIdType>> delta_clique_extra_;
    Vector<Vector<InnerIdType>> delta_node_to_cids_;
    Vector<uint8_t> inactive_nodes_;
    Vector<uint8_t> retired_cliques_;
    uint64_t active_clique_count_{0};
    uint64_t inactive_node_count_{0};
    uint64_t retired_clique_count_{0};
    // Transient, not serialized. Unlike inactive_node_count_, this distinguishes already
    // compacted tombstones from new changes. Clear only after successful compaction.
    bool needs_compaction_{false};
    std::atomic<uint64_t> available_total_{0};

    mutable std::shared_mutex mutex_;
};

}  // namespace vsag
