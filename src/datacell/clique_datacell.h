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
#include <shared_mutex>

#include "container_types.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

class StreamReader;
class StreamWriter;

DEFINE_POINTER(CliqueDataCell);

struct CliqueDataCellBaseView {
    // Pins the CSR pointer storage for the lifetime of this view.
    std::shared_lock<std::shared_mutex> guard;
    const InnerIdType* p_maxc{nullptr};
    const InnerIdType* maxcs{nullptr};
    const InnerIdType* p_node_to_cid{nullptr};
    const InnerIdType* node_to_cids{nullptr};
    uint64_t total_clique_count{0};
};

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
struct CliqueDataCellSearchView : CliqueDataCellBaseView {
    uint64_t base_clique_count{0};
    uint64_t base_node_count{0};
    uint64_t total_nodes{0};
    const Vector<Vector<InnerIdType>>* delta_cliques{nullptr};
    const Vector<Vector<InnerIdType>>* delta_extra{nullptr};
    const Vector<Vector<InnerIdType>>* delta_node_cids{nullptr};
    const uint8_t* inactive_nodes{nullptr};
    const uint8_t* retired_cliques{nullptr};

    bool
    IsLiveNode(InnerIdType id) const {
        return id < total_nodes and inactive_nodes[id] == 0;
    }

    template <typename Visitor>
    void
    ForEachNodeClique(InnerIdType id, Visitor&& visit) const {
        if (not IsLiveNode(id)) {
            return;
        }
        if (id < base_node_count) {
            for (auto offset = p_node_to_cid[id]; offset < p_node_to_cid[id + 1]; ++offset) {
                const auto cid = node_to_cids[offset];
                if (retired_cliques[cid] == 0 and not visit(cid)) {
                    return;
                }
            }
        }
        for (auto cid : (*delta_node_cids)[id]) {
            if (retired_cliques[cid] == 0 and not visit(cid)) {
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

    void
    ResetDelta(uint64_t total);

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
    EnsureDeltaNodeRows(uint64_t total);

    void
    MarkUnavailable();

    void
    MarkAvailable(uint64_t total);

    [[nodiscard]] bool
    HasCliqueIndex(uint64_t total) const;

    [[nodiscard]] uint64_t
    TotalBaseCliqueCount() const {
        return total_clique_count_;
    }

    [[nodiscard]] uint64_t
    TotalLogicalCliqueCount() const;

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

    void
    Deserialize(StreamReader& reader, uint64_t format_version = 2);

    [[nodiscard]] uint64_t
    GetMemoryUsage() const;

    [[nodiscard]] bool
    TryGetBaseView(uint64_t total, CliqueDataCellBaseView& view) const;

    [[nodiscard]] bool
    TryGetSearchView(uint64_t total, CliqueDataCellSearchView& view) const;

    [[nodiscard]] CliqueDataCellStats
    CollectStats(uint64_t total) const;

private:
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
    append_new_clique_unlocked(const Vector<InnerIdType>& members, uint64_t total);

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
    std::atomic<uint64_t> available_total_{0};

    mutable std::shared_mutex mutex_;
};

}  // namespace vsag
