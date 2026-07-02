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

#include "container_types.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

class StreamReader;
class StreamWriter;

DEFINE_POINTER(CliqueDataCell);

struct CliqueDataCellBaseView {
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

    void
    EnsureDeltaNodeRows(uint64_t total);

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

    void
    AppendNodeToClique(InnerIdType node_id, InnerIdType clique_id, uint64_t total);

    void
    AppendNewClique(const Vector<InnerIdType>& members, uint64_t total);

    void
    Serialize(StreamWriter& writer) const;

    void
    Deserialize(StreamReader& reader);

    [[nodiscard]] uint64_t
    GetMemoryUsage() const;

    [[nodiscard]] bool
    TryGetBaseView(uint64_t total, CliqueDataCellBaseView& view) const;

    [[nodiscard]] CliqueDataCellStats
    CollectStats(uint64_t total) const;

private:
    void
    validate(uint64_t total) const;

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

    mutable std::shared_mutex mutex_;
};

}  // namespace vsag
