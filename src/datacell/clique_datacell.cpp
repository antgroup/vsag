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

#include "clique_datacell.h"

#include <fmt/format.h>

#include <algorithm>
#include <limits>
#include <utility>

#include "common.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"

namespace vsag {
namespace {

uint64_t
nested_vector_memory(const Vector<Vector<InnerIdType>>& values) {
    uint64_t memory = values.capacity() * sizeof(Vector<InnerIdType>);
    for (const auto& row : values) {
        memory += row.capacity() * sizeof(InnerIdType);
    }
    return memory;
}

void
write_nested_vector(StreamWriter& writer, const Vector<Vector<InnerIdType>>& values) {
    StreamWriter::WriteObj(writer, static_cast<uint64_t>(values.size()));
    for (const auto& row : values) {
        StreamWriter::WriteVector(writer, row);
    }
}

void
read_nested_vector(StreamReader& reader,
                   Vector<Vector<InnerIdType>>& values,
                   Allocator* allocator) {
    uint64_t size = 0;
    StreamReader::ReadObj(reader, size);
    values.clear();
    values.reserve(size);
    for (uint64_t i = 0; i < size; ++i) {
        values.emplace_back(allocator);
        StreamReader::ReadVector(reader, values.back());
    }
}

}  // namespace

CliqueDataCell::CliqueDataCell(Allocator* allocator)
    : allocator_(allocator),
      p_maxc_(allocator),
      maxcs_(allocator),
      p_node_to_cid_(allocator),
      node_to_cids_(allocator),
      delta_cliques_(allocator),
      delta_clique_extra_(allocator),
      delta_node_to_cids_(allocator),
      inactive_nodes_(allocator),
      retired_cliques_(allocator) {
    p_maxc_.push_back(0);
    p_node_to_cid_.push_back(0);
}

void
CliqueDataCell::Clear(uint64_t total) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    p_maxc_.clear();
    maxcs_.clear();
    p_node_to_cid_.clear();
    node_to_cids_.clear();
    p_maxc_.push_back(0);
    p_node_to_cid_.assign(total + 1, 0);
    total_clique_count_ = 0;
    active_clique_count_ = 0;
    inactive_node_count_ = 0;
    retired_clique_count_ = 0;
    ResetDelta(total);
    inactive_nodes_.assign(total, 0);
    retired_cliques_.clear();
    available_total_.store(0, std::memory_order_release);
}

void
CliqueDataCell::Assign(Vector<InnerIdType>&& p_maxc,
                       Vector<InnerIdType>&& maxcs,
                       Vector<InnerIdType>&& p_node_to_cid,
                       Vector<InnerIdType>&& node_to_cids,
                       uint64_t total) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    p_maxc_ = std::move(p_maxc);
    maxcs_ = std::move(maxcs);
    p_node_to_cid_ = std::move(p_node_to_cid);
    node_to_cids_ = std::move(node_to_cids);
    total_clique_count_ = p_maxc_.empty() ? 0 : p_maxc_.size() - 1;
    active_clique_count_ = total_clique_count_;
    inactive_node_count_ = 0;
    retired_clique_count_ = 0;
    ResetDelta(total);
    inactive_nodes_.assign(total, 0);
    retired_cliques_.assign(total_clique_count_, 0);
    validate(total);
    available_total_.store(total, std::memory_order_release);
}

void
CliqueDataCell::ResetDelta(uint64_t total) {
    delta_cliques_.clear();
    delta_clique_extra_.clear();
    delta_clique_extra_.resize(total_clique_count_, Vector<InnerIdType>(allocator_));
    delta_node_to_cids_.clear();
    delta_node_to_cids_.reserve(total);
    for (uint64_t i = 0; i < total; ++i) {
        delta_node_to_cids_.emplace_back(allocator_);
    }
}

void
CliqueDataCell::Flush(uint64_t total) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    CHECK_ARGUMENT(delta_node_to_cids_.size() == total, "cannot flush an incomplete MCI companion");
    compact_unlocked(total, nullptr);
}

void
CliqueDataCell::RemapNodes(const Vector<InnerIdType>& old_to_new, uint64_t total) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    CHECK_ARGUMENT(old_to_new.size() == inactive_nodes_.size(), "invalid MCI remap size");
    compact_unlocked(total, &old_to_new);
    available_total_.store(0, std::memory_order_release);
}

Vector<InnerIdType>
CliqueDataCell::GetInactiveNodeIds() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    Vector<InnerIdType> ids(allocator_);
    ids.reserve(inactive_node_count_);
    for (uint64_t id = 0; id < inactive_nodes_.size(); ++id) {
        if (inactive_nodes_[id] != 0) {
            ids.push_back(static_cast<InnerIdType>(id));
        }
    }
    return ids;
}

void
CliqueDataCell::compact_unlocked(uint64_t total, const Vector<InnerIdType>* old_to_new) {
    Vector<uint8_t> new_inactive(allocator_);
    uint64_t inactive_count = 0;
    if (old_to_new != nullptr) {
        new_inactive.assign(total, 0);
        Vector<uint8_t> seen(total, 0, allocator_);
        uint64_t mapped_count = 0;
        for (uint64_t old_id = 0; old_id < old_to_new->size(); ++old_id) {
            const auto id = (*old_to_new)[old_id];
            if (id == std::numeric_limits<InnerIdType>::max()) {
                continue;
            }
            CHECK_ARGUMENT(id < total, "MCI remapped node exceeds total");
            CHECK_ARGUMENT(seen[id] == 0, "duplicate MCI remapped node");
            seen[id] = 1;
            ++mapped_count;
            new_inactive[id] = inactive_nodes_[old_id];
            inactive_count += static_cast<uint64_t>(new_inactive[id] != 0);
        }
        CHECK_ARGUMENT(mapped_count == total, "incomplete MCI node permutation");
    }
    Vector<InnerIdType> offsets(allocator_);
    Vector<InnerIdType> members(allocator_);
    Vector<InnerIdType> inverse_offsets(total + 1, 0, allocator_);
    offsets.push_back(0);
    Vector<InnerIdType> row(allocator_);
    for (uint64_t cid = 0; cid < total_logical_clique_count_unlocked(); ++cid) {
        row.clear();
        get_clique_members_unlocked(static_cast<InnerIdType>(cid), row);
        if (old_to_new != nullptr) {
            uint64_t live_count = 0;
            for (auto id : row) {
                const auto mapped = (*old_to_new)[id];
                if (mapped != std::numeric_limits<InnerIdType>::max()) {
                    row[live_count++] = mapped;
                }
            }
            row.resize(live_count);
        }
        if (row.empty()) {
            continue;
        }
        CHECK_ARGUMENT(members.size() + row.size() <= std::numeric_limits<InnerIdType>::max(),
                       "MCI CSR memberships exceed offset capacity");
        members.insert(members.end(), row.begin(), row.end());
        offsets.push_back(static_cast<InnerIdType>(members.size()));
        for (auto id : row) {
            ++inverse_offsets[id + 1];
        }
    }
    for (uint64_t id = 0; id < total; ++id) {
        inverse_offsets[id + 1] += inverse_offsets[id];
    }
    Vector<InnerIdType> inverse(members.size(), 0, allocator_);
    Vector<InnerIdType> cursor(inverse_offsets, allocator_);
    const auto count = offsets.size() - 1;
    for (uint64_t cid = 0; cid < count; ++cid) {
        for (auto offset = offsets[cid]; offset < offsets[cid + 1]; ++offset) {
            inverse[cursor[members[offset]]++] = static_cast<InnerIdType>(cid);
        }
    }
    // Allocate every replacement before publishing, so allocation failure leaves the index intact.
    Vector<Vector<InnerIdType>> new_delta(allocator_);
    Vector<Vector<InnerIdType>> new_extra(count, Vector<InnerIdType>(allocator_), allocator_);
    Vector<Vector<InnerIdType>> new_node_delta(total, Vector<InnerIdType>(allocator_), allocator_);
    Vector<uint8_t> new_retired(count, 0, allocator_);
    p_maxc_.swap(offsets);
    maxcs_.swap(members);
    p_node_to_cid_.swap(inverse_offsets);
    node_to_cids_.swap(inverse);
    delta_cliques_.swap(new_delta);
    delta_clique_extra_.swap(new_extra);
    delta_node_to_cids_.swap(new_node_delta);
    retired_cliques_.swap(new_retired);
    total_clique_count_ = count;
    active_clique_count_ = count;
    retired_clique_count_ = 0;
    if (old_to_new != nullptr) {
        inactive_nodes_.swap(new_inactive);
        inactive_node_count_ = inactive_count;
    }
    // Plain Flush keeps the inactive mask; physical compaction remaps surviving tombstones.
}

void
CliqueDataCell::EnsureDeltaNodeRows(uint64_t total) {
    while (delta_node_to_cids_.size() < total) {
        delta_node_to_cids_.emplace_back(allocator_);
    }
    if (delta_clique_extra_.size() < total_clique_count_) {
        delta_clique_extra_.resize(total_clique_count_, Vector<InnerIdType>(allocator_));
    }
    if (inactive_nodes_.size() < total) {
        inactive_nodes_.resize(total, 0);
    }
    const auto logical_count = total_logical_clique_count_unlocked();
    if (retired_cliques_.size() < logical_count) {
        retired_cliques_.resize(logical_count, 0);
    }
}

void
CliqueDataCell::MarkUnavailable() {
    available_total_.store(0, std::memory_order_release);
}

void
CliqueDataCell::MarkAvailable(uint64_t total) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    CHECK_ARGUMENT(delta_node_to_cids_.size() == total,
                   "cannot publish an incomplete MCI companion");
    available_total_.store(total, std::memory_order_release);
}

bool
CliqueDataCell::HasCliqueIndex(uint64_t total) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (available_total_.load(std::memory_order_acquire) != total or
        p_maxc_.size() != total_clique_count_ + 1 or p_node_to_cid_.size() > total + 1 or
        delta_node_to_cids_.size() != total) {
        return false;
    }
    return active_clique_count_ > 0;
}

uint64_t
CliqueDataCell::TotalLogicalCliqueCount() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return total_logical_clique_count_unlocked();
}

uint64_t
CliqueDataCell::total_logical_clique_count_unlocked() const {
    return total_clique_count_ + delta_cliques_.size();
}

bool
CliqueDataCell::is_node_inactive_unlocked(InnerIdType node_id) const {
    return node_id < inactive_nodes_.size() and inactive_nodes_[node_id] != 0;
}

bool
CliqueDataCell::is_clique_retired_unlocked(InnerIdType clique_id) const {
    return clique_id < retired_cliques_.size() and retired_cliques_[clique_id] != 0;
}

void
CliqueDataCell::collect_node_clique_ids_unlocked(InnerIdType node_id,
                                                 Vector<InnerIdType>& clique_ids) const {
    if (is_node_inactive_unlocked(node_id)) {
        return;
    }
    const auto logical_count = total_logical_clique_count_unlocked();
    if (node_id + 1 < p_node_to_cid_.size()) {
        const auto begin = p_node_to_cid_[node_id];
        const auto end = p_node_to_cid_[node_id + 1];
        for (auto offset = begin; offset < end; ++offset) {
            const auto cid = node_to_cids_[offset];
            if (cid < logical_count and not is_clique_retired_unlocked(cid)) {
                clique_ids.push_back(cid);
            }
        }
    }
    if (node_id < delta_node_to_cids_.size()) {
        for (auto cid : delta_node_to_cids_[node_id]) {
            if (cid < logical_count and not is_clique_retired_unlocked(cid)) {
                clique_ids.push_back(cid);
            }
        }
    }
}

void
CliqueDataCell::CollectNodeCliqueIds(InnerIdType node_id, Vector<InnerIdType>& clique_ids) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    collect_node_clique_ids_unlocked(node_id, clique_ids);
}

void
CliqueDataCell::get_clique_members_unlocked(InnerIdType clique_id,
                                            Vector<InnerIdType>& members) const {
    if (is_clique_retired_unlocked(clique_id)) {
        return;
    }
    auto append_live = [&](auto begin, auto end) {
        for (auto iter = begin; iter != end; ++iter) {
            if (not is_node_inactive_unlocked(*iter)) {
                members.push_back(*iter);
            }
        }
    };
    if (clique_id < total_clique_count_) {
        const auto begin = p_maxc_[clique_id];
        const auto end = p_maxc_[clique_id + 1];
        append_live(maxcs_.begin() + begin, maxcs_.begin() + end);
        if (clique_id < delta_clique_extra_.size()) {
            append_live(delta_clique_extra_[clique_id].begin(),
                        delta_clique_extra_[clique_id].end());
        }
    } else {
        const auto delta_id = clique_id - total_clique_count_;
        if (delta_id < delta_cliques_.size()) {
            append_live(delta_cliques_[delta_id].begin(), delta_cliques_[delta_id].end());
        }
    }
}

void
CliqueDataCell::GetCliqueMembers(InnerIdType clique_id, Vector<InnerIdType>& members) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    get_clique_members_unlocked(clique_id, members);
}

uint64_t
CliqueDataCell::GetCliqueMemberCount(InnerIdType clique_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    Vector<InnerIdType> members(allocator_);
    get_clique_members_unlocked(clique_id, members);
    return members.size();
}

bool
CliqueDataCell::AppendNodeToClique(InnerIdType node_id,
                                   InnerIdType clique_id,
                                   uint64_t total,
                                   uint64_t max_members) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    EnsureDeltaNodeRows(total);
    if (clique_id >= total_logical_clique_count_unlocked() or
        node_id >= delta_node_to_cids_.size() or is_node_inactive_unlocked(node_id) or
        is_clique_retired_unlocked(clique_id)) {
        return false;
    }
    Vector<InnerIdType> active_members(allocator_);
    get_clique_members_unlocked(clique_id, active_members);
    if (active_members.size() >= max_members or
        std::find(active_members.begin(), active_members.end(), node_id) != active_members.end()) {
        return false;
    }
    auto& node_cliques = delta_node_to_cids_[node_id];
    if (std::find(node_cliques.begin(), node_cliques.end(), clique_id) != node_cliques.end()) {
        return false;
    }
    if (clique_id < total_clique_count_) {
        auto& members = delta_clique_extra_[clique_id];
        if (std::find(members.begin(), members.end(), node_id) == members.end()) {
            members.push_back(node_id);
        }
    } else {
        auto& members = delta_cliques_[clique_id - total_clique_count_];
        if (std::find(members.begin(), members.end(), node_id) == members.end()) {
            members.push_back(node_id);
        }
    }
    node_cliques.push_back(clique_id);
    return true;
}

void
CliqueDataCell::AppendNewClique(const Vector<InnerIdType>& members, uint64_t total) {
    if (members.empty()) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    append_new_clique_unlocked(members, total);
}

void
CliqueDataCell::append_new_clique_unlocked(const Vector<InnerIdType>& members, uint64_t total) {
    EnsureDeltaNodeRows(total);
    const auto new_clique_id = static_cast<InnerIdType>(total_logical_clique_count_unlocked());
    Vector<InnerIdType> normalized(allocator_);
    normalized.reserve(members.size());
    for (auto node_id : members) {
        if (node_id < total and not is_node_inactive_unlocked(node_id)) {
            normalized.push_back(node_id);
        }
    }
    std::sort(normalized.begin(), normalized.end());
    normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
    if (normalized.empty()) {
        return;
    }
    delta_cliques_.push_back(std::move(normalized));
    retired_cliques_.push_back(0);
    ++active_clique_count_;
    for (auto node_id : delta_cliques_.back()) {
        if (node_id >= delta_node_to_cids_.size()) {
            continue;
        }
        auto& node_cliques = delta_node_to_cids_[node_id];
        if (std::find(node_cliques.begin(), node_cliques.end(), new_clique_id) ==
            node_cliques.end()) {
            node_cliques.push_back(new_clique_id);
        }
    }
}

MCIDeleteSnapshot
CliqueDataCell::PrepareDelete(const Vector<InnerIdType>& node_ids,
                              uint64_t clique_size_threshold,
                              uint64_t node_mct_threshold) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    MCIDeleteSnapshot snapshot(allocator_);
    UnorderedSet<InnerIdType> deleting(allocator_);
    deleting.reserve(node_ids.size());
    for (auto node_id : node_ids) {
        if (node_id < inactive_nodes_.size() and not is_node_inactive_unlocked(node_id)) {
            deleting.insert(node_id);
            collect_node_clique_ids_unlocked(node_id, snapshot.affected_clique_ids);
        }
    }
    std::sort(snapshot.affected_clique_ids.begin(), snapshot.affected_clique_ids.end());
    snapshot.affected_clique_ids.erase(
        std::unique(snapshot.affected_clique_ids.begin(), snapshot.affected_clique_ids.end()),
        snapshot.affected_clique_ids.end());

    UnorderedSet<InnerIdType> repair_candidates(allocator_);
    Vector<InnerIdType> members(allocator_);
    for (auto clique_id : snapshot.affected_clique_ids) {
        members.clear();
        get_clique_members_unlocked(clique_id, members);
        uint64_t remaining_count = 0;
        for (auto member : members) {
            if (deleting.count(member) == 0) {
                ++remaining_count;
            }
        }
        if (remaining_count >= clique_size_threshold) {
            continue;
        }
        snapshot.retired_clique_ids.push_back(clique_id);
        for (auto member : members) {
            if (deleting.count(member) == 0) {
                repair_candidates.insert(member);
            }
        }
    }

    UnorderedSet<InnerIdType> retiring(allocator_);
    retiring.reserve(snapshot.retired_clique_ids.size());
    for (auto clique_id : snapshot.retired_clique_ids) {
        retiring.insert(clique_id);
    }
    Vector<InnerIdType> clique_ids(allocator_);
    for (auto node_id : repair_candidates) {
        clique_ids.clear();
        collect_node_clique_ids_unlocked(node_id, clique_ids);
        uint64_t projected_mct = 0;
        for (auto clique_id : clique_ids) {
            if (retiring.count(clique_id) == 0) {
                ++projected_mct;
            }
        }
        if (projected_mct < node_mct_threshold) {
            snapshot.repair_node_ids.push_back(node_id);
        }
    }
    std::sort(snapshot.repair_node_ids.begin(), snapshot.repair_node_ids.end());
    return snapshot;
}

void
CliqueDataCell::CommitDelete(const Vector<InnerIdType>& node_ids,
                             const Vector<InnerIdType>& retired_clique_ids,
                             uint64_t total) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    EnsureDeltaNodeRows(total);
    for (auto node_id : node_ids) {
        if (node_id < inactive_nodes_.size() and inactive_nodes_[node_id] == 0) {
            inactive_nodes_[node_id] = 1;
            ++inactive_node_count_;
        }
    }
    for (auto clique_id : retired_clique_ids) {
        if (clique_id < retired_cliques_.size() and retired_cliques_[clique_id] == 0) {
            retired_cliques_[clique_id] = 1;
            --active_clique_count_;
            ++retired_clique_count_;
        }
    }
}

void
CliqueDataCell::Serialize(StreamWriter& writer) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    StreamWriter::WriteVector(writer, p_maxc_);
    StreamWriter::WriteVector(writer, maxcs_);
    StreamWriter::WriteVector(writer, p_node_to_cid_);
    StreamWriter::WriteVector(writer, node_to_cids_);
    StreamWriter::WriteObj(writer, total_clique_count_);
    write_nested_vector(writer, delta_cliques_);
    write_nested_vector(writer, delta_clique_extra_);
    write_nested_vector(writer, delta_node_to_cids_);
    StreamWriter::WriteVector(writer, inactive_nodes_);
    StreamWriter::WriteVector(writer, retired_cliques_);
}

void
CliqueDataCell::Deserialize(StreamReader& reader, uint64_t format_version) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    StreamReader::ReadVector(reader, p_maxc_);
    StreamReader::ReadVector(reader, maxcs_);
    StreamReader::ReadVector(reader, p_node_to_cid_);
    StreamReader::ReadVector(reader, node_to_cids_);
    StreamReader::ReadObj(reader, total_clique_count_);
    read_nested_vector(reader, delta_cliques_, allocator_);
    read_nested_vector(reader, delta_clique_extra_, allocator_);
    read_nested_vector(reader, delta_node_to_cids_, allocator_);
    const auto base_total = p_node_to_cid_.empty() ? 0 : p_node_to_cid_.size() - 1;
    const auto total = std::max<uint64_t>(base_total, delta_node_to_cids_.size());
    if (format_version >= 2) {
        StreamReader::ReadVector(reader, inactive_nodes_);
        StreamReader::ReadVector(reader, retired_cliques_);
    } else {
        inactive_nodes_.assign(total, 0);
        retired_cliques_.assign(total_logical_clique_count_unlocked(), 0);
    }
    active_clique_count_ = 0;
    inactive_node_count_ = 0;
    retired_clique_count_ = 0;
    for (auto inactive : inactive_nodes_) {
        if (inactive != 0) {
            ++inactive_node_count_;
        }
    }
    for (auto retired : retired_cliques_) {
        if (retired == 0) {
            ++active_clique_count_;
        } else {
            ++retired_clique_count_;
        }
    }
    validate(total);
    available_total_.store(total, std::memory_order_release);
}

uint64_t
CliqueDataCell::GetMemoryUsage() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return sizeof(CliqueDataCell) + p_maxc_.capacity() * sizeof(InnerIdType) +
           maxcs_.capacity() * sizeof(InnerIdType) +
           p_node_to_cid_.capacity() * sizeof(InnerIdType) +
           node_to_cids_.capacity() * sizeof(InnerIdType) + nested_vector_memory(delta_cliques_) +
           nested_vector_memory(delta_clique_extra_) + nested_vector_memory(delta_node_to_cids_) +
           inactive_nodes_.capacity() * sizeof(uint8_t) +
           retired_cliques_.capacity() * sizeof(uint8_t);
}

bool
CliqueDataCell::TryGetBaseView(uint64_t total, CliqueDataCellBaseView& view) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    const auto has_index =
        available_total_.load(std::memory_order_acquire) == total and
        total_logical_clique_count_unlocked() > 0 and p_maxc_.size() == total_clique_count_ + 1 and
        p_node_to_cid_.size() == total + 1 and delta_node_to_cids_.size() == total;
    if (not has_index or not delta_cliques_.empty() or inactive_node_count_ != 0 or
        retired_clique_count_ != 0) {
        return false;
    }
    for (const auto& extra_members : delta_clique_extra_) {
        if (not extra_members.empty()) {
            return false;
        }
    }
    for (const auto& node_cliques : delta_node_to_cids_) {
        if (not node_cliques.empty()) {
            return false;
        }
    }

    view.p_maxc = p_maxc_.data();
    view.maxcs = maxcs_.data();
    view.p_node_to_cid = p_node_to_cid_.data();
    view.node_to_cids = node_to_cids_.data();
    view.total_clique_count = total_clique_count_;
    view.guard = std::move(lock);
    return true;
}

bool
CliqueDataCell::TryGetSearchView(uint64_t total, CliqueDataCellSearchView& view) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (available_total_.load(std::memory_order_acquire) != total or active_clique_count_ == 0 or
        delta_node_to_cids_.size() != total or inactive_nodes_.size() != total) {
        return false;
    }
    view.p_maxc = p_maxc_.data();
    view.maxcs = maxcs_.data();
    view.p_node_to_cid = p_node_to_cid_.data();
    view.node_to_cids = node_to_cids_.data();
    view.base_clique_count = total_clique_count_;
    view.base_node_count = p_node_to_cid_.size() - 1;
    view.total_clique_count = total_logical_clique_count_unlocked();
    view.total_nodes = total;
    view.delta_cliques = &delta_cliques_;
    view.delta_extra = &delta_clique_extra_;
    view.delta_node_cids = &delta_node_to_cids_;
    view.inactive_nodes = inactive_nodes_.data();
    view.retired_cliques = retired_cliques_.data();
    view.guard = std::move(lock);
    return true;
}

CliqueDataCellStats
CliqueDataCell::CollectStats(uint64_t total) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    CliqueDataCellStats stats;
    stats.total_nodes = total;
    stats.base_clique_count = total_clique_count_;
    stats.delta_clique_count = delta_cliques_.size();
    stats.total_clique_count = total_clique_count_ + delta_cliques_.size();
    stats.inactive_node_count = inactive_node_count_;
    stats.retired_clique_count = retired_clique_count_;
    Vector<uint64_t> node_memberships(total, 0, allocator_);
    Vector<InnerIdType> members(allocator_);
    uint64_t active_clique_count = 0;
    for (InnerIdType clique_id = 0; clique_id < stats.total_clique_count; ++clique_id) {
        if (is_clique_retired_unlocked(clique_id)) {
            continue;
        }
        members.clear();
        get_clique_members_unlocked(clique_id, members);
        if (members.empty()) {
            continue;
        }
        ++active_clique_count;
        stats.max_clique_size = std::max<uint64_t>(stats.max_clique_size, members.size());
        stats.total_membership_count += members.size();
        if (clique_id < total_clique_count_) {
            for (auto member : members) {
                const auto begin = p_maxc_[clique_id];
                const auto end = p_maxc_[clique_id + 1];
                if (std::find(maxcs_.begin() + begin, maxcs_.begin() + end, member) !=
                    maxcs_.begin() + end) {
                    ++stats.base_membership_count;
                } else {
                    ++stats.delta_extra_membership_count;
                }
            }
        } else {
            stats.delta_clique_membership_count += members.size();
        }
        for (auto member : members) {
            if (member < node_memberships.size()) {
                ++node_memberships[member];
            }
        }
    }
    for (uint64_t node_id = 0; node_id < node_memberships.size(); ++node_id) {
        if (is_node_inactive_unlocked(static_cast<InnerIdType>(node_id))) {
            continue;
        }
        const auto membership = node_memberships[node_id];
        if (membership > 0) {
            ++stats.covered_nodes;
        }
        stats.max_membership_per_node = std::max(stats.max_membership_per_node, membership);
    }
    if (total > 0) {
        stats.covered_node_ratio =
            static_cast<double>(stats.covered_nodes) / static_cast<double>(total);
        stats.avg_membership_per_node =
            static_cast<double>(stats.total_membership_count) / static_cast<double>(total);
    }
    if (active_clique_count > 0) {
        stats.avg_clique_size = static_cast<double>(stats.total_membership_count) /
                                static_cast<double>(active_clique_count);
    }
    stats.has_index = available_total_.load(std::memory_order_acquire) == total and
                      active_clique_count > 0 and p_maxc_.size() == total_clique_count_ + 1 and
                      p_node_to_cid_.size() <= total + 1 and delta_node_to_cids_.size() == total;
    return stats;
}

void
CliqueDataCell::validate(uint64_t total) const {
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        not p_maxc_.empty(),
        "clique datacell pMaxC must not be empty");
    CHECK_ARGUMENT(p_node_to_cid_.size() <= total + 1,
                   fmt::format("clique datacell pNodeToCid size {} must not exceed total + 1 ({})",
                               p_node_to_cid_.size(),
                               total + 1));
    CHECK_ARGUMENT(delta_node_to_cids_.size() == total,
                   "clique datacell delta node rows size mismatch");
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        p_maxc_.front() == 0 and p_node_to_cid_.front() == 0,
        "clique datacell CSR offsets must start from 0");
    CHECK_ARGUMENT(p_maxc_.back() == maxcs_.size(), "clique datacell pMaxC tail mismatch");
    CHECK_ARGUMENT(p_node_to_cid_.back() == node_to_cids_.size(),
                   "clique datacell pNodeToCid tail mismatch");
    CHECK_ARGUMENT(p_maxc_.size() == total_clique_count_ + 1,
                   "clique datacell pMaxC size inconsistent with clique count");
    CHECK_ARGUMENT(std::is_sorted(p_maxc_.begin(), p_maxc_.end()),
                   "clique datacell pMaxC offsets must be sorted");
    CHECK_ARGUMENT(std::is_sorted(p_node_to_cid_.begin(), p_node_to_cid_.end()),
                   "clique datacell pNodeToCid offsets must be sorted");
    CHECK_ARGUMENT(inactive_nodes_.size() == total,
                   "clique datacell inactive node mask size mismatch");
    CHECK_ARGUMENT(retired_cliques_.size() == total_logical_clique_count_unlocked(),
                   "clique datacell retired clique mask size mismatch");
}

}  // namespace vsag
