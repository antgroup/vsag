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
      delta_node_to_cids_(allocator) {
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
    ResetDelta(total);
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
    validate(total);
    ResetDelta(total);
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
CliqueDataCell::EnsureDeltaNodeRows(uint64_t total) {
    while (delta_node_to_cids_.size() < total) {
        delta_node_to_cids_.emplace_back(allocator_);
    }
    if (delta_clique_extra_.size() < total_clique_count_) {
        delta_clique_extra_.resize(total_clique_count_, Vector<InnerIdType>(allocator_));
    }
}

bool
CliqueDataCell::HasCliqueIndex(uint64_t total) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return TotalLogicalCliqueCount() > 0 and p_maxc_.size() == total_clique_count_ + 1 and
           p_node_to_cid_.size() <= total + 1 and delta_node_to_cids_.size() == total;
}

uint64_t
CliqueDataCell::TotalLogicalCliqueCount() const {
    return total_clique_count_ + delta_cliques_.size();
}

void
CliqueDataCell::CollectNodeCliqueIds(InnerIdType node_id, Vector<InnerIdType>& clique_ids) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    const auto logical_count = TotalLogicalCliqueCount();
    if (node_id + 1 < p_node_to_cid_.size()) {
        const auto begin = p_node_to_cid_[node_id];
        const auto end = p_node_to_cid_[node_id + 1];
        for (auto offset = begin; offset < end; ++offset) {
            const auto cid = node_to_cids_[offset];
            if (cid < logical_count) {
                clique_ids.push_back(cid);
            }
        }
    }
    if (node_id < delta_node_to_cids_.size()) {
        for (auto cid : delta_node_to_cids_[node_id]) {
            if (cid < logical_count) {
                clique_ids.push_back(cid);
            }
        }
    }
}

void
CliqueDataCell::GetCliqueMembers(InnerIdType clique_id, Vector<InnerIdType>& members) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (clique_id < total_clique_count_) {
        const auto begin = p_maxc_[clique_id];
        const auto end = p_maxc_[clique_id + 1];
        members.insert(members.end(), maxcs_.begin() + begin, maxcs_.begin() + end);
        if (clique_id < delta_clique_extra_.size()) {
            members.insert(members.end(),
                           delta_clique_extra_[clique_id].begin(),
                           delta_clique_extra_[clique_id].end());
        }
    } else {
        const auto delta_id = clique_id - total_clique_count_;
        if (delta_id < delta_cliques_.size()) {
            members.insert(
                members.end(), delta_cliques_[delta_id].begin(), delta_cliques_[delta_id].end());
        }
    }
}

uint64_t
CliqueDataCell::GetCliqueMemberCount(InnerIdType clique_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (clique_id < total_clique_count_) {
        uint64_t count = p_maxc_[clique_id + 1] - p_maxc_[clique_id];
        if (clique_id < delta_clique_extra_.size()) {
            count += delta_clique_extra_[clique_id].size();
        }
        return count;
    }
    const auto delta_id = clique_id - total_clique_count_;
    return delta_id < delta_cliques_.size() ? delta_cliques_[delta_id].size() : 0;
}

void
CliqueDataCell::AppendNodeToClique(InnerIdType node_id, InnerIdType clique_id, uint64_t total) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    EnsureDeltaNodeRows(total);
    if (clique_id >= TotalLogicalCliqueCount() or node_id >= delta_node_to_cids_.size()) {
        return;
    }
    auto& node_cliques = delta_node_to_cids_[node_id];
    if (std::find(node_cliques.begin(), node_cliques.end(), clique_id) != node_cliques.end()) {
        return;
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
}

void
CliqueDataCell::AppendNewClique(const Vector<InnerIdType>& members, uint64_t total) {
    if (members.empty()) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    EnsureDeltaNodeRows(total);
    const auto new_clique_id = static_cast<InnerIdType>(TotalLogicalCliqueCount());
    Vector<InnerIdType> normalized(allocator_);
    normalized.assign(members.begin(), members.end());
    std::sort(normalized.begin(), normalized.end());
    normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
    delta_cliques_.push_back(std::move(normalized));
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
}

void
CliqueDataCell::Deserialize(StreamReader& reader) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    StreamReader::ReadVector(reader, p_maxc_);
    StreamReader::ReadVector(reader, maxcs_);
    StreamReader::ReadVector(reader, p_node_to_cid_);
    StreamReader::ReadVector(reader, node_to_cids_);
    StreamReader::ReadObj(reader, total_clique_count_);
    read_nested_vector(reader, delta_cliques_, allocator_);
    read_nested_vector(reader, delta_clique_extra_, allocator_);
    read_nested_vector(reader, delta_node_to_cids_, allocator_);
    const auto total = p_node_to_cid_.empty() ? 0 : p_node_to_cid_.size() - 1;
    validate(total);
}

uint64_t
CliqueDataCell::GetMemoryUsage() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return sizeof(CliqueDataCell) + p_maxc_.capacity() * sizeof(InnerIdType) +
           maxcs_.capacity() * sizeof(InnerIdType) +
           p_node_to_cid_.capacity() * sizeof(InnerIdType) +
           node_to_cids_.capacity() * sizeof(InnerIdType) + nested_vector_memory(delta_cliques_) +
           nested_vector_memory(delta_clique_extra_) + nested_vector_memory(delta_node_to_cids_);
}

bool
CliqueDataCell::TryGetBaseView(uint64_t total, CliqueDataCellBaseView& view) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    const auto has_index =
        TotalLogicalCliqueCount() > 0 and p_maxc_.size() == total_clique_count_ + 1 and
        p_node_to_cid_.size() == total + 1 and delta_node_to_cids_.size() == total;
    if (not has_index or not delta_cliques_.empty()) {
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
    stats.base_membership_count = node_to_cids_.size();
    stats.has_index = stats.total_clique_count > 0 and p_maxc_.size() == total_clique_count_ + 1 and
                      p_node_to_cid_.size() <= total + 1 and delta_node_to_cids_.size() == total;

    Vector<uint64_t> node_memberships(total, 0, allocator_);
    const auto base_node_count = p_node_to_cid_.empty() ? 0 : p_node_to_cid_.size() - 1;
    for (uint64_t node_id = 0; node_id < std::min<uint64_t>(total, base_node_count); ++node_id) {
        node_memberships[node_id] += p_node_to_cid_[node_id + 1] - p_node_to_cid_[node_id];
    }

    for (uint64_t clique_id = 0; clique_id < total_clique_count_; ++clique_id) {
        uint64_t clique_size = p_maxc_[clique_id + 1] - p_maxc_[clique_id];
        if (clique_id < delta_clique_extra_.size()) {
            clique_size += delta_clique_extra_[clique_id].size();
            stats.delta_extra_membership_count += delta_clique_extra_[clique_id].size();
        }
        stats.max_clique_size = std::max(stats.max_clique_size, clique_size);
    }

    for (const auto& clique : delta_cliques_) {
        stats.delta_clique_membership_count += clique.size();
        stats.max_clique_size = std::max<uint64_t>(stats.max_clique_size, clique.size());
    }

    for (uint64_t node_id = 0; node_id < delta_node_to_cids_.size(); ++node_id) {
        if (node_id < node_memberships.size()) {
            node_memberships[node_id] += delta_node_to_cids_[node_id].size();
        }
    }

    stats.total_membership_count = stats.base_membership_count +
                                   stats.delta_extra_membership_count +
                                   stats.delta_clique_membership_count;
    for (auto membership : node_memberships) {
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
    if (stats.total_clique_count > 0) {
        stats.avg_clique_size = static_cast<double>(stats.total_membership_count) /
                                static_cast<double>(stats.total_clique_count);
    }
    return stats;
}

void
CliqueDataCell::validate(uint64_t total) const {
    CHECK_ARGUMENT(not p_maxc_.empty(), "clique datacell pMaxC must not be empty");
    CHECK_ARGUMENT(p_node_to_cid_.size() == total + 1,
                   fmt::format("clique datacell pNodeToCid size {} must be total + 1 ({})",
                               p_node_to_cid_.size(),
                               total + 1));
    CHECK_ARGUMENT(p_maxc_.front() == 0 and p_node_to_cid_.front() == 0,
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
}

}  // namespace vsag
