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

#include "sparse_duplicate_tracker.h"

#include <algorithm>

namespace vsag {

namespace {

auto
collect_group_members(const UnorderedMap<InnerIdType, InnerIdType>& next_ids, InnerIdType start_id)
    -> std::vector<InnerIdType> {
    auto iter = next_ids.find(start_id);
    if (iter == next_ids.end()) {
        return {};
    }

    std::vector<InnerIdType> members;
    members.push_back(start_id);
    auto current_id = iter->second;
    while (current_id != start_id) {
        members.push_back(current_id);
        current_id = next_ids.at(current_id);
    }
    return members;
}

auto
canonical_group_id(const std::vector<InnerIdType>& members) -> InnerIdType {
    return *std::min_element(members.begin(), members.end());
}

void
rebuild_group(UnorderedMap<InnerIdType, InnerIdType>& next_ids,
              size_t& duplicate_group_count,
              const std::vector<InnerIdType>& ordered_members) {
    if (ordered_members.size() < 2) {
        return;
    }

    next_ids[ordered_members[0]] = ordered_members[1];
    for (size_t i = 1; i + 1 < ordered_members.size(); ++i) {
        next_ids[ordered_members[i]] = ordered_members[i + 1];
    }
    next_ids[ordered_members.back()] = ordered_members[0];
    duplicate_group_count++;
}

void
write_group(StreamWriter& writer,
            const UnorderedMap<InnerIdType, InnerIdType>& next_ids,
            InnerIdType group_id,
            Allocator* allocator) {
    auto members = collect_group_members(next_ids, group_id);

    StreamWriter::WriteObj(writer, group_id);

    Vector<InnerIdType> duplicates(allocator);
    duplicates.reserve(members.size() - 1);
    for (size_t i = 1; i < members.size(); ++i) {
        duplicates.push_back(members[i]);
    }
    StreamWriter::WriteVector(writer, duplicates);
}

auto
to_ordered_members(const std::vector<InnerIdType>& members) -> std::vector<InnerIdType> {
    if (members.size() < 2) {
        return members;
    }

    auto group_id = canonical_group_id(members);
    std::vector<InnerIdType> ordered_members;
    ordered_members.reserve(members.size());
    ordered_members.push_back(group_id);
    for (const auto& member : members) {
        if (member != group_id) {
            ordered_members.push_back(member);
        }
    }
    return ordered_members;
}

}  // namespace

SparseDuplicateTracker::SparseDuplicateTracker(Allocator* allocator)
    : allocator_(allocator), next_ids_(0, allocator_) {
}

void
SparseDuplicateTracker::SetDuplicateId(InnerIdType group_id, InnerIdType duplicate_id) {
    std::scoped_lock lock(mutex_);

    if (group_id == duplicate_id || next_ids_.count(duplicate_id) > 0) {
        return;
    }

    if (next_ids_.count(group_id) == 0) {
        next_ids_[group_id] = duplicate_id;
        next_ids_[duplicate_id] = group_id;
        duplicate_count_++;
        return;
    }

    next_ids_[duplicate_id] = next_ids_[group_id];
    next_ids_[group_id] = duplicate_id;
}

auto
SparseDuplicateTracker::GetDuplicateIds(InnerIdType id) const -> std::vector<InnerIdType> {
    std::shared_lock lock(mutex_);

    auto iter = next_ids_.find(id);
    if (iter == next_ids_.end()) {
        return {};
    }

    std::vector<InnerIdType> result;
    auto current_id = iter->second;
    while (current_id != id) {
        result.push_back(current_id);
        current_id = next_ids_.at(current_id);
    }

    return result;
}

auto
SparseDuplicateTracker::GetGroupId(InnerIdType id) const -> InnerIdType {
    std::shared_lock lock(mutex_);

    auto iter = next_ids_.find(id);
    if (iter == next_ids_.end()) {
        return id;
    }

    InnerIdType group_id = id;
    auto current_id = iter->second;
    while (current_id != id) {
        group_id = std::min(group_id, current_id);
        current_id = next_ids_.at(current_id);
    }
    return group_id;
}

void
SparseDuplicateTracker::Serialize(StreamWriter& writer) const {
    std::shared_lock lock(mutex_);

    StreamWriter::WriteObj(writer, duplicate_count_);

    UnorderedSet<InnerIdType> visited(0, allocator_);
    for (const auto& [id, next_id] : next_ids_) {
        (void)next_id;
        if (visited.contains(id)) {
            continue;
        }

        auto members = collect_group_members(next_ids_, id);
        auto group_id = canonical_group_id(members);
        for (const auto& member : members) {
            visited.insert(member);
        }
        write_group(writer, next_ids_, group_id, allocator_);
    }
}

void
SparseDuplicateTracker::Deserialize(StreamReader& reader) {
    std::scoped_lock lock(mutex_);

    if (has_deserialized_) {
        return;
    }

    next_ids_.clear();
    StreamReader::ReadObj(reader, duplicate_count_);

    size_t group_count = duplicate_count_;
    duplicate_count_ = 0;
    for (size_t i = 0; i < group_count; ++i) {
        InnerIdType group_id;
        StreamReader::ReadObj(reader, group_id);
        Vector<InnerIdType> dup_list(allocator_);
        StreamReader::ReadVector(reader, dup_list);

        std::vector<InnerIdType> members;
        members.reserve(dup_list.size() + 1);
        members.push_back(group_id);
        members.insert(members.end(), dup_list.begin(), dup_list.end());
        rebuild_group(next_ids_, duplicate_count_, to_ordered_members(members));
    }
    has_deserialized_ = true;
}

void
SparseDuplicateTracker::DeserializeFromLegacyFormat(StreamReader& reader, size_t total_size) {
    std::scoped_lock lock(mutex_);

    (void)total_size;

    if (has_deserialized_) {
        return;
    }

    next_ids_.clear();
    StreamReader::ReadObj(reader, duplicate_count_);
    size_t legacy_group_count = duplicate_count_;
    duplicate_count_ = 0;
    for (size_t i = 0; i < legacy_group_count; ++i) {
        InnerIdType legacy_group_id;
        StreamReader::ReadObj(reader, legacy_group_id);
        Vector<InnerIdType> dup_list(allocator_);
        StreamReader::ReadVector(reader, dup_list);

        if (dup_list.empty()) {
            continue;
        }

        std::vector<InnerIdType> members;
        members.reserve(dup_list.size() + 1);
        members.push_back(legacy_group_id);
        members.insert(members.end(), dup_list.begin(), dup_list.end());
        rebuild_group(next_ids_, duplicate_count_, to_ordered_members(members));
    }
    has_deserialized_ = true;
}

}  // namespace vsag
