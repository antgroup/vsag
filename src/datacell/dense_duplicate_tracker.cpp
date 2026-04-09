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

#include "dense_duplicate_tracker.h"

#include <algorithm>

namespace vsag {

DenseDuplicateTracker::DenseDuplicateTracker(Allocator* allocator)
    : allocator_(allocator), duplicate_ids_(0, allocator_) {
}

void
DenseDuplicateTracker::SetDuplicateId(InnerIdType group_id, InnerIdType duplicate_id) {
    std::scoped_lock lock(mutex_);

    if (group_id == duplicate_id || duplicate_ids_.size() <= duplicate_id ||
        duplicate_ids_.size() <= group_id) {
        return;
    }

    if (duplicate_ids_[duplicate_id] != duplicate_id) {
        return;
    }

    if (duplicate_ids_[group_id] == group_id) {
        duplicate_count_++;
    }
    duplicate_ids_[duplicate_id] = duplicate_ids_[group_id];
    duplicate_ids_[group_id] = duplicate_id;
}

auto
DenseDuplicateTracker::GetDuplicateIds(InnerIdType id) const -> std::vector<InnerIdType> {
    std::shared_lock lock(mutex_);

    if (id >= duplicate_ids_.size()) {
        return {};
    }

    std::vector<InnerIdType> ids;
    auto current_id = id;
    while (duplicate_ids_[current_id] != id) {
        ids.push_back(duplicate_ids_[current_id]);
        current_id = duplicate_ids_[current_id];
    }
    return ids;
}

auto
DenseDuplicateTracker::GetGroupId(InnerIdType id) const -> InnerIdType {
    std::shared_lock lock(mutex_);

    if (id >= duplicate_ids_.size()) {
        return id;
    }

    InnerIdType group_id = id;
    auto current_id = duplicate_ids_[id];
    while (current_id != id) {
        group_id = std::min(group_id, current_id);
        current_id = duplicate_ids_[current_id];
    }
    return group_id;
}

void
DenseDuplicateTracker::Serialize(StreamWriter& writer) const {
    std::shared_lock lock(mutex_);

    StreamWriter::WriteObj(writer, duplicate_count_);
    size_t size = duplicate_ids_.size();
    StreamWriter::WriteObj(writer, size);

    Vector<bool> visited(size, false, allocator_);
    for (InnerIdType i = 0; i < size; ++i) {
        if (duplicate_ids_[i] != i && !visited[i]) {
            StreamWriter::WriteObj(writer, i);
            Vector<InnerIdType> id_list(allocator_);
            auto current_id = i;
            visited[current_id] = true;
            while (duplicate_ids_[current_id] != i) {
                id_list.push_back(duplicate_ids_[current_id]);
                current_id = duplicate_ids_[current_id];
                visited[current_id] = true;
            }
            StreamWriter::WriteVector(writer, id_list);
        }
    }
}

void
DenseDuplicateTracker::Deserialize(StreamReader& reader) {
    std::scoped_lock lock(mutex_);

    if (has_deserialized_) {
        return;
    }

    StreamReader::ReadObj(reader, duplicate_count_);
    size_t size;
    StreamReader::ReadObj(reader, size);
    duplicate_ids_.resize(size);
    for (InnerIdType i = 0; i < size; ++i) {
        duplicate_ids_[i] = i;
    }

    for (size_t i = 0; i < duplicate_count_; ++i) {
        InnerIdType head_id;
        StreamReader::ReadObj(reader, head_id);
        Vector<InnerIdType> id_list(allocator_);
        StreamReader::ReadVector(reader, id_list);

        auto current_id = head_id;
        for (const auto& dup_id : id_list) {
            duplicate_ids_[current_id] = dup_id;
            current_id = dup_id;
        }
        duplicate_ids_[current_id] = head_id;
    }
    has_deserialized_ = true;
}

void
DenseDuplicateTracker::DeserializeFromLegacyFormat(StreamReader& reader, size_t total_size) {
    std::scoped_lock lock(mutex_);

    if (has_deserialized_) {
        return;
    }

    StreamReader::ReadObj(reader, duplicate_count_);
    duplicate_ids_.resize(total_size);
    for (size_t i = 0; i < total_size; ++i) {
        duplicate_ids_[i] = static_cast<InnerIdType>(i);
    }

    for (InnerIdType i = 0; i < duplicate_count_; ++i) {
        InnerIdType id;
        StreamReader::ReadObj<InnerIdType>(reader, id);
        Vector<InnerIdType> id_list(allocator_);
        StreamReader::ReadVector(reader, id_list);
        auto current_id = id;
        for (const auto& duplicate_id : id_list) {
            duplicate_ids_[current_id] = duplicate_id;
            current_id = duplicate_id;
        }
        duplicate_ids_[current_id] = id;
    }
    has_deserialized_ = true;
}

void
DenseDuplicateTracker::Resize(InnerIdType new_size) {
    std::scoped_lock lock(mutex_);

    if (new_size <= duplicate_ids_.size()) {
        return;
    }

    size_t old_size = duplicate_ids_.size();
    duplicate_ids_.resize(new_size);
    for (size_t i = old_size; i < new_size; ++i) {
        duplicate_ids_[i] = static_cast<InnerIdType>(i);
    }
}

}  // namespace vsag
