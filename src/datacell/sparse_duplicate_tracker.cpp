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

namespace vsag {

SparseDuplicateTracker::SparseDuplicateTracker(Allocator* allocator)
    : allocator_(allocator),
      original_to_duplicates_(0, allocator_),
      duplicate_to_original_(0, allocator_) {
}

void
SparseDuplicateTracker::SetDuplicateId(InnerIdType original_id, InnerIdType duplicate_id) {
    std::scoped_lock lock(mutex_);

    if (duplicate_to_original_.count(duplicate_id) > 0) {
        return;
    }

    InnerIdType root_id = original_id;
    if (duplicate_to_original_.count(original_id) > 0) {
        root_id = duplicate_to_original_[original_id];
    }

    auto& dup_list = original_to_duplicates_[root_id];
    bool is_first_duplicate = dup_list.empty();
    dup_list.push_back(duplicate_id);

    duplicate_to_original_[duplicate_id] = root_id;

    if (is_first_duplicate) {
        duplicate_count_++;
    }
}

auto
SparseDuplicateTracker::GetDuplicateIds(InnerIdType id) const -> std::vector<InnerIdType> {
    std::shared_lock lock(mutex_);

    InnerIdType root_id = id;
    if (duplicate_to_original_.count(id) > 0) {
        root_id = duplicate_to_original_.at(id);
    }

    if (original_to_duplicates_.count(root_id) == 0) {
        return {};
    }

    std::vector<InnerIdType> result;
    const auto& dup_list = original_to_duplicates_.at(root_id);
    for (const auto& dup_id : dup_list) {
        if (dup_id != id) {
            result.push_back(dup_id);
        }
    }

    if (root_id != id) {
        result.push_back(root_id);
    }

    return result;
}

void
SparseDuplicateTracker::Serialize(StreamWriter& writer) const {
    std::shared_lock lock(mutex_);

    StreamWriter::WriteObj(writer, duplicate_count_);

    StreamWriter::WriteObj(writer, original_to_duplicates_.size());
    for (const auto& [original_id, dup_list] : original_to_duplicates_) {
        StreamWriter::WriteObj(writer, original_id);
        StreamWriter::WriteVector(writer, dup_list);
    }
}

void
SparseDuplicateTracker::Deserialize(StreamReader& reader) {
    std::scoped_lock lock(mutex_);

    if (has_deserialized_) {
        return;
    }

    StreamReader::ReadObj(reader, duplicate_count_);

    size_t map_size;
    StreamReader::ReadObj(reader, map_size);

    for (size_t i = 0; i < map_size; ++i) {
        InnerIdType original_id;
        StreamReader::ReadObj(reader, original_id);
        std::vector<InnerIdType> dup_list;
        StreamReader::ReadVector(reader, dup_list);

        original_to_duplicates_[original_id] = std::move(dup_list);
        for (const auto& dup_id : original_to_duplicates_[original_id]) {
            duplicate_to_original_[dup_id] = original_id;
        }
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

    StreamReader::ReadObj(reader, duplicate_count_);
    for (size_t i = 0; i < duplicate_count_; ++i) {
        InnerIdType original_id;
        StreamReader::ReadObj(reader, original_id);
        std::vector<InnerIdType> dup_list;
        StreamReader::ReadVector(reader, dup_list);

        if (dup_list.empty()) {
            continue;
        }

        original_to_duplicates_[original_id] = dup_list;
        for (const auto& dup_id : dup_list) {
            duplicate_to_original_[dup_id] = original_id;
        }
    }
    has_deserialized_ = true;
}

}  // namespace vsag
