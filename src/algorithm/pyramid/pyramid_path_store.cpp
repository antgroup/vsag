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

#include "pyramid_path_store.h"

#include <algorithm>
#include <limits>
#include <mutex>

#include "common.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "vsag_exception.h"

namespace vsag {

namespace {

constexpr uint64_t MISSING_PATH_OFFSET = std::numeric_limits<uint64_t>::max();

template <typename T>
void
read_pyramid_path_vector(StreamReader& reader, Vector<T>& values, uint64_t max_count) {
    uint64_t size = 0;
    StreamReader::ReadObj(reader, size);
    if (size > max_count) {
        throw VsagException(ErrorType::READ_ERROR, "corrupted Pyramid path vector size");
    }

    const auto cursor = reader.GetCursor();
    const auto reader_length = reader.Length();
    if (cursor > reader_length || size > (reader_length - cursor) / sizeof(T)) {
        throw VsagException(ErrorType::READ_ERROR,
                            "corrupted Pyramid path vector exceeds remaining payload");
    }
    if (size > values.max_size()) {
        throw VsagException(ErrorType::READ_ERROR, "Pyramid path vector is too large");
    }

    values.resize(size);
    if (size > 0) {
        reader.Read(reinterpret_cast<char*>(values.data()), size * sizeof(T));
    }
}

bool
are_pyramid_path_rows_valid(const Vector<uint64_t>& offsets,
                            const Vector<uint16_t>& counts,
                            uint64_t path_count) {
    if (offsets.size() != counts.size()) {
        return false;
    }
    for (uint64_t slot = 0; slot < offsets.size(); ++slot) {
        const auto path_offset = offsets[slot];
        const auto row_path_count = static_cast<uint64_t>(counts[slot]);
        if (path_offset == MISSING_PATH_OFFSET) {
            if (row_path_count != 0) {
                return false;
            }
            continue;
        }
        if (path_offset > path_count || row_path_count > path_count - path_offset) {
            return false;
        }
    }
    return true;
}

void
reserve_paths_geometrically(Vector<std::string>& paths, uint64_t required_capacity) {
    const auto current_capacity = static_cast<uint64_t>(paths.capacity());
    if (required_capacity <= current_capacity) {
        return;
    }

    const auto max_capacity = static_cast<uint64_t>(paths.max_size());
    auto target_capacity = std::max<uint64_t>(current_capacity, 1);
    if (target_capacity <= max_capacity / 2) {
        target_capacity *= 2;
    } else {
        target_capacity = max_capacity;
    }
    paths.reserve(std::max(required_capacity, target_capacity));
}

}  // namespace

std::string
ReadPyramidPathString(StreamReader& reader) {
    uint64_t length = 0;
    StreamReader::ReadObj(reader, length);
    const auto cursor = reader.GetCursor();
    const auto reader_length = reader.Length();
    if (cursor > reader_length || length > reader_length - cursor) {
        throw VsagException(ErrorType::READ_ERROR, "corrupted Pyramid path string length");
    }
    if (length > std::string{}.max_size()) {
        throw VsagException(ErrorType::READ_ERROR, "Pyramid path string is too large");
    }
    std::string value(length, '\0');
    if (length > 0) {
        reader.Read(value.data(), length);
    }
    return value;
}

void
PyramidPathStore::Writer::Prepare(uint64_t slot_count, uint64_t additional_path_count) {
    constexpr uint64_t max_slot_count =
        static_cast<uint64_t>(std::numeric_limits<InnerIdType>::max()) + 1;
    CHECK_ARGUMENT(slot_count <= max_slot_count, "Pyramid path slot count is too large");
    CHECK_ARGUMENT(slot_count <= store_.offsets_.max_size(),
                   "Pyramid path slot count is too large");
    CHECK_ARGUMENT(slot_count <= store_.counts_.max_size(), "Pyramid path slot count is too large");

    const auto path_count = static_cast<uint64_t>(store_.paths_.size());
    CHECK_ARGUMENT(additional_path_count <= store_.paths_.max_size() - path_count,
                   "Pyramid path count is too large");

    const auto old_offset_count = store_.offsets_.size();
    const auto old_count_count = store_.counts_.size();
    try {
        if (slot_count > old_offset_count) {
            store_.offsets_.resize(slot_count, MISSING_PATH_OFFSET);
        }
        if (slot_count > old_count_count) {
            store_.counts_.resize(slot_count, 0);
        }
        reserve_paths_geometrically(store_.paths_, path_count + additional_path_count);
    } catch (...) {
        store_.offsets_.resize(old_offset_count);
        store_.counts_.resize(old_count_count);
        throw;
    }
}

void
PyramidPathStore::Writer::Insert(InnerIdType inner_id, const std::string& path) {
    Insert(inner_id, &path, 1);
}

void
PyramidPathStore::Writer::Insert(InnerIdType inner_id,
                                 const std::string* paths,
                                 uint64_t path_count) {
    CHECK_ARGUMENT(path_count <= std::numeric_limits<uint16_t>::max(),
                   "too many Pyramid paths for one inner id");
    if (path_count > 0) {
        CHECK_ARGUMENT(paths != nullptr, "Pyramid paths must not be null");
    }
    const auto slot = static_cast<uint64_t>(inner_id);
    if (slot < store_.offsets_.size()) {
        CHECK_ARGUMENT(store_.offsets_[slot] == MISSING_PATH_OFFSET,
                       "inner id already has Pyramid paths");
    }

    const auto old_path_count = static_cast<uint64_t>(store_.paths_.size());
    const auto old_offset_count = store_.offsets_.size();
    const auto old_count_count = store_.counts_.size();
    try {
        Prepare(slot + 1, path_count);
        for (uint64_t offset = 0; offset < path_count; ++offset) {
            store_.paths_.emplace_back(paths[offset]);
        }
        store_.offsets_[slot] = old_path_count;
        store_.counts_[slot] = static_cast<uint16_t>(path_count);
    } catch (...) {
        store_.paths_.resize(old_path_count);
        store_.offsets_.resize(old_offset_count);
        store_.counts_.resize(old_count_count);
        throw;
    }
}

PyramidPathStore::Writer
PyramidPathStore::AcquireWriter() {
    return Writer(*this);
}

bool
PyramidPathStore::GetPaths(const Vector<InnerIdType>& inner_ids, std::string* paths) const {
    std::shared_lock lock(mutex_);
    if (offsets_.size() != counts_.size()) {
        return false;
    }
    for (uint64_t offset = 0; offset < inner_ids.size(); ++offset) {
        const auto inner_id = static_cast<uint64_t>(inner_ids[offset]);
        if (inner_id >= offsets_.size() || counts_[inner_id] != 1 ||
            offsets_[inner_id] >= paths_.size()) {
            return false;
        }
        paths[offset] = paths_[offsets_[inner_id]];
    }
    return true;
}

bool
PyramidPathStore::GetPathRows(const Vector<InnerIdType>& inner_ids,
                              std::vector<std::vector<std::string>>& path_rows) const {
    std::shared_lock lock(mutex_);
    if (offsets_.size() != counts_.size()) {
        return false;
    }
    path_rows.clear();
    path_rows.reserve(inner_ids.size());
    for (const auto inner_id : inner_ids) {
        const auto slot = static_cast<uint64_t>(inner_id);
        if (slot >= offsets_.size()) {
            return false;
        }
        const auto path_offset = offsets_[slot];
        const auto path_count = static_cast<uint64_t>(counts_[slot]);
        if (path_offset == MISSING_PATH_OFFSET || path_offset > paths_.size() ||
            path_count > paths_.size() - path_offset) {
            return false;
        }
        const auto begin_offset = static_cast<Vector<std::string>::difference_type>(path_offset);
        const auto end_offset =
            static_cast<Vector<std::string>::difference_type>(path_offset + path_count);
        path_rows.emplace_back(paths_.begin() + begin_offset, paths_.begin() + end_offset);
    }
    return true;
}

void
PyramidPathStore::Serialize(StreamWriter& writer) const {
    std::shared_lock lock(mutex_);
    if (offsets_.size() != counts_.size()) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "Pyramid path store has inconsistent slot arrays");
    }
    StreamWriter::WriteVector(writer, offsets_);
    StreamWriter::WriteVector(writer, counts_);
    StreamWriter::WriteObj(writer, static_cast<uint64_t>(paths_.size()));
    for (const auto& path : paths_) {
        StreamWriter::WriteString(writer, path);
    }
}

void
PyramidPathStore::Deserialize(StreamReader& reader, uint64_t max_count) {
    constexpr uint64_t max_slot_count =
        static_cast<uint64_t>(std::numeric_limits<InnerIdType>::max()) + 1;
    const auto max_slot_vector_count = std::min(max_count, max_slot_count);

    Vector<uint64_t> restored_offsets(allocator_);
    Vector<uint16_t> restored_counts(allocator_);
    read_pyramid_path_vector(reader, restored_offsets, max_slot_vector_count);
    read_pyramid_path_vector(reader, restored_counts, restored_offsets.size());
    if (restored_offsets.size() != restored_counts.size()) {
        throw VsagException(ErrorType::READ_ERROR,
                            "corrupted Pyramid path slot vector sizes do not match");
    }

    uint64_t path_count = 0;
    StreamReader::ReadObj(reader, path_count);
    Vector<std::string> restored_paths(allocator_);
    const auto path_cursor = reader.GetCursor();
    const auto reader_length = reader.Length();
    if (path_cursor > reader_length ||
        path_count > (reader_length - path_cursor) / sizeof(uint64_t)) {
        throw VsagException(ErrorType::READ_ERROR,
                            "corrupted Pyramid path count exceeds remaining payload");
    }
    if (path_count > restored_paths.max_size()) {
        throw VsagException(ErrorType::READ_ERROR, "Pyramid path count is too large");
    }
    if (!are_pyramid_path_rows_valid(restored_offsets, restored_counts, path_count)) {
        throw VsagException(ErrorType::READ_ERROR, "corrupted Pyramid path row range");
    }
    restored_paths.reserve(path_count);
    for (uint64_t offset = 0; offset < path_count; ++offset) {
        restored_paths.emplace_back(ReadPyramidPathString(reader));
    }

    std::unique_lock lock(mutex_);
    offsets_.swap(restored_offsets);
    counts_.swap(restored_counts);
    paths_.swap(restored_paths);
}

}  // namespace vsag
