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

#include <mutex>

#include "common.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "vsag_exception.h"

namespace vsag {

void
PyramidPathStore::Writer::EnsureSlots(uint64_t slot_count) {
    if (slot_count <= store_.paths_by_inner_id_.size()) {
        return;
    }

    const auto old_path_count = store_.paths_by_inner_id_.size();
    const auto old_presence_count = store_.has_path_.size();
    try {
        store_.paths_by_inner_id_.resize(slot_count);
        store_.has_path_.resize(slot_count, 0);
    } catch (...) {
        store_.paths_by_inner_id_.resize(old_path_count);
        store_.has_path_.resize(old_presence_count);
        throw;
    }
}

void
PyramidPathStore::Writer::Insert(InnerIdType inner_id, const std::string& path) {
    const auto slot = static_cast<uint64_t>(inner_id);
    EnsureSlots(slot + 1);
    CHECK_ARGUMENT(store_.has_path_[slot] == 0, "inner id already has a Pyramid path");
    store_.paths_by_inner_id_[slot] = path;
    store_.has_path_[slot] = 1;
}

PyramidPathStore::Writer
PyramidPathStore::AcquireWriter() {
    return Writer(*this);
}

bool
PyramidPathStore::GetPaths(const Vector<InnerIdType>& inner_ids, std::string* paths) const {
    std::shared_lock lock(mutex_);
    for (uint64_t offset = 0; offset < inner_ids.size(); ++offset) {
        const auto inner_id = static_cast<uint64_t>(inner_ids[offset]);
        if (inner_id >= has_path_.size() || has_path_[inner_id] == 0) {
            return false;
        }
        paths[offset] = paths_by_inner_id_[inner_id];
    }
    return true;
}

void
PyramidPathStore::Serialize(StreamWriter& writer) const {
    std::shared_lock lock(mutex_);
    if (paths_by_inner_id_.size() != has_path_.size()) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Pyramid path store size mismatch");
    }
    StreamWriter::WriteObj(writer, static_cast<uint64_t>(paths_by_inner_id_.size()));
    for (uint64_t inner_id = 0; inner_id < has_path_.size(); ++inner_id) {
        const auto present = has_path_[inner_id];
        StreamWriter::WriteObj(writer, present);
        if (present != 0) {
            StreamWriter::WriteString(writer, paths_by_inner_id_[inner_id]);
        }
    }
}

void
PyramidPathStore::Deserialize(StreamReader& reader, uint64_t expected_count) {
    uint64_t slot_count = 0;
    StreamReader::ReadObj(reader, slot_count);
    if (slot_count != expected_count) {
        throw VsagException(ErrorType::READ_ERROR, "corrupted Pyramid path slot count");
    }
    const auto cursor = reader.GetCursor();
    const auto reader_length = reader.Length();
    if (cursor > reader_length || slot_count > reader_length - cursor) {
        throw VsagException(ErrorType::READ_ERROR,
                            "corrupted Pyramid path slots exceed remaining payload");
    }

    Vector<std::string> restored_paths(slot_count, std::string{}, allocator_);
    Vector<uint8_t> restored_has_path(slot_count, 0, allocator_);
    for (uint64_t inner_id = 0; inner_id < slot_count; ++inner_id) {
        uint8_t present = 0;
        StreamReader::ReadObj(reader, present);
        if (present > 1) {
            throw VsagException(ErrorType::READ_ERROR, "corrupted Pyramid path presence value");
        }
        restored_has_path[inner_id] = present;
        if (present != 0) {
            restored_paths[inner_id] = StreamReader::ReadString(reader);
        }
    }

    std::unique_lock lock(mutex_);
    paths_by_inner_id_.swap(restored_paths);
    has_path_.swap(restored_has_path);
}

}  // namespace vsag
