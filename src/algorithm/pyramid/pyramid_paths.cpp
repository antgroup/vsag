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

#include <fmt/format.h>

#include <memory>

#include "pyramid.h"
#include "pyramid_path_store.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "vsag_exception.h"

namespace vsag {

void
Pyramid::serialize_paths(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, static_cast<uint64_t>(hierarchies_.size()));
    for (const auto& [hierarchy_name, hierarchy] : hierarchies_) {
        if (hierarchy->path_store == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "Pyramid path store is missing");
        }
        StreamWriter::WriteString(writer, hierarchy_name);
        hierarchy->path_store->Serialize(writer);
    }
}

void
Pyramid::deserialize_paths(StreamReader& reader, uint64_t max_count) {
    uint64_t hierarchy_count = 0;
    StreamReader::ReadObj(reader, hierarchy_count);
    if (hierarchy_count != hierarchies_.size()) {
        throw VsagException(ErrorType::READ_ERROR, "corrupted Pyramid path hierarchy count");
    }

    for (uint64_t offset = 0; offset < hierarchy_count; ++offset) {
        auto hierarchy_name = ReadPyramidPathString(reader);
        const auto hierarchy = hierarchies_.find(hierarchy_name);
        if (hierarchy == hierarchies_.end()) {
            throw VsagException(ErrorType::READ_ERROR,
                                fmt::format("unknown Pyramid path hierarchy '{}'", hierarchy_name));
        }
        if (hierarchy->second->path_store == nullptr) {
            throw VsagException(ErrorType::READ_ERROR, "Pyramid path storage is disabled");
        }
        hierarchy->second->path_store->Deserialize(reader, max_count);
    }
}

DatasetPtr
Pyramid::GetDataByIdsWithFlag(const int64_t* ids,
                              int64_t count,
                              uint64_t selected_data_flag) const {
    const bool wants_paths = (selected_data_flag & DATA_FLAG_PATH) != 0U;
    if (wants_paths) {
        CHECK_ARGUMENT(store_paths_,
                       "DATA_FLAG_PATH requires store_paths=true in the Pyramid build parameters");
    }

    Vector<InnerIdType> inner_ids(allocator_);
    auto result = this->get_data_by_ids_with_flag(ids, count, selected_data_flag, inner_ids);
    if (not wants_paths) {
        return result;
    }

    for (const auto& [hierarchy_name, hierarchy] : hierarchies_) {
        if (hierarchy->path_store == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "Pyramid path store is missing");
        }
        auto paths = std::make_unique<std::string[]>(static_cast<uint64_t>(count));
        if (not hierarchy->path_store->GetPaths(inner_ids, paths.get())) {
            continue;
        }
        if (hierarchy_name.empty()) {
            result->Paths(paths.get());
        } else {
            result->Paths(hierarchy_name, paths.get());
        }
        paths.release();
    }
    return result;
}

}  // namespace vsag
