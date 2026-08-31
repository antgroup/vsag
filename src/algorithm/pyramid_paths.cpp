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

#include <memory>

#include "pyramid.h"
#include "vsag_exception.h"

namespace vsag {

void
Pyramid::serialize_paths(StreamWriter& writer) const {
    if (path_store_ == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Pyramid path store is missing");
    }
    path_store_->Serialize(writer);
}

void
Pyramid::deserialize_paths(StreamReader& reader, uint64_t expected_count) {
    if (path_store_ == nullptr) {
        throw VsagException(ErrorType::READ_ERROR, "Pyramid path storage is disabled");
    }
    path_store_->Deserialize(reader, expected_count);
}

DatasetPtr
Pyramid::GetDataByIdsWithFlag(const int64_t* ids,
                              int64_t count,
                              uint64_t selected_data_flag) const {
    const bool wants_paths = (selected_data_flag & DATA_FLAG_PATH) != 0U;
    if (wants_paths) {
        CHECK_ARGUMENT(store_paths_,
                       "DATA_FLAG_PATH requires store_paths=true in the Pyramid build parameters");
        if (path_store_ == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "Pyramid path store is missing");
        }
    }

    Vector<InnerIdType> inner_ids(allocator_);
    auto result = this->get_data_by_ids_with_flag(ids, count, selected_data_flag, inner_ids);
    if (not wants_paths) {
        return result;
    }

    auto paths = std::make_unique<std::string[]>(static_cast<uint64_t>(count));
    if (not path_store_->GetPaths(inner_ids, paths.get())) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "Pyramid path is unavailable for a requested id");
    }
    result->Paths(paths.get());
    paths.release();
    return result;
}

}  // namespace vsag
