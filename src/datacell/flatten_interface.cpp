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

#include "flatten_interface.h"

#include "flatten_interface_factory.h"
#include "index_common_param.h"
#include "inner_string_params.h"

namespace vsag {

BatchNodeReadGuard::BatchNodeReadGuard(MutexArray& mutexes,
                                       const InnerIdType* ids,
                                       InnerIdType count,
                                       uint64_t total_count)
    : mutexes_(mutexes) {
    if (count == 0) {
        return;
    }
    ids_.assign(ids, ids + count);
    // Order acquisitions and avoid recursively locking a shared_mutex for duplicate IDs.
    std::sort(ids_.begin(), ids_.end());
    ids_.erase(std::unique(ids_.begin(), ids_.end()), ids_.end());
    if (ids_.back() >= total_count) {
        throw VsagException(ErrorType::READ_ERROR, "invalid internal id for node lock");
    }

    try {
        for (const auto id : ids_) {
            mutexes_.SharedLock(id);
            ++acquired_;
        }
    } catch (...) {
        unlock_acquired();
        throw;
    }
}

BatchNodeReadGuard::~BatchNodeReadGuard() {
    unlock_acquired();
}

void
BatchNodeReadGuard::unlock_acquired() noexcept {
    while (acquired_ > 0) {
        mutexes_.SharedUnlock(ids_[--acquired_]);
    }
}

IndexCommonParam
FlattenInterface::ExportCommonParam() {
    throw VsagException(ErrorType::INTERNAL_ERROR, "ExportCommonParam is not implemented");
}

FlattenInterfacePtr
FlattenInterface::MakeInstance(const FlattenInterfaceParamPtr& param,
                               const IndexCommonParam& common_param) {
    if (param->name == MULTI_VECTOR_DATA_CELL) {
        return MakeMultiVectorDataCell(param, common_param);
    }
    if (param->name == SPARSE_VECTOR_DATA_CELL) {
        return MakeSparseVectorDataCell(param, common_param);
    }
    return MakeFlattenDataCell(param, common_param);
}

}  // namespace vsag
