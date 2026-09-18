// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "io/external_storage_io/external_storage_io.h"

#include "index_common_param.h"
#include "vsag_exception.h"

namespace vsag {
namespace {

ExternalStorageIOParamPtr
require_external_storage_param(const IOParamPtr& param) {
    auto storage_param = std::dynamic_pointer_cast<ExternalStorageIOParameter>(param);
    if (storage_param == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "external_storage_io requires ExternalStorageIOParameter");
    }
    return storage_param;
}

ExternalStorageIOParamPtr
bind_external_storage(const ExternalStorageIOParamPtr& param,
                      const IndexCommonParam& common_param) {
    if (param == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "external_storage_io parameter is null");
    }
    if (common_param.external_storages_ == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "external_storage_io requires ExternalStorageSet");
    }
    const auto* storage = common_param.external_storages_->Get(param->external_storage);
    if (storage == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "external storage not found: " + param->external_storage);
    }
    if (storage->reader == nullptr or storage->writer == nullptr) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            "external storage requires non-null Reader and Writer: " + param->external_storage);
    }
    auto bound_param = std::make_shared<ExternalStorageIOParameter>(*param);
    bound_param->reader = storage->reader;
    bound_param->writer = storage->writer;
    return bound_param;
}

}  // namespace

ExternalStorageIO::ExternalStorageIO(Allocator* allocator) : Base(allocator) {
}

ExternalStorageIO::ExternalStorageIO(const ExternalStorageIOParamPtr& param,
                                     const IndexCommonParam& common_param)
    : ExternalStorageIO(common_param.allocator_.get()) {
    InitIO(bind_external_storage(param, common_param));
}

ExternalStorageIO::ExternalStorageIO(const IOParamPtr& param, const IndexCommonParam& common_param)
    : ExternalStorageIO(require_external_storage_param(param), common_param) {
    EnableReadCache(param);
}

}  // namespace vsag
