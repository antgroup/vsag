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

#include "io/user_defined_io/user_defined_io.h"

#include "index_common_param.h"
#include "vsag_exception.h"

namespace vsag {
namespace {

UserDefinedIOParamPtr
require_user_defined_io_param(const IOParamPtr& param) {
    auto storage_param = std::dynamic_pointer_cast<UserDefinedIOParameter>(param);
    if (storage_param == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "user_defined_io requires UserDefinedIOParameter");
    }
    return storage_param;
}

UserDefinedIOParamPtr
bind_user_defined_io(const UserDefinedIOParamPtr& param, const IndexCommonParam& common_param) {
    if (param == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "user_defined_io parameter is null");
    }
    if (common_param.user_defined_ios_ == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "user_defined_io requires UserDefinedIOSet");
    }
    const auto* storage = common_param.user_defined_ios_->Get(param->user_defined_io);
    if (storage == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "user defined IO not found: " + param->user_defined_io);
    }
    if (storage->reader == nullptr or storage->writer == nullptr) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            "user defined IO requires non-null Reader and Writer: " + param->user_defined_io);
    }
    auto bound_param = std::make_shared<UserDefinedIOParameter>(*param);
    bound_param->reader = storage->reader;
    bound_param->writer = storage->writer;
    return bound_param;
}

}  // namespace

UserDefinedIO::UserDefinedIO(Allocator* allocator) : Base(allocator) {
}

UserDefinedIO::UserDefinedIO(const UserDefinedIOParamPtr& param,
                             const IndexCommonParam& common_param)
    : UserDefinedIO(common_param.allocator_.get()) {
    InitIO(bind_user_defined_io(param, common_param));
}

UserDefinedIO::UserDefinedIO(const IOParamPtr& param, const IndexCommonParam& common_param)
    : UserDefinedIO(require_user_defined_io_param(param), common_param) {
    EnableReadCache(param);
}

}  // namespace vsag
