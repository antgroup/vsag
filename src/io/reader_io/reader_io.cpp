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

#include "io/reader_io/reader_io.h"

#include "index_common_param.h"

namespace vsag {
namespace {

ReaderIOParamPtr
require_reader_param(const IOParamPtr& param) {
    auto reader_param = std::dynamic_pointer_cast<ReaderIOParameter>(param);
    if (reader_param == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "ReaderIO requires a ReaderIOParameter");
    }
    return reader_param;
}

}  // namespace

ReaderIO::ReaderIO(Allocator* allocator) : Base(allocator) {
}

ReaderIO::ReaderIO(const ReaderIOParamPtr& reader_param, const IndexCommonParam& common_param)
    : ReaderIO(common_param.allocator_.get()) {
    if (reader_param != nullptr and reader_param->reader != nullptr) {
        InitIO(reader_param);
    }
}

ReaderIO::ReaderIO(const IOParamPtr& param, const IndexCommonParam& common_param)
    : ReaderIO(require_reader_param(param), common_param) {
    EnableReadCache(param);
}

}  // namespace vsag
