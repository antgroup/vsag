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

#if HAVE_LIBURING

#include "io/uring_io/uring_io_parameter.h"

#include <fmt/format.h>

#include "common.h"
#include "inner_string_params.h"

namespace vsag {

UringIOParameter::UringIOParameter() : IOParameter(IO_TYPE_VALUE_URING_IO) {
}

UringIOParameter::UringIOParameter(const vsag::JsonType& json)
    : IOParameter(IO_TYPE_VALUE_URING_IO) {
    this->FromJson(json);  // NOLINT(clang-analyzer-optin.cplusplus.VirtualCall)
}

void
UringIOParameter::FromJson(const vsag::JsonType& json) {
    this->direct_read_ = false;
    CHECK_ARGUMENT(json.Contains(IO_FILE_PATH_KEY), "missing file_path parameter for uring_io");
    CHECK_ARGUMENT(json[IO_FILE_PATH_KEY].IsString(), "file_path must be a string");
    this->path_ = json[IO_FILE_PATH_KEY].GetString();
    if (json.Contains(IO_DIRECT_READ_KEY)) {
        CHECK_ARGUMENT(json[IO_DIRECT_READ_KEY].IsBool(), "direct_read must be a boolean");
        this->direct_read_ = json[IO_DIRECT_READ_KEY].GetBool();
    }
}

JsonType
UringIOParameter::ToJson() const {
    JsonType json;
    json[TYPE_KEY].SetString(IO_TYPE_VALUE_URING_IO);
    json[IO_FILE_PATH_KEY].SetString(this->path_);
    json[IO_DIRECT_READ_KEY].SetBool(this->direct_read_);
    return json;
}
}  // namespace vsag
#endif  // HAVE_LIBURING
