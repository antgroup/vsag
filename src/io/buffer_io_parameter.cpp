
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

#include "buffer_io_parameter.h"

#include "inner_string_params.h"

namespace vsag {

BufferIOParameter::BufferIOParameter() : IOParameter(IO_TYPE_VALUE_BUFFER_IO) {
}

BufferIOParameter::BufferIOParameter(const vsag::JsonType& json)
    : IOParameter(IO_TYPE_VALUE_BUFFER_IO) {
    this->FromJson(json);  // NOLINT(clang-analyzer-optin.cplusplus.VirtualCall)
}

void
BufferIOParameter::FromJson(const JsonType& json) {
    CHECK_ARGUMENT(json.contains(IO_FILE_PATH), "miss file_path param in buffer io type");
    this->path_ = json[IO_FILE_PATH];
}

JsonType
BufferIOParameter::ToJson() {
    JsonType json;
    json[IO_TYPE_KEY] = IO_TYPE_VALUE_BUFFER_IO;
    json[IO_FILE_PATH] = this->path_;
    return json;
}
}  // namespace vsag
