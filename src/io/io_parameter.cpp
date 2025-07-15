
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

#include "io_parameter.h"

#include "async_io_parameter.h"
#include "buffer_io_parameter.h"
#include "inner_string_params.h"
#include "memory_block_io_parameter.h"
#include "memory_io_parameter.h"
#include "mmap_io_parameter.h"
#include "reader_io_parameter.h"

namespace vsag {

IOParamPtr
IOParameter::GetIOParameterByJson(const JsonType& json) {
    IOParamPtr io_ptr = nullptr;
    try {
        auto type_name = Parameter::TryToParseType(json);
        if (type_name == IO_TYPE_VALUE_MEMORY_IO) {
            io_ptr = std::make_shared<MemoryIOParameter>();
            io_ptr->FromJson(json);
        } else if (type_name == IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
            io_ptr = std::make_shared<MemoryBlockIOParameter>();
            io_ptr->FromJson(json);
        } else if (type_name == IO_TYPE_VALUE_BUFFER_IO) {
            io_ptr = std::make_shared<BufferIOParameter>();
            io_ptr->FromJson(json);
        } else if (type_name == IO_TYPE_VALUE_ASYNC_IO) {
            io_ptr = std::make_shared<AsyncIOParameter>();
            io_ptr->FromJson(json);
        } else if (type_name == IO_TYPE_VALUE_MMAP_IO) {
            io_ptr = std::make_shared<MMapIOParameter>();
            io_ptr->FromJson(json);
        } else if (type_name == IO_TYPE_VALUE_READER_IO) {
            io_ptr = std::make_shared<ReaderIOParameter>();
            io_ptr->FromJson(json);
        }
    } catch (std::invalid_argument& error) {
        return nullptr;
    }
    return io_ptr;
}
IOParameter::IOParameter(std::string name) : name_(std::move(name)) {
}
}  // namespace vsag
