
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

#include "memory_block_io_parameter.h"

#include "inner_string_params.h"
#include "vsag/options.h"

namespace vsag {

MemoryBlockIOParameter::MemoryBlockIOParameter(const JsonType& json) : MemoryBlockIOParameter() {
    this->FromJson(json);
}
MemoryBlockIOParameter::MemoryBlockIOParameter() : IOParameter(IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
    auto block_size = Options::Instance().block_size_limit();
    this->block_size_ = NearestPowerOfTwo(block_size);
}

void
MemoryBlockIOParameter::FromJson(const JsonType& json) {
    auto block_size = Options::Instance().block_size_limit();
    this->block_size_ = NearestPowerOfTwo(block_size);
}

JsonType
MemoryBlockIOParameter::ToJson() const {
    JsonType json;
    json[IO_TYPE_KEY] = IO_TYPE_VALUE_BLOCK_MEMORY_IO;
    return json;
}

uint64_t
MemoryBlockIOParameter::NearestPowerOfTwo(uint64_t value) {
    if (value == 0) {
        return 0;
    }
    if ((value & (value - 1)) == 0) {
        return value;
    }
    uint64_t bit = 0;
    while (value > 0) {
        value >>= 1;
        bit++;
    }
    return 1ULL << (bit - 1);
}

}  // namespace vsag
