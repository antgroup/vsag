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

#pragma once

#include "index_common_param.h"
#include "io/backend/block_memory_backend.h"
#include "io/cache/no_cache.h"
#include "io/core/byte_io.h"
#include "io/memory_block_io/memory_block_io_parameter.h"

namespace vsag {

class MemoryBlockIO : public ByteIO<BlockMemoryBackend, NoCache> {
public:
    using Base = ByteIO<BlockMemoryBackend, NoCache>;

    MemoryBlockIO(uint64_t block_size, Allocator* allocator) : Base(block_size, allocator) {
    }

    MemoryBlockIO(const MemoryBlockIOParamPtr& param, const IndexCommonParam& common_param)
        : MemoryBlockIO(param->block_size_, common_param.allocator_.get()) {
    }

    MemoryBlockIO(const IOParamPtr& param, const IndexCommonParam& common_param)
        : MemoryBlockIO(std::dynamic_pointer_cast<MemoryBlockIOParameter>(param), common_param) {
        EnableReadCache(param);
    }
};

}  // namespace vsag
