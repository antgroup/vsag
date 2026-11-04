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
#include "io/backend/contiguous_backend.h"
#include "io/backend/heap_region.h"
#include "io/cache/no_cache.h"
#include "io/core/byte_io.h"
#include "io/memory_io/memory_io_parameter.h"

namespace vsag {

class MemoryIO : public ByteIO<ContiguousBackend<HeapRegion>, NoCache> {
public:
    using Base = ByteIO<ContiguousBackend<HeapRegion>, NoCache>;

    explicit MemoryIO(Allocator* allocator) : Base(allocator) {
    }

    MemoryIO(const MemoryIOParamPtr&, const IndexCommonParam& common_param)
        : MemoryIO(common_param.allocator_.get()) {
    }

    MemoryIO(const IOParamPtr& param, const IndexCommonParam& common_param)
        : MemoryIO(std::dynamic_pointer_cast<MemoryIOParameter>(param), common_param) {
        EnableReadCache(param);
    }
};

}  // namespace vsag
