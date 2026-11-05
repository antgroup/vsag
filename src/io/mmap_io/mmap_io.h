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

#include <string>

#include "index_common_param.h"
#include "io/backend/contiguous_backend.h"
#include "io/backend/mmap_region.h"
#include "io/cache/no_cache.h"
#include "io/core/byte_io.h"
#include "io/mmap_io/mmap_io_parameter.h"

namespace vsag {

class MMapIO : public ByteIO<ContiguousBackend<MMapRegion>, NoCache> {
public:
    using Base = ByteIO<ContiguousBackend<MMapRegion>, NoCache>;

    MMapIO(std::string filename, Allocator* allocator) : Base(std::move(filename), allocator) {
    }

    MMapIO(const MMapIOParamPtr& param, const IndexCommonParam& common_param)
        : MMapIO(param->path_, common_param.allocator_.get()) {
    }

    MMapIO(const IOParamPtr& param, const IndexCommonParam& common_param)
        : MMapIO(std::dynamic_pointer_cast<MMapIOParameter>(param), common_param) {
        EnableReadCache(param);
    }
};

}  // namespace vsag
