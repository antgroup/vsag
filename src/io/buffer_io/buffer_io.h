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
#include <utility>

#include "index_common_param.h"
#include "io/backend/posix_file_backend.h"
#include "io/buffer_io/buffer_io_parameter.h"
#include "io/cache/optional_page_cache.h"
#include "io/core/byte_io.h"
#include "io/policy/buffered_single_read.h"
#include "io/policy/durability_policy.h"
#include "io/policy/sequential_batch_read.h"

namespace vsag {

// The legacy BufferIO copy and batch APIs delegated logical bounds handling to pread. Preserve that
// hot-path behavior while ReadAt/ReadMany retain the stricter canonical contract.
using BufferFileBackend = PosixFileBackend<BufferedSingleRead, SequentialBatchRead, NoFlush, true>;

class BufferIO : public ByteIO<BufferFileBackend, OptionalPageCache> {
public:
    using Base = ByteIO<BufferFileBackend, OptionalPageCache>;

    BufferIO(std::string filename, Allocator* allocator) : Base(MakeOptions(filename), allocator) {
    }

    BufferIO(const BufferIOParameterPtr& param, const IndexCommonParam& common_param)
        : BufferIO(param->path_, common_param.allocator_.get()) {
    }

    BufferIO(const IOParamPtr& param, const IndexCommonParam& common_param)
        : BufferIO(std::dynamic_pointer_cast<BufferIOParameter>(param), common_param) {
        EnableReadCache(param);
    }

private:
    [[nodiscard]] static FileOpenOptions
    MakeOptions(const std::string& path) {
        const auto ownership = OwnershipForPath(path);
        return FileOpenOptions{path, true, false, false, ownership};
    }
};

}  // namespace vsag
