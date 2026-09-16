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

#include <mutex>
#include <string>

#include "impl/logger/logger.h"
#include "index_common_param.h"
#include "io/async_io/async_io_parameter.h"
#include "io/backend/posix_file_backend.h"
#include "io/buffer_io/buffer_io_parameter.h"
#include "io/cache/optional_page_cache.h"
#include "io/core/byte_io.h"
#include "io/core/io_environment.h"
#include "io/policy/buffered_single_read.h"
#include "io/policy/direct_single_read.h"
#include "io/policy/durability_policy.h"
#include "io/policy/libaio_batch_read.h"
#include "io/policy/sequential_batch_read.h"

namespace vsag {

#if HAVE_LIBAIO
using AsyncFileBackend =
    PosixFileBackend<DirectSingleRead, PlatformLibAioBatchRead, FsyncAfterWrite>;
#else
using AsyncFileBackend = PosixFileBackend<BufferedSingleRead, PlatformLibAioBatchRead, NoFlush>;
#endif

class AsyncIO : public ByteIO<AsyncFileBackend, OptionalPageCache> {
public:
    using Base = ByteIO<AsyncFileBackend, OptionalPageCache>;

    AsyncIO(std::string filename, Allocator* allocator)
        : Base(MakeOptions(filename), MakeDefaultIOEnvironment(allocator)) {
    }

    AsyncIO(std::string filename, IOEnvironment environment)
        : Base(MakeOptions(filename), environment) {
    }

    AsyncIO(const AsyncIOParameterPtr& param, const IndexCommonParam& common_param)
        : AsyncIO(param->path_, common_param.allocator_.get()) {
    }

    AsyncIO(const IOParamPtr& param, const IndexCommonParam& common_param)
        : AsyncIO(PathFromParameter(param), common_param.allocator_.get()) {
        EnableReadCache(param);
    }

private:
    [[nodiscard]] static FileOpenOptions
    MakeOptions(const std::string& path) {
        const auto ownership = OwnershipForPath(path);
#if !HAVE_LIBAIO
        static std::once_flag fallback_warning;
        std::call_once(fallback_warning, []() {
            logger::warn(
                "libaio is unavailable, async_io is using its buffered compatibility backend");
        });
#endif
        return FileOpenOptions {
            path, true,
#if HAVE_LIBAIO
                true, true,
#else
                false, false,
#endif
                ownership
        };
    }

    [[nodiscard]] static std::string
    PathFromParameter(const IOParamPtr& param) {
        if (auto async_param = std::dynamic_pointer_cast<AsyncIOParameter>(param);
            async_param != nullptr) {
            return async_param->path_;
        }
#if !HAVE_LIBAIO
        if (auto buffer_param = std::dynamic_pointer_cast<BufferIOParameter>(param);
            buffer_param != nullptr) {
            return buffer_param->path_;
        }
#endif
        throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid AsyncIO parameter type");
    }
};

}  // namespace vsag
