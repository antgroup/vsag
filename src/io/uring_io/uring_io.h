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
#include "io/cache/optional_page_cache.h"
#include "io/core/byte_io.h"
#include "io/core/io_environment.h"
#include "io/policy/configurable_single_read.h"
#include "io/policy/durability_policy.h"
#include "io/policy/uring_batch_read.h"
#include "io/uring_io/uring_io_parameter.h"

namespace vsag {

#if HAVE_LIBURING
using UringFileBackend = PosixFileBackend<ConfigurableSingleRead, PlatformUringBatchRead, NoFlush>;
#else
// Without liburing, UringIO historically aliases BufferIO. Preserve BufferIO's compatibility
// Read/MultiRead behavior while canonical ReadAt/ReadMany keep strict logical range validation.
using UringFileBackend =
    PosixFileBackend<ConfigurableSingleRead, PlatformUringBatchRead, NoFlush, true>;
#endif

class UringIO : public ByteIO<UringFileBackend, OptionalPageCache> {
public:
    using Base = ByteIO<UringFileBackend, OptionalPageCache>;

    UringIO(std::string filename, Allocator* allocator, bool direct_read = false)
        : UringIO(std::move(filename), MakeEnvironment(allocator, direct_read)) {
    }

    UringIO(std::string filename, IOEnvironment environment)
        : Base(MakeOptions(filename, EffectiveDirectRead(environment.direct_read)),
               NormalizeEnvironment(environment)) {
    }

    UringIO(const UringIOParameterPtr& param, const IndexCommonParam& common_param)
        : UringIO(param->path_, common_param.allocator_.get(), ParamDirectRead(param)) {
    }

    UringIO(const IOParamPtr& param, const IndexCommonParam& common_param)
        : UringIO(std::dynamic_pointer_cast<UringIOParameter>(param), common_param) {
        EnableReadCache(param);
    }

private:
    [[nodiscard]] static bool
    ParamDirectRead(const UringIOParameterPtr& param) {
#if HAVE_LIBURING
        return param->direct_read_;
#else
        (void)param;
        return false;
#endif
    }

    [[nodiscard]] static bool
    EffectiveDirectRead(bool direct_read) {
#if HAVE_LIBURING
        return direct_read;
#else
        (void)direct_read;
        return false;
#endif
    }

    [[nodiscard]] static IOEnvironment
    MakeEnvironment(Allocator* allocator, bool direct_read) {
        IOEnvironment environment = MakeDefaultIOEnvironment(allocator);
        environment.direct_read = EffectiveDirectRead(direct_read);
        return environment;
    }

    [[nodiscard]] static IOEnvironment
    NormalizeEnvironment(IOEnvironment environment) {
        environment.direct_read = EffectiveDirectRead(environment.direct_read);
        return environment;
    }

    [[nodiscard]] static FileOpenOptions
    MakeOptions(const std::string& path, bool direct_read) {
        const auto ownership = OwnershipForPath(path);
        return FileOpenOptions{path, true, direct_read, true, ownership};
    }
};

}  // namespace vsag
