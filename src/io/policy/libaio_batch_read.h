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

#include "io/core/io_environment.h"
#include "io/policy/sequential_batch_read.h"

#if HAVE_LIBAIO

#include <cstdint>

#include "io/backend/posix_file.h"
#include "io/core/read_operation.h"
#include "io/core/read_request.h"

namespace vsag {

class LibAioBatchRead {
public:
    using Operation = ImmediateOperation;
    static constexpr bool AsyncReadable = true;

    explicit LibAioBatchRead(const IOEnvironment& environment)
        : context_pool_(environment.aio_context_pool) {
        if (context_pool_ == nullptr) {
            context_pool_ = &DefaultAioContextPool();
        }
    }

    template <typename SingleReadPolicy>
    [[nodiscard]] bool
    ReadMany(const PosixFile& file,
             const SingleReadPolicy&,
             const ReadRequest* requests,
             uint64_t count) const {
        return ReadScatter(file, requests, count);
    }

    template <typename SingleReadPolicy>
    [[nodiscard]] bool
    ReadManyContiguous(const PosixFile& file,
                       const SingleReadPolicy&,
                       uint8_t* destination,
                       const uint64_t* sizes,
                       const uint64_t* offsets,
                       uint64_t count) const {
        return ReadContiguous(file, destination, sizes, offsets, count);
    }

    template <typename SingleReadPolicy>
    [[nodiscard]] Operation
    SubmitReads(const PosixFile& file,
                const SingleReadPolicy& single_read,
                const ReadRequest* requests,
                uint64_t count) const {
        return Operation(ReadMany(file, single_read, requests, count));
    }

private:
    [[nodiscard]] bool
    ReadScatter(const PosixFile& file, const ReadRequest* requests, uint64_t count) const;

    [[nodiscard]] bool
    ReadContiguous(const PosixFile& file,
                   uint8_t* destination,
                   const uint64_t* sizes,
                   const uint64_t* offsets,
                   uint64_t count) const;

    IOContextPool* context_pool_{nullptr};
};

using PlatformLibAioBatchRead = LibAioBatchRead;

}  // namespace vsag

#else

namespace vsag {

using PlatformLibAioBatchRead = SequentialBatchRead;

}  // namespace vsag

#endif
