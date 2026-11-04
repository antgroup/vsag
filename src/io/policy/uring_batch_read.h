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

#if HAVE_LIBURING

#include <cstdint>

#include "io/backend/posix_file.h"
#include "io/core/read_request.h"
#include "io/core/uring_read_operation.h"

namespace vsag {

class UringBatchRead {
public:
    using Operation = UringReadOperation;
    static constexpr bool AsyncReadable = true;

    explicit UringBatchRead(const IOEnvironment& environment)
        : context_pool_(environment.uring_context_pool), direct_read_(environment.direct_read) {
        if (context_pool_ == nullptr) {
            context_pool_ = &DefaultUringContextPool();
        }
    }

    template <typename SingleReadPolicy>
    [[nodiscard]] bool
    ReadMany(const PosixFile& file,
             const SingleReadPolicy& single_read,
             const ReadRequest* requests,
             uint64_t count) const {
        ReadStatus status = ReadScatter(file, requests, count);
        if (status == ReadStatus::Unavailable) {
            return sequential_.ReadMany(file, single_read, requests, count);
        }
        return status == ReadStatus::Completed;
    }

    template <typename SingleReadPolicy>
    [[nodiscard]] bool
    ReadManyContiguous(const PosixFile& file,
                       const SingleReadPolicy& single_read,
                       uint8_t* destination,
                       const uint64_t* sizes,
                       const uint64_t* offsets,
                       uint64_t count) const {
        ReadStatus status = ReadContiguous(file, destination, sizes, offsets, count);
        if (status == ReadStatus::Unavailable) {
            return sequential_.ReadManyContiguous(
                file, single_read, destination, sizes, offsets, count);
        }
        return status == ReadStatus::Completed;
    }

    template <typename SingleReadPolicy>
    [[nodiscard]] Operation
    SubmitReads(const PosixFile& file,
                const SingleReadPolicy& single_read,
                const ReadRequest* requests,
                uint64_t count) const {
        return Operation(ReadMany(file, single_read, requests, count));
    }

public:
    enum class ReadStatus : uint8_t {
        Completed,
        Unavailable,
        Invalid,
    };

private:
    [[nodiscard]] ReadStatus
    ReadScatter(const PosixFile& file, const ReadRequest* requests, uint64_t count) const;

    [[nodiscard]] ReadStatus
    ReadContiguous(const PosixFile& file,
                   uint8_t* destination,
                   const uint64_t* sizes,
                   const uint64_t* offsets,
                   uint64_t count) const;

    UringIOContextPool* context_pool_{nullptr};
    bool direct_read_{false};
    SequentialBatchRead sequential_;
};

using PlatformUringBatchRead = UringBatchRead;

}  // namespace vsag

#else

namespace vsag {

using PlatformUringBatchRead = SequentialBatchRead;

}  // namespace vsag

#endif
