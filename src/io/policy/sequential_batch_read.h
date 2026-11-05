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

#include <cstdint>

#include "io/backend/posix_file.h"
#include "io/core/io_environment.h"
#include "io/core/read_operation.h"
#include "io/core/read_request.h"

namespace vsag {

class SequentialBatchRead {
public:
    using Operation = ImmediateOperation;
    static constexpr bool AsyncReadable = false;

    SequentialBatchRead() = default;

    explicit SequentialBatchRead(const IOEnvironment&) {
    }

    template <typename SingleReadPolicy>
    [[nodiscard]] bool
    ReadMany(const PosixFile& file,
             const SingleReadPolicy& single_read,
             const ReadRequest* requests,
             uint64_t count) const {
        for (uint64_t i = 0; i < count; ++i) {
            if (not single_read.ReadAt(
                    file, requests[i].offset, requests[i].size, requests[i].destination)) {
                return false;
            }
        }
        return true;
    }

    template <typename SingleReadPolicy>
    [[nodiscard]] Operation
    SubmitReads(const PosixFile& file,
                const SingleReadPolicy& single_read,
                const ReadRequest* requests,
                uint64_t count) const {
        return Operation(ReadMany(file, single_read, requests, count));
    }

    template <typename SingleReadPolicy>
    [[nodiscard]] bool
    ReadManyContiguous(const PosixFile& file,
                       const SingleReadPolicy& single_read,
                       uint8_t* destination,
                       const uint64_t* sizes,
                       const uint64_t* offsets,
                       uint64_t count) const {
        // This compatibility form packs results consecutively in destination. Callers needing
        // independently placed outputs should use the ReadRequest-based ReadMany API.
        for (uint64_t i = 0; i < count; ++i) {
            if (not single_read.ReadAt(file, offsets[i], sizes[i], destination)) {
                return false;
            }
            destination += sizes[i];
        }
        return true;
    }
};

}  // namespace vsag
