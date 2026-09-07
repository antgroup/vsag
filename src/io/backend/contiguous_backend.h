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
#include <cstring>
#include <utility>

#include "io/core/read_lease.h"
#include "io/core/read_operation.h"
#include "io/core/read_request.h"
#include "utils/prefetch.h"

namespace vsag {

template <typename Region>
struct ContiguousBackendCapabilities {
    static constexpr bool InMemory = Region::InMemory;
    static constexpr bool RequiresInitialization = false;
    static constexpr bool CanBindSerializedRange = false;
    static constexpr bool LegacyBatchRangeThrows = false;
    static constexpr bool LegacyUncheckedReadable = false;
    static constexpr bool BorrowedReadable = true;
    static constexpr bool BatchReadable = true;
    static constexpr bool AsyncReadable = false;
    static constexpr bool Writable = true;
    static constexpr bool Resizable = true;
};

template <typename Region>
class ContiguousBackend {
public:
    using Capabilities = ContiguousBackendCapabilities<Region>;
    using Lease = BorrowedLease;
    using Operation = ImmediateOperation;

    template <typename... Args>
    explicit ContiguousBackend(Args&&... args) : region_(std::forward<Args>(args)...) {
    }

    [[nodiscard]] uint64_t
    InitialLogicalSize() const {
        return region_.InitialLogicalSize();
    }

    [[nodiscard]] bool
    ReadAt(uint64_t offset, uint64_t size, uint8_t* destination) const {
        if (size == 0) {
            return true;
        }
        std::memcpy(destination, region_.Data() + offset, size);
        return true;
    }

    [[nodiscard]] bool
    ReadMany(const ReadRequest* requests, uint64_t count) const {
        const uint8_t* data = region_.Data();
        for (uint64_t i = 0; i < count; ++i) {
            if (requests[i].size > 0) {
                std::memcpy(requests[i].destination, data + requests[i].offset, requests[i].size);
            }
        }
        return true;
    }

    [[nodiscard]] bool
    ReadManyContiguous(uint8_t* destination,
                       const uint64_t* sizes,
                       const uint64_t* offsets,
                       uint64_t count) const {
        for (uint64_t i = 0; i < count; ++i) {
            if (sizes[i] > 0) {
                std::memcpy(destination, region_.Data() + offsets[i], sizes[i]);
                destination += sizes[i];
            }
        }
        return true;
    }

    [[nodiscard]] Operation
    SubmitReads(const ReadRequest* requests, uint64_t count) const {
        return Operation(ReadMany(requests, count));
    }

    [[nodiscard]] Lease
    Acquire(uint64_t offset, uint64_t size) const {
        if (size == 0) {
            return {};
        }
        return Lease(region_.Data() + offset, size, BorrowedOwner{});
    }

    [[nodiscard]] const uint8_t*
    LegacyRead(uint64_t offset, uint64_t, bool& need_release) const {
        need_release = false;
        return region_.Data() + offset;
    }

    void
    Release(const uint8_t*) const {
    }

    void
    WriteAt(uint64_t offset, const uint8_t* source, uint64_t size) {
        region_.EnsureCapacity(offset + size);
        if (size > 0) {
            std::memcpy(region_.Data() + offset, source, size);
        }
    }

    void
    ResizePhysical(uint64_t size) {
        region_.ResizePhysical(size);
    }

    void
    ShrinkPhysical(uint64_t size) {
        region_.ShrinkPhysical(size);
    }

    void
    Prefetch(uint64_t offset, uint64_t cache_line) {
        PrefetchLines(region_.Data() + offset, cache_line);
    }

    [[nodiscard]] int64_t
    MemoryUsage(uint64_t logical_size) const {
        return region_.MemoryUsage(logical_size);
    }

    [[nodiscard]] const uint8_t*
    Data() const {
        return region_.Data();
    }

    [[nodiscard]] Allocator*
    AllocatorPtr() const {
        return region_.AllocatorPtr();
    }

private:
    Region region_;
};

}  // namespace vsag
