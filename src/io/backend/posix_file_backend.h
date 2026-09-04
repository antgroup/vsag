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

#include <cerrno>
#include <cstdint>
#include <utility>

#include "fmt/format.h"
#include "io/backend/posix_file.h"
#include "io/common/io_syscall.h"
#include "io/core/io_environment.h"
#include "io/core/read_request.h"
#include "vsag/allocator.h"
#include "vsag_exception.h"

namespace vsag {

template <typename BatchReadPolicy, bool PreserveLegacyUncheckedRead>
struct PosixFileBackendCapabilities {
    static constexpr bool InMemory = false;
    static constexpr bool RequiresInitialization = false;
    static constexpr bool CanBindSerializedRange = false;
    static constexpr bool LegacyBatchRangeThrows = false;
    static constexpr bool LegacyUncheckedReadable = PreserveLegacyUncheckedRead;
    static constexpr bool BorrowedReadable = false;
    static constexpr bool BatchReadable = true;
    static constexpr bool AsyncReadable = BatchReadPolicy::AsyncReadable;
    static constexpr bool Writable = true;
    static constexpr bool Resizable = true;
};

template <typename SingleReadPolicy,
          typename BatchReadPolicy,
          typename DurabilityPolicy,
          bool PreserveLegacyUncheckedRead = false>
class PosixFileBackend {
public:
    using Capabilities = PosixFileBackendCapabilities<BatchReadPolicy, PreserveLegacyUncheckedRead>;
    using Lease = typename SingleReadPolicy::Lease;
    using Operation = typename BatchReadPolicy::Operation;

    PosixFileBackend(FileOpenOptions options, Allocator* allocator)
        : PosixFileBackend(std::move(options), MakeDefaultIOEnvironment(allocator)) {
    }

    PosixFileBackend(FileOpenOptions options, IOEnvironment environment)
        : file_(std::move(options)),
          allocator_(environment.allocator),
          single_read_(environment),
          batch_read_(environment) {
    }

    [[nodiscard]] uint64_t
    InitialLogicalSize() const {
        return file_.Size();
    }

    [[nodiscard]] bool
    ReadAt(uint64_t offset, uint64_t size, uint8_t* destination) const {
        return single_read_.ReadAt(file_, offset, size, destination);
    }

    [[nodiscard]] bool
    ReadMany(const ReadRequest* requests, uint64_t count) const {
        return batch_read_.ReadMany(file_, single_read_, requests, count);
    }

    [[nodiscard]] bool
    ReadManyContiguous(uint8_t* destination,
                       const uint64_t* sizes,
                       const uint64_t* offsets,
                       uint64_t count) const {
        return batch_read_.ReadManyContiguous(
            file_, single_read_, destination, sizes, offsets, count);
    }

    [[nodiscard]] Operation
    SubmitReads(const ReadRequest* requests, uint64_t count) const {
        return batch_read_.SubmitReads(file_, single_read_, requests, count);
    }

    [[nodiscard]] Lease
    Acquire(uint64_t offset, uint64_t size) const {
        return single_read_.Acquire(file_, allocator_, offset, size);
    }

    [[nodiscard]] const uint8_t*
    LegacyRead(uint64_t offset, uint64_t size, bool& need_release) const {
        return single_read_.LegacyRead(file_, allocator_, offset, size, need_release);
    }

    void
    Release(const uint8_t* data) const {
        single_read_.Release(allocator_, data);
    }

    void
    WriteAt(uint64_t offset, const uint8_t* source, uint64_t size) {
        uint64_t written = 0;
        while (written < size) {
            const ssize_t result = IOSyscall::PWrite(
                file_.WriteFd(), source + written, size - written, offset + written);
            if (result < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw VsagException(
                    ErrorType::INTERNAL_ERROR,
                    fmt::format("pwrite failed: fd={}, offset={}, size={}, written={}, errno={}",
                                file_.WriteFd(),
                                offset,
                                size,
                                written,
                                errno));
            }
            if (result == 0) {
                throw VsagException(
                    ErrorType::INTERNAL_ERROR,
                    fmt::format("pwrite made no progress: fd={}, offset={}, size={}, written={}",
                                file_.WriteFd(),
                                offset,
                                size,
                                written));
            }
            written += static_cast<uint64_t>(result);
        }
        durability_.AfterWrite(file_.WriteFd());
        file_.RecordWriteEnd(offset + size);
    }

    void
    ResizePhysical(uint64_t size) {
        file_.Truncate(size);
    }

    void
    ShrinkPhysical(uint64_t size) {
        file_.Truncate(size);
    }

    void
    Prefetch(uint64_t, uint64_t) {
    }

    [[nodiscard]] int64_t
    MemoryUsage(uint64_t logical_size) const {
        return static_cast<int64_t>(logical_size);
    }

    [[nodiscard]] const uint8_t*
    Data() const {
        return nullptr;
    }

    [[nodiscard]] Allocator*
    AllocatorPtr() const {
        return allocator_;
    }

private:
    PosixFile file_;
    Allocator* allocator_{nullptr};
    SingleReadPolicy single_read_;
    BatchReadPolicy batch_read_;
    DurabilityPolicy durability_;
};

}  // namespace vsag
