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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "io/core/io_utils.h"
#include "io/core/read_lease.h"
#include "io/core/read_operation.h"
#include "io/core/read_request.h"
#include "io/user_defined_io/user_defined_io_parameter.h"
#include "vsag/allocator.h"
#include "vsag/readerset.h"
#include "vsag_exception.h"

namespace vsag {

// NOLINTBEGIN(readability-identifier-naming) -- platform naming convention (see .clang-tidy)
struct UserDefinedBackendCapabilities {
    static constexpr bool InMemory = false;
    static constexpr bool RequiresInitialization = true;
    static constexpr bool CanBindSerializedRange = false;
    static constexpr bool LegacyBatchRangeThrows = false;
    static constexpr bool LegacyUncheckedReadable = false;
    static constexpr bool BorrowedReadable = false;
    static constexpr bool BatchReadable = true;
    // SubmitReads completes inline, matching the synchronous memory backend contract.
    static constexpr bool AsyncReadable = false;
    static constexpr bool Writable = true;
    static constexpr bool Resizable = true;
    static constexpr bool CanResizeForOverwrite = false;
};
// NOLINTEND(readability-identifier-naming)

class UserDefinedBackend {
public:
    using Capabilities = UserDefinedBackendCapabilities;
    using Lease = AllocatorLease;
    using Operation = ImmediateOperation;

    explicit UserDefinedBackend(Allocator* allocator) : allocator_(allocator) {
    }

    [[nodiscard]] static uint64_t
    InitialLogicalSize() {
        return 0;
    }

    [[nodiscard]] uint64_t
    Initialize(const IOParamPtr& io_param,
               bool /*unused*/,
               uint64_t /*unused*/,
               uint64_t /*unused*/) {
        auto param = std::dynamic_pointer_cast<UserDefinedIOParameter>(io_param);
        if (param == nullptr or param->GetReader() == nullptr or param->GetWriter() == nullptr) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "user_defined_io requires a non-null Reader and Writer pair");
        }
        reader_ = param->GetReader();
        writer_ = param->GetWriter();
        prefetcher_ = dynamic_cast<ReaderPrefetcher*>(reader_.get());
        return reader_->Size();
    }

    // The bool result is required by ByteIO's backend concept; Reader::Read failures propagate.
    // A null destination with a non-zero size is a caller bug and is rejected rather than
    // dereferenced, matching the guard in ReadMany and ReadManyContiguous.
    [[nodiscard]] bool
    ReadAt(uint64_t offset, uint64_t size, uint8_t* destination) const {
        ensure_initialized();
        if (size == 0) {
            return true;
        }
        if (destination == nullptr) {
            return false;
        }
        reader_->Read(offset, size, destination);
        return true;
    }

    // Like ReadAt, returns true on success, false for invalid destination arguments, and
    // propagates custom Reader failures as exceptions.
    // A null request array with a non-zero count is a caller bug and is rejected rather than
    // dereferenced. Entries are validated up front so a malformed request array is reported before
    // any custom Reader sees it: ReadFuncReader dereferences the destination without a check,
    // unlike CallbackStorage which throws std::invalid_argument.
    [[nodiscard]] bool
    ReadMany(const ReadRequest* requests, uint64_t count) const {
        ensure_initialized();
        if (count == 0) {
            return true;
        }
        if (requests == nullptr) {
            return false;
        }
        for (uint64_t i = 0; i < count; ++i) {
            if (requests[i].size > 0 and requests[i].destination == nullptr) {
                return false;
            }
        }
        for (uint64_t i = 0; i < count; ++i) {
            if (requests[i].size > 0) {
                reader_->Read(requests[i].offset, requests[i].size, requests[i].destination);
            }
        }
        return true;
    }

    // Preserve Reader::MultiRead's contract: false reports callback failures; validation and
    // custom Reader implementations may still throw. ByteIO callers must handle both channels.
    // Zero-size entries are filtered before dispatch so a custom MultiRead override never has to
    // expect them, matching the read-only ReaderIO backend.
    [[nodiscard]] bool
    ReadManyContiguous(uint8_t* destination,
                       const uint64_t* sizes,
                       const uint64_t* offsets,
                       uint64_t count) const {
        ensure_initialized();
        if (count == 0) {
            return true;
        }
        if (sizes == nullptr or offsets == nullptr) {
            return false;
        }
        // Fast path for the expected HGraph case: every code has the same fixed size, so there is
        // nothing to filter and no temporary allocation is needed on this hot read path.
        // The destination is validated before any allocation so the error path does no heap work.
        bool has_zero_size = false;
        for (uint64_t i = 0; i < count; ++i) {
            if (sizes[i] == 0) {
                has_zero_size = true;
                break;
            }
        }
        if (not has_zero_size) {
            if (destination == nullptr) {
                return false;
            }
            return reader_->MultiRead(destination, sizes, offsets, count);
        }
        // Slow path: zero-size entries must be filtered so a custom MultiRead override never sees
        // them. Everything below is only reached when such an entry exists.
        bool has_data = false;
        for (uint64_t i = 0; i < count; ++i) {
            if (sizes[i] > 0) {
                has_data = true;
                break;
            }
        }
        if (not has_data) {
            return true;
        }
        if (destination == nullptr) {
            return false;
        }
        std::vector<uint64_t> non_empty_sizes;
        std::vector<uint64_t> real_offsets;
        non_empty_sizes.reserve(count);
        real_offsets.reserve(count);
        for (uint64_t i = 0; i < count; ++i) {
            if (sizes[i] > 0) {
                non_empty_sizes.emplace_back(sizes[i]);
                real_offsets.emplace_back(offsets[i]);
            }
        }
        return reader_->MultiRead(
            destination, non_empty_sizes.data(), real_offsets.data(), non_empty_sizes.size());
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
        // Delegates to AllocateAndRead so Acquire and LegacyRead share one allocate-then-read
        // vocabulary: a future change to either path cannot leave the other stale.
        auto owner = AllocateAndRead(offset, size);
        auto* data = owner.Data();
        return Lease{data, size, std::move(owner)};
    }

    // Single allocate-then-read vocabulary shared by Acquire and LegacyRead. The returned owner
    // frees the buffer unless the caller takes ownership, so a throwing ReadAt cannot leak.
    // Requires size > 0: both callers return before reaching this on a zero-size read, and ReadAt
    // treats size == 0 as a trivially successful no-op that would leave the buffer uninitialized.
    [[nodiscard]] AllocatorOwner
    AllocateAndRead(uint64_t offset, uint64_t size) const {
        auto* data = static_cast<uint8_t*>(allocator_->Allocate(size));
        if (data == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                "UserDefinedBackend allocation failed");
        }
        AllocatorOwner owner(allocator_, data);
        if (not ReadAt(offset, size, data)) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "UserDefinedBackend failed to read into allocated buffer");
        }
        return owner;
    }

    [[nodiscard]] const uint8_t*
    LegacyRead(uint64_t offset, uint64_t size, bool& need_release) const {
        need_release = false;
        if (size == 0) {
            return nullptr;
        }
        // Shares AllocateAndRead with Acquire so both read paths clean up through AllocatorOwner
        // and cannot diverge in future changes.
        auto owner = AllocateAndRead(offset, size);
        auto* data = owner.Data();
        owner.Release();
        need_release = true;
        return data;
    }

    void
    Release(const uint8_t* data) const {
        allocator_->Deallocate(const_cast<uint8_t*>(data));
    }

    void
    WriteAt(uint64_t offset, const uint8_t* source, uint64_t size) {
        ensure_initialized();
        writer_->Write(offset, size, source);
    }

    void
    ResizePhysical(uint64_t size) {
        ensure_initialized();
        writer_->Resize(size);
    }

    // Shrinks physical storage to `size`; no-op for equal/larger sizes, never grows storage.
    //
    // WARNING: ShrinkPhysical is not atomic with respect to writes and resizes. It reads
    // reader_->Size() and only then calls writer_->Resize(size), so a concurrent WriteAt or
    // ResizePhysical landing in between can be truncated by the shrink and lose data. Callers
    // MUST serialize ShrinkPhysical against writes and resizes, including for custom Reader/Writer
    // pairs. Per-callback locking does not help: Reader/Writer expose no combined check-and-shrink
    // primitive, and holding a lock the backend does not own would not cover a custom pair that
    // does not share the bundled adapter's mutex.
    void
    ShrinkPhysical(uint64_t size) {
        ensure_initialized();
        if (size < reader_->Size()) {
            writer_->Resize(size);
        }
    }

    static void
    BindSerializedRange(uint64_t /*unused*/, uint64_t /*unused*/) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "user_defined_io cannot bind serialized ranges");
    }

    // Forward hints when the user Reader also implements ReaderPrefetcher, matching the read-only
    // ReaderIO backend. Hints are advisory: failures are suppressed and never affect later reads.
    void
    Prefetch(uint64_t offset, uint64_t len) {
        ensure_initialized();
        if (prefetcher_ == nullptr or len == 0) {
            return;
        }
        try {
            const auto size = reader_->Size();
            if (offset < size) {
                prefetcher_->Prefetch(offset, std::min(len, size - offset));
            }
        } catch (...) {
            // Prefetch is advisory; a failed size query or hint must not affect later reads.
        }
    }

    [[nodiscard]] static int64_t
    MemoryUsage(uint64_t /*unused*/) {
        return 0;
    }

    [[nodiscard]] static const uint8_t*
    Data() {
        return nullptr;
    }

    [[nodiscard]] Allocator*
    AllocatorPtr() const {
        return allocator_;
    }

private:
    void
    ensure_initialized() const {
        if (reader_ == nullptr or writer_ == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "user_defined_io is not initialized");
        }
    }

    Allocator* allocator_{nullptr};
    ReaderPtr reader_;
    WriterPtr writer_;
    // Non-owning view into reader_ when it also implements ReaderPrefetcher; valid only while
    // reader_ holds that object alive.
    ReaderPrefetcher* prefetcher_{nullptr};
};

}  // namespace vsag
