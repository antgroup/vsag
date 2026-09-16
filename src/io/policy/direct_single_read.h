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
#include <cstdlib>
#include <cstring>
#include <utility>

#include "fmt/format.h"
#include "io/backend/posix_file.h"
#include "io/common/io_syscall.h"
#include "io/core/io_environment.h"
#include "io/core/io_utils.h"
#include "io/core/read_lease.h"
#include "vsag/options.h"
#include "vsag_exception.h"

namespace vsag {

class DirectReadBuffer {
public:
    DirectReadBuffer() = default;

    DirectReadBuffer(uint64_t requested_size, uint64_t requested_offset) {
        Initialize(requested_size, requested_offset, Alignment());
    }

    DirectReadBuffer(uint64_t requested_size, uint64_t requested_offset, uint64_t alignment) {
        Initialize(requested_size, requested_offset, alignment);
    }

    ~DirectReadBuffer() {
        Release();
    }

    DirectReadBuffer(const DirectReadBuffer&) = delete;
    DirectReadBuffer&
    operator=(const DirectReadBuffer&) = delete;

    DirectReadBuffer(DirectReadBuffer&& other) noexcept
        : base_(other.base_),
          data_(other.data_),
          submit_size_(other.submit_size_),
          submit_offset_(other.submit_offset_),
          requested_size_(other.requested_size_),
          prefix_(other.prefix_) {
        other.Clear();
    }

    DirectReadBuffer&
    operator=(DirectReadBuffer&& other) noexcept {
        if (this != &other) {
            Release();
            base_ = other.base_;
            data_ = other.data_;
            submit_size_ = other.submit_size_;
            submit_offset_ = other.submit_offset_;
            requested_size_ = other.requested_size_;
            prefix_ = other.prefix_;
            other.Clear();
        }
        return *this;
    }

    void
    Reset(uint64_t requested_size, uint64_t requested_offset) {
        Reset(requested_size, requested_offset, Alignment());
    }

    void
    Reset(uint64_t requested_size, uint64_t requested_offset, uint64_t alignment) {
        Release();
        Initialize(requested_size, requested_offset, alignment);
    }

    void
    Release() {
        if (base_ != nullptr) {
            std::free(base_);
        }
        Clear();
    }

    [[nodiscard]] static uint64_t
    Alignment() {
        return 1ULL << Options::Instance().direct_IO_object_align_bit();
    }

    [[nodiscard]] uint8_t*
    Base() const {
        return base_;
    }

    [[nodiscard]] uint8_t*
    Data() const {
        return data_;
    }

    [[nodiscard]] uint64_t
    SubmitSize() const {
        return submit_size_;
    }

    [[nodiscard]] uint64_t
    SubmitOffset() const {
        return submit_offset_;
    }

    [[nodiscard]] uint64_t
    MinimumResultSize() const {
        return prefix_ + requested_size_;
    }

    [[nodiscard]] uint64_t
    RequestedSize() const {
        return requested_size_;
    }

    [[nodiscard]] uint8_t*
    ReleaseBase() {
        uint8_t* base = base_;
        Clear();
        return base;
    }

private:
    void
    Clear() noexcept {
        base_ = nullptr;
        data_ = nullptr;
        submit_size_ = 0;
        submit_offset_ = 0;
        requested_size_ = 0;
        prefix_ = 0;
    }

    void
    Initialize(uint64_t requested_size, uint64_t requested_offset, uint64_t alignment) {
        requested_size_ = requested_size;
        uint64_t mask = alignment - 1;
        submit_offset_ = requested_offset & ~mask;
        prefix_ = requested_offset - submit_offset_;
        uint64_t required = CheckedEnd(prefix_, requested_size);
        if (required > UINT64_MAX - mask) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "direct read aligned size overflow");
        }
        submit_size_ = (required + mask) & ~mask;
        base_ = static_cast<uint8_t*>(std::aligned_alloc(alignment, submit_size_));
        if (base_ == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                "direct read aligned allocation failed");
        }
        data_ = base_ + prefix_;
    }

    uint8_t* base_{nullptr};
    uint8_t* data_{nullptr};
    uint64_t submit_size_{0};
    uint64_t submit_offset_{0};
    uint64_t requested_size_{0};
    uint64_t prefix_{0};
};

class DirectSingleRead {
public:
    using Lease = AlignedLease;

    DirectSingleRead() = default;

    explicit DirectSingleRead(const IOEnvironment&) {
    }

    [[nodiscard]] bool
    ReadAt(const PosixFile& file, uint64_t offset, uint64_t size, uint8_t* destination) const {
        if (size == 0) {
            return true;
        }
        DirectReadBuffer buffer(size, offset);
        ReadBuffer(file, buffer);
        std::memcpy(destination, buffer.Data(), size);
        return true;
    }

    [[nodiscard]] Lease
    Acquire(const PosixFile& file, Allocator*, uint64_t offset, uint64_t size) const {
        if (size == 0) {
            return Lease{};
        }
        auto buffer = AcquireBuffer(file, offset, size);
        const uint8_t* data = buffer.Data();
        uint8_t* base = buffer.ReleaseBase();
        return Lease(data, size, AlignedOwner(base));
    }

    [[nodiscard]] const uint8_t*
    LegacyRead(const PosixFile& file,
               Allocator*,
               uint64_t offset,
               uint64_t size,
               bool& need_release) const {
        need_release = false;
        if (size == 0) {
            return nullptr;
        }
        DirectReadBuffer buffer(size, offset);
        ReadBuffer(file, buffer);
        uint8_t* base = buffer.Base();
        // The legacy API must return the allocation base, while the requested bytes begin at an
        // aligned prefix within that same allocation; the source and destination can overlap.
        if (buffer.Data() != base) {
            std::memmove(base, buffer.Data(), size);
        }
        (void)buffer.ReleaseBase();
        need_release = true;
        return base;
    }

    void
    Release(Allocator*, const uint8_t* data) const {
        std::free(const_cast<uint8_t*>(data));
    }

    [[nodiscard]] DirectReadBuffer
    AcquireBuffer(const PosixFile& file, uint64_t offset, uint64_t size) const {
        DirectReadBuffer buffer(size, offset);
        ReadBuffer(file, buffer);
        return buffer;
    }

private:
    static void
    ReadBuffer(const PosixFile& file, const DirectReadBuffer& buffer) {
        ssize_t result;
        do {
            result = IOSyscall::PRead(
                file.ReadFd(), buffer.Base(), buffer.SubmitSize(), buffer.SubmitOffset());
        } while (result < 0 and errno == EINTR);
        if (result < 0) {
            int error = errno;
            throw VsagException(
                ErrorType::INTERNAL_ERROR,
                fmt::format(
                    "direct pread failed: fd={}, submit_offset={}, submit_size={}, errno={}",
                    file.ReadFd(),
                    buffer.SubmitOffset(),
                    buffer.SubmitSize(),
                    error));
        }
        if (static_cast<uint64_t>(result) < buffer.MinimumResultSize()) {
            throw VsagException(
                ErrorType::INTERNAL_ERROR,
                fmt::format(
                    "direct read bytes {} less than {}", result, buffer.MinimumResultSize()));
        }
    }
};

}  // namespace vsag
