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
#include "io/core/read_lease.h"
#include "vsag/allocator.h"
#include "vsag_exception.h"

namespace vsag {

class BufferedSingleRead {
public:
    using Lease = AllocatorLease;

    BufferedSingleRead() = default;

    explicit BufferedSingleRead(const IOEnvironment&) {
    }

    [[nodiscard]] bool
    ReadAt(const PosixFile& file, uint64_t offset, uint64_t size, uint8_t* destination) const {
        if (size == 0) {
            return true;
        }
        uint64_t completed = 0;
        while (completed < size) {
            ssize_t result = IOSyscall::PRead(
                file.ReadFd(), destination + completed, size - completed, offset + completed);
            if (result < 0) {
                int error = errno;
                if (error == EINTR) {
                    continue;
                }
                throw VsagException(
                    ErrorType::INTERNAL_ERROR,
                    fmt::format("buffered pread failed: fd={}, offset={}, size={}, completed={}, "
                                "errno={}",
                                file.ReadFd(),
                                offset,
                                size,
                                completed,
                                error));
            }
            if (result == 0) {
                throw VsagException(
                    ErrorType::INTERNAL_ERROR,
                    fmt::format("buffered pread reached unexpected EOF: fd={}, offset={}, size={}, "
                                "completed={}",
                                file.ReadFd(),
                                offset,
                                size,
                                completed));
            }
            completed += static_cast<uint64_t>(result);
        }
        return true;
    }

    [[nodiscard]] Lease
    Acquire(const PosixFile& file, Allocator* allocator, uint64_t offset, uint64_t size) const {
        if (size == 0) {
            return Lease{};
        }
        auto* data = static_cast<uint8_t*>(allocator->Allocate(size));
        if (data == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                "BufferedSingleRead allocation failed");
        }
        AllocatorOwner owner(allocator, data);
        (void)ReadAt(file, offset, size, data);
        return Lease(data, size, std::move(owner));
    }

    [[nodiscard]] const uint8_t*
    LegacyRead(const PosixFile& file,
               Allocator* allocator,
               uint64_t offset,
               uint64_t size,
               bool& need_release) const {
        need_release = false;
        if (size == 0) {
            return nullptr;
        }
        auto* data = static_cast<uint8_t*>(allocator->Allocate(size));
        if (data == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                "BufferedSingleRead allocation failed");
        }
        try {
            (void)ReadAt(file, offset, size, data);
        } catch (...) {
            allocator->Deallocate(data);
            throw;
        }
        need_release = true;
        return data;
    }

    void
    Release(Allocator* allocator, const uint8_t* data) const {
        allocator->Deallocate(const_cast<uint8_t*>(data));
    }
};

}  // namespace vsag
