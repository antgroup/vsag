// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <fcntl.h>
#include <unistd.h>
#ifndef __APPLE__
#include <linux/falloc.h>
#endif

#include <cerrno>
#include <cstdint>

#ifdef _MSC_VER
#define FORCEINLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define FORCEINLINE inline __attribute__((always_inline))
#else
#define FORCEINLINE inline
#endif

namespace vsag {

class IOSyscall {
public:
    static FORCEINLINE ssize_t
    PRead(int fd, void* buf, size_t count, uint64_t offset) {
#ifdef __APPLE__
        return pread(fd, buf, count, static_cast<off_t>(offset));
#else
        return pread64(fd, buf, count, static_cast<int64_t>(offset));
#endif
    }

    static FORCEINLINE ssize_t
    PWrite(int fd, const void* buf, size_t count, uint64_t offset) {
#ifdef __APPLE__
        return pwrite(fd, buf, count, static_cast<off_t>(offset));
#else
        return pwrite64(fd, buf, count, static_cast<int64_t>(offset));
#endif
    }

    static FORCEINLINE int
    FTruncate(int fd, uint64_t length) {
#ifdef __APPLE__
        return ftruncate(fd, static_cast<off_t>(length));
#else
        return ftruncate64(fd, static_cast<int64_t>(length));
#endif
    }

    /// Reserve blocks for [0, length) so a later store into a mapping of this
    /// file cannot fail with SIGBUS once the filesystem is full, and so the
    /// extent is allocated in one step instead of by scattered page faults.
    /// The file size is left untouched, so the caller still sizes the file with
    /// FTruncate. This does not change the fact that unwritten bytes read as
    /// zero.
    ///
    /// Uses the raw fallocate(2) rather than posix_fallocate on purpose: glibc
    /// emulates posix_fallocate by writing zeros block by block when the
    /// filesystem lacks support, which silently turns a metadata-only
    /// reservation into an O(length) write and never reports EOPNOTSUPP.
    ///
    /// Returns 0 on success or an errno value; EOPNOTSUPP (or ENOTSUP where the
    /// platform has no equivalent at all) means the extent was not reserved,
    /// which callers should treat as a missing guarantee rather than a failure.
    static FORCEINLINE int
    Fallocate(int fd, uint64_t length) {
#ifdef __APPLE__
        (void)fd;
        (void)length;
        return ENOTSUP;
#else
        if (fallocate(fd, FALLOC_FL_KEEP_SIZE, 0, static_cast<off_t>(length)) == 0) {
            return 0;
        }
        return errno;
#endif
    }
};

}  // namespace vsag
