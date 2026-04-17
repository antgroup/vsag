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

#ifdef _WIN32
#include <io.h>
#include <windows.h>

#include <cstdint>

typedef ptrdiff_t ssize_t;
#else
#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#endif

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
#ifdef _WIN32
        HANDLE hFile = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
        if (hFile == INVALID_HANDLE_VALUE)
            return -1;
        OVERLAPPED ov = {};
        ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
        ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
        DWORD bytesRead = 0;
        if (!ReadFile(hFile, buf, static_cast<DWORD>(count), &bytesRead, &ov))
            return -1;
        return static_cast<ssize_t>(bytesRead);
#elif defined(__APPLE__)
        return pread(fd, buf, count, static_cast<off_t>(offset));
#else
        return pread64(fd, buf, count, static_cast<int64_t>(offset));
#endif
    }

    static FORCEINLINE ssize_t
    PWrite(int fd, const void* buf, size_t count, uint64_t offset) {
#ifdef _WIN32
        HANDLE hFile = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
        if (hFile == INVALID_HANDLE_VALUE)
            return -1;
        OVERLAPPED ov = {};
        ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
        ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
        DWORD bytesWritten = 0;
        if (!WriteFile(hFile, buf, static_cast<DWORD>(count), &bytesWritten, &ov))
            return -1;
        return static_cast<ssize_t>(bytesWritten);
#elif defined(__APPLE__)
        return pwrite(fd, buf, count, static_cast<off_t>(offset));
#else
        return pwrite64(fd, buf, count, static_cast<int64_t>(offset));
#endif
    }

    static FORCEINLINE int
    FTruncate(int fd, uint64_t length) {
#ifdef _WIN32
        return _chsize_s(fd, static_cast<long long>(length));
#elif defined(__APPLE__)
        return ftruncate(fd, static_cast<off_t>(length));
#else
        return ftruncate64(fd, static_cast<int64_t>(length));
#endif
    }
};

}  // namespace vsag
