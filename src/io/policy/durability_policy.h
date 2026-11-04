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

#include <unistd.h>

#include <cerrno>

#include "fmt/format.h"
#include "vsag_exception.h"

namespace vsag {

struct NoFlush {
    void
    AfterWrite(int) const {
    }
};

struct FsyncAfterWrite {
    void
    AfterWrite(int fd) const {
        int result;
        do {
            result = fsync(fd);
        } while (result != 0 and errno == EINTR);
        if (result != 0) {
            const int error = errno;
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("fsync failed: fd={}, errno={}", fd, error));
        }
    }
};

}  // namespace vsag
