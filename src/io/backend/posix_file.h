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
#include <string>

namespace vsag {

enum class FileOwnership {
    Keep,
    DeleteOnClose,
};

[[nodiscard]] FileOwnership
OwnershipForPath(const std::string& path);

struct FileOpenOptions {
    std::string path;
    bool create{true};
    bool direct_read{false};
    bool separate_read_write_fd{false};
    FileOwnership ownership{FileOwnership::Keep};
};

class PosixFile {
public:
    explicit PosixFile(FileOpenOptions options);
    ~PosixFile();

    PosixFile(const PosixFile&) = delete;
    PosixFile&
    operator=(const PosixFile&) = delete;

    [[nodiscard]] int
    ReadFd() const {
        return read_fd_;
    }

    [[nodiscard]] int
    WriteFd() const {
        return write_fd_;
    }

    [[nodiscard]] uint64_t
    Size() const {
        return size_;
    }

    [[nodiscard]] const std::string&
    Path() const {
        return options_.path;
    }

    void
    Truncate(uint64_t size);

    void
    RecordWriteEnd(uint64_t end) {
        if (end > size_) {
            size_ = end;
        }
    }

private:
    void
    CleanupFailedConstruction() noexcept;

    FileOpenOptions options_;
    int read_fd_{-1};
    int write_fd_{-1};
    uint64_t size_{0};
};

}  // namespace vsag
