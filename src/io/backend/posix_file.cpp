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

#include "io/backend/posix_file.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <filesystem>
#include <system_error>
#include <utility>

#include "fmt/format.h"
#include "io/common/io_syscall.h"
#include "vsag_exception.h"

namespace vsag {

namespace {

[[noreturn]] void
throw_file_error(const char* operation, const std::string& path) {
    const int saved_errno = errno;
    throw VsagException(
        ErrorType::INTERNAL_ERROR,
        fmt::format("{} file {} failed (errno={}): {}",
                    operation,
                    path,
                    saved_errno,
                    std::error_code(saved_errno, std::system_category()).message()));
}

[[noreturn]] void
throw_filesystem_error(const char* operation,
                       const std::string& path,
                       const std::error_code& error) {
    throw VsagException(
        ErrorType::INTERNAL_ERROR,
        fmt::format(
            "{} file {} failed (error={}): {}", operation, path, error.value(), error.message()));
}

}  // namespace

FileOwnership
OwnershipForPath(const std::string& path) {
    std::error_code error;
    const auto status = std::filesystem::status(path, error);
    if (error and error != std::errc::no_such_file_or_directory) {
        throw_filesystem_error("inspect", path, error);
    }
    return std::filesystem::exists(status) ? FileOwnership::Keep : FileOwnership::DeleteOnClose;
}

PosixFile::PosixFile(FileOpenOptions options) : options_(std::move(options)) {
    std::error_code error;
    const auto status = std::filesystem::status(options_.path, error);
    if (error and error != std::errc::no_such_file_or_directory) {
        throw_filesystem_error("inspect", options_.path, error);
    }
    if (std::filesystem::is_directory(status)) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("{} is a directory", options_.path));
    }

    int common_flags = O_RDWR | (options_.create ? O_CREAT : 0);
    int read_flags = common_flags;
#ifdef O_DIRECT
    if (options_.direct_read) {
        read_flags |= O_DIRECT;
    }
#else
    if (options_.direct_read) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "direct file reads are unsupported on this platform");
    }
#endif
    read_fd_ = open(options_.path.c_str(), read_flags, 0644);
    if (read_fd_ < 0) {
        throw_file_error("open", options_.path);
    }

    try {
        if (options_.separate_read_write_fd) {
            write_fd_ = open(options_.path.c_str(), common_flags, 0644);
            if (write_fd_ < 0) {
                throw_file_error("open write descriptor for", options_.path);
            }
        } else {
            write_fd_ = read_fd_;
        }

        struct stat file_stat {};
        if (fstat(write_fd_, &file_stat) != 0) {
            throw_file_error("fstat", options_.path);
        }
        size_ = static_cast<uint64_t>(file_stat.st_size);
    } catch (...) {
        CleanupFailedConstruction();
        throw;
    }
}

PosixFile::~PosixFile() {
    if (write_fd_ >= 0 and write_fd_ != read_fd_) {
        (void)close(write_fd_);
    }
    if (read_fd_ >= 0) {
        (void)close(read_fd_);
    }
    if (options_.ownership == FileOwnership::DeleteOnClose) {
        std::error_code ignored;
        std::filesystem::remove(options_.path, ignored);
    }
}

void
PosixFile::Truncate(uint64_t size) {
    if (IOSyscall::FTruncate(write_fd_, size) != 0) {
        throw_file_error("ftruncate", options_.path);
    }
    size_ = size;
}

void
PosixFile::CleanupFailedConstruction() noexcept {
    if (write_fd_ >= 0 and write_fd_ != read_fd_) {
        (void)close(write_fd_);
    }
    if (read_fd_ >= 0) {
        (void)close(read_fd_);
    }
    write_fd_ = -1;
    read_fd_ = -1;
    if (options_.ownership == FileOwnership::DeleteOnClose) {
        std::error_code ignored;
        std::filesystem::remove(options_.path, ignored);
    }
}

}  // namespace vsag
