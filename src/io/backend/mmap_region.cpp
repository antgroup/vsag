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

#include "io/backend/mmap_region.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
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
throw_system_error(const char* operation, const std::string& path, uint64_t size = 0) {
    const int saved_errno = errno;
    throw VsagException(
        ErrorType::INTERNAL_ERROR,
        fmt::format("{}(path={}, size={}) failed (errno={}): {}",
                    operation,
                    path,
                    size,
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
            "{}(path={}) failed (error={}): {}", operation, path, error.value(), error.message()));
}

}  // namespace

MMapRegion::MMapRegion(std::string filename, Allocator* allocator)
    : allocator_(allocator), filepath_(std::move(filename)) {
    std::error_code error;
    const auto status = std::filesystem::status(filepath_, error);
    if (error and error != std::errc::no_such_file_or_directory) {
        throw_filesystem_error("inspect", filepath_, error);
    }
    existed_ = std::filesystem::exists(status);
    if (std::filesystem::is_directory(status)) {
        throw VsagException(ErrorType::INTERNAL_ERROR, fmt::format("{} is a directory", filepath_));
    }

    fd_ = open(filepath_.c_str(), O_CREAT | O_RDWR, 0644);
    if (fd_ < 0) {
        throw_system_error("open", filepath_);
    }

    try {
        struct stat file_stat {};
        if (fstat(fd_, &file_stat) != 0) {
            throw_system_error("fstat", filepath_);
        }
        initial_logical_size_ = static_cast<uint64_t>(file_stat.st_size);
        file_size_ = initial_logical_size_;
        mapped_capacity_ = std::max(initial_logical_size_, MINIMUM_MAPPING_SIZE);
        if (file_size_ < mapped_capacity_) {
            if (IOSyscall::FTruncate(fd_, mapped_capacity_) != 0) {
                throw_system_error("ftruncate", filepath_, mapped_capacity_);
            }
            file_size_ = mapped_capacity_;
        }

        void* address = mmap(nullptr, mapped_capacity_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (address == MAP_FAILED) {
            mapped_capacity_ = 0;
            throw_system_error("mmap", filepath_, file_size_);
        }
        mapped_data_ = static_cast<uint8_t*>(address);
    } catch (...) {
        CleanupFailedConstruction();
        throw;
    }
}

MMapRegion::~MMapRegion() {
    if (mapped_data_ != nullptr) {
        (void)munmap(mapped_data_, mapped_capacity_);
    }
    if (fd_ >= 0) {
        (void)close(fd_);
    }
    if (not existed_) {
        std::error_code ignored;
        std::filesystem::remove(filepath_, ignored);
    }
}

void
MMapRegion::EnsureCapacity(uint64_t size) {
    if (size > mapped_capacity_) {
        Remap(size);
    }
}

void
MMapRegion::ResizePhysical(uint64_t size) {
    Remap(std::max(size, MINIMUM_MAPPING_SIZE));
}

void
MMapRegion::ShrinkPhysical(uint64_t size) {
    Remap(std::max(size, MINIMUM_MAPPING_SIZE));
}

void
MMapRegion::Remap(uint64_t mapped_size) {
    if (mapped_size == mapped_capacity_) {
        return;
    }

    const uint64_t old_file_size = file_size_;
    if (IOSyscall::FTruncate(fd_, mapped_size) != 0) {
        throw_system_error("ftruncate", filepath_, mapped_size);
    }

#ifdef __APPLE__
    void* new_address = mmap(nullptr, mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (new_address == MAP_FAILED) {
        (void)IOSyscall::FTruncate(fd_, old_file_size);
        throw_system_error("mmap", filepath_, mapped_size);
    }
    if (munmap(mapped_data_, mapped_capacity_) != 0) {
        const int unmap_errno = errno;
        (void)munmap(new_address, mapped_size);
        (void)IOSyscall::FTruncate(fd_, old_file_size);
        errno = unmap_errno;
        throw_system_error("munmap", filepath_, mapped_capacity_);
    }
#else
    void* new_address = mremap(mapped_data_, mapped_capacity_, mapped_size, MREMAP_MAYMOVE);
    if (new_address == MAP_FAILED) {
        (void)IOSyscall::FTruncate(fd_, old_file_size);
        throw_system_error("mremap", filepath_, mapped_size);
    }
#endif

    mapped_data_ = static_cast<uint8_t*>(new_address);
    mapped_capacity_ = mapped_size;
    file_size_ = mapped_size;
}

void
MMapRegion::CleanupFailedConstruction() noexcept {
    if (mapped_data_ != nullptr) {
        (void)munmap(mapped_data_, mapped_capacity_);
        mapped_data_ = nullptr;
    }
    if (fd_ >= 0) {
        (void)close(fd_);
        fd_ = -1;
    }
    if (not existed_) {
        std::error_code ignored;
        std::filesystem::remove(filepath_, ignored);
    }
}

}  // namespace vsag
