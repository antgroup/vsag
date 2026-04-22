
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

#include "mmap_io.h"

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <cerrno>
#include <filesystem>
#include <system_error>
#include <utility>

#include "index_common_param.h"
#include "io_syscall.h"

namespace vsag {

#ifdef _WIN32

void
MMapIO::RemapWin32(uint64_t old_size, uint64_t new_size) {
    if (this->start_) {
        UnmapViewOfFile(this->start_);
        this->start_ = nullptr;
    }
    if (this->hMapping_) {
        CloseHandle(this->hMapping_);
        this->hMapping_ = NULL;
    }

    LARGE_INTEGER liSize;
    liSize.QuadPart = static_cast<LONGLONG>(new_size);
    if (!SetFilePointerEx(this->hFile_, liSize, NULL, FILE_BEGIN) ||
        !SetEndOfFile(this->hFile_)) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "SetEndOfFile failed");
    }

    this->hMapping_ = CreateFileMappingA(this->hFile_,
                                         NULL,
                                         PAGE_READWRITE,
                                         static_cast<DWORD>(new_size >> 32),
                                         static_cast<DWORD>(new_size & 0xFFFFFFFF),
                                         NULL);
    if (!this->hMapping_) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("CreateFileMapping failed: {}", GetLastError()));
    }

    void* addr = MapViewOfFile(this->hMapping_, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!addr) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("MapViewOfFile failed: {}", GetLastError()));
    }
    this->start_ = static_cast<uint8_t*>(addr);
}

MMapIO::MMapIO(std::string filename, Allocator* allocator)
    : BasicIO<MMapIO>(allocator), filepath_(std::move(filename)) {
    this->exist_file_ = std::filesystem::exists(this->filepath_);
    if (std::filesystem::is_directory(this->filepath_)) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("{} is a directory", this->filepath_));
    }

    this->hFile_ = CreateFileA(filepath_.c_str(),
                               GENERIC_READ | GENERIC_WRITE,
                               FILE_SHARE_READ,
                               NULL,
                               OPEN_ALWAYS,
                               FILE_ATTRIBUTE_NORMAL,
                               NULL);
    if (this->hFile_ == INVALID_HANDLE_VALUE) {
        throw VsagException(
            ErrorType::INTERNAL_ERROR,
            fmt::format("open file {} error {}", this->filepath_, GetLastError()));
    }

    auto mmap_size = this->size_;
    if (this->size_ == 0) {
        mmap_size = DEFAULT_INIT_MMAP_SIZE;
    }

    RemapWin32(0, mmap_size);
}

MMapIO::~MMapIO() {
    if (this->start_) {
        UnmapViewOfFile(this->start_);
    }
    if (this->hMapping_) {
        CloseHandle(this->hMapping_);
    }
    if (this->hFile_ != INVALID_HANDLE_VALUE) {
        CloseHandle(this->hFile_);
    }
    if (not this->exist_file_) {
        std::filesystem::remove(this->filepath_);
    }
}

void
MMapIO::WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset) {
    auto new_size = size + offset;
    auto old_size = this->size_;
    if (old_size == 0) {
        old_size = DEFAULT_INIT_MMAP_SIZE;
    }
    if (new_size > old_size) {
        RemapWin32(old_size, new_size);
    }
    this->size_ = std::max(this->size_, new_size);
    memcpy(this->start_ + offset, data, size);
}

void
MMapIO::ResizeImpl(uint64_t size) {
    auto new_size = size;
    auto old_size = this->size_;
    if (old_size == 0) {
        old_size = DEFAULT_INIT_MMAP_SIZE;
    }
    if (new_size != old_size) {
        RemapWin32(old_size, new_size);
    }
    this->size_ = new_size;
}

#else  // !_WIN32

namespace {

void
throw_mremap_error(uint64_t old_size, uint64_t new_size) {
    const int saved_errno = errno;
    throw VsagException(
        ErrorType::INTERNAL_ERROR,
        fmt::format("mremap(old_size={}, new_size={}) failed (errno={}): {}",
                    old_size,
                    new_size,
                    saved_errno,
                    std::error_code(saved_errno, std::system_category()).message()));
}

}  // namespace

MMapIO::MMapIO(std::string filename, Allocator* allocator)
    : BasicIO<MMapIO>(allocator), filepath_(std::move(filename)) {
    this->exist_file_ = std::filesystem::exists(this->filepath_);
    if (std::filesystem::is_directory(this->filepath_)) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("{} is a directory", this->filepath_));
    }

    this->fd_ = open(filepath_.c_str(), O_CREAT | O_RDWR, 0644);
    if (this->fd_ < 0) {
        auto saved_errno = errno;
        throw VsagException(
            ErrorType::INTERNAL_ERROR,
            fmt::format("open file {} failed (errno={}): {}",
                        this->filepath_,
                        saved_errno,
                        std::error_code(saved_errno, std::system_category()).message()));
    }
    auto mmap_size = this->size_;
    if (this->size_ == 0) {
        mmap_size = DEFAULT_INIT_MMAP_SIZE;
        auto ret = IOSyscall::FTruncate(this->fd_, mmap_size);
        if (ret == -1) {
            close(this->fd_);
            throw VsagException(ErrorType::INTERNAL_ERROR, "ftruncate failed");
        }
    }
    void* addr = mmap(nullptr, mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd_, 0);
    if (addr == MAP_FAILED) {
        auto saved_errno = errno;
        close(this->fd_);
        throw VsagException(
            ErrorType::INTERNAL_ERROR,
            fmt::format("mmap(size={}) failed (errno={}): {}",
                        mmap_size,
                        saved_errno,
                        std::error_code(saved_errno, std::system_category()).message()));
    }
    this->start_ = static_cast<uint8_t*>(addr);
}

MMapIO::~MMapIO() {
    munmap(this->start_, this->size_);
    close(this->fd_);
    if (not this->exist_file_) {
        std::filesystem::remove(this->filepath_);
    }
}

void
MMapIO::WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset) {
    auto new_size = size + offset;
    auto old_size = this->size_;
    if (old_size == 0) {
        old_size = DEFAULT_INIT_MMAP_SIZE;
    }
    if (new_size > old_size) {
        auto ret = IOSyscall::FTruncate(this->fd_, new_size);
        if (ret == -1) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "ftruncate failed");
        }

#ifdef __APPLE__
        munmap(this->start_, old_size);
        void* addr = mmap(nullptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd_, 0);
        if (addr == MAP_FAILED) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "mmap remap failed");
        }
        this->start_ = static_cast<uint8_t*>(addr);
#else
        void* new_addr = mremap(this->start_, old_size, new_size, MREMAP_MAYMOVE);
        if (new_addr == MAP_FAILED) {
            throw_mremap_error(old_size, new_size);
        }
        this->start_ = static_cast<uint8_t*>(new_addr);
#endif
    }
    this->size_ = std::max(this->size_, new_size);
    memcpy(this->start_ + offset, data, size);
}

void
MMapIO::ResizeImpl(uint64_t size) {
    auto new_size = size;
    auto old_size = this->size_;
    if (old_size == 0) {
        old_size = DEFAULT_INIT_MMAP_SIZE;
    }
    if (new_size > old_size) {
        auto ret = IOSyscall::FTruncate(this->fd_, new_size);
        if (ret == -1) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "ftruncate failed");
        }

#ifdef __APPLE__
        munmap(this->start_, old_size);
        void* addr = mmap(nullptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd_, 0);
        if (addr == MAP_FAILED) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "mmap remap failed");
        }
        this->start_ = static_cast<uint8_t*>(addr);
#else
        void* new_addr = mremap(this->start_, old_size, new_size, MREMAP_MAYMOVE);
        if (new_addr == MAP_FAILED) {
            throw_mremap_error(old_size, new_size);
        }
        this->start_ = static_cast<uint8_t*>(new_addr);
#endif
    } else if (new_size < old_size) {
#ifdef __APPLE__
        munmap(this->start_, old_size);
        void* addr = mmap(nullptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd_, 0);
        if (addr == MAP_FAILED) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "mmap remap failed");
        }
        this->start_ = static_cast<uint8_t*>(addr);
#else
        void* new_addr = mremap(this->start_, old_size, new_size, MREMAP_MAYMOVE);
        if (new_addr == MAP_FAILED) {
            throw_mremap_error(old_size, new_size);
        }
        this->start_ = static_cast<uint8_t*>(new_addr);
#endif
        auto ret = IOSyscall::FTruncate(this->fd_, new_size);
        if (ret == -1) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "ftruncate failed");
        }
    }
    this->size_ = new_size;
}

#endif  // _WIN32

MMapIO::MMapIO(const MMapIOParamPtr& io_param, const IndexCommonParam& common_param)
    : MMapIO(io_param->path_, common_param.allocator_.get()){};

MMapIO::MMapIO(const IOParamPtr& param, const IndexCommonParam& common_param)
    : MMapIO(std::dynamic_pointer_cast<MMapIOParameter>(param), common_param){};

bool
MMapIO::ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const {
    if (offset + size > this->size_) {
        throw VsagException(
            ErrorType::INTERNAL_ERROR,
            fmt::format("read offset {} + size {} > size {}", offset, size, this->size_));
    }
    memcpy(data, this->start_ + offset, size);
    return true;
}

[[nodiscard]] const uint8_t*
MMapIO::DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const {
    if (not check_valid_offset(size + offset)) {
        return nullptr;
    }
    need_release = false;
    return reinterpret_cast<const uint8_t*>(this->start_ + offset);
}

bool
MMapIO::MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
    bool ret = true;
    for (uint64_t i = 0; i < count; ++i) {
        ret &= ReadImpl(sizes[i], offsets[i], datas);
        datas += sizes[i];
    }
    return ret;
}

}  // namespace vsag
