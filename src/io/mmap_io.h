
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

#include <sys/mman.h>

#include <filesystem>
#include <utility>

#include "basic_io.h"
#include "index/index_common_param.h"
#include "mmap_io_parameter.h"

namespace vsag {

class MMapIO : public BasicIO<MMapIO> {
public:
    MMapIO(std::string filename, Allocator* allocator)
        : BasicIO<MMapIO>(allocator), filepath_(std::move(filename)) {
        this->fd_ = open(filepath_.c_str(), O_CREAT | O_RDWR, 0644);
        auto mmap_size = this->size_;
        if (this->size_ == 0) {
            mmap_size = DEFAULT_INIT_MMAP_SIZE;
            auto ret = ftruncate64(this->fd_, mmap_size);
            if (ret == -1) {
                throw VsagException(ErrorType::INTERNAL_ERROR, "ftruncate64 failed");
            }
        }
        void* addr = mmap(nullptr, mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd_, 0);
        this->start_ = static_cast<uint8_t*>(addr);
    }

    explicit MMapIO(const MMapIOParameterPtr& io_param, const IndexCommonParam& common_param)
        : MMapIO(io_param->path_, common_param.allocator_.get()){};

    explicit MMapIO(const IOParamPtr& param, const IndexCommonParam& common_param)
        : MMapIO(std::dynamic_pointer_cast<MMapIOParameter>(param), common_param){};

    ~MMapIO() override {
        munmap(this->start_, this->size_);
        close(this->fd_);
        // remove file
        std::filesystem::remove(this->filepath_);
    }

    inline void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset) {
        auto new_size = size + offset;
        auto old_size = this->size_;
        if (old_size == 0) {
            old_size = DEFAULT_INIT_MMAP_SIZE;
        }
        if (new_size > old_size) {
            auto ret = ftruncate64(this->fd_, new_size);
            if (ret == -1) {
                throw VsagException(ErrorType::INTERNAL_ERROR, "ftruncate64 failed");
            }
            this->start_ =
                static_cast<uint8_t*>(mremap(this->start_, old_size, new_size, MREMAP_MAYMOVE));
        }
        this->size_ = std::max(this->size_, new_size);
        memcpy(this->start_ + offset, data, size);
    }

    inline bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const {
        if (offset + size > this->size_) {
            throw VsagException(
                ErrorType::INTERNAL_ERROR,
                fmt::format("read offset {} + size {} > size {}", offset, size, this->size_));
        }
        memcpy(data, this->start_ + offset, size);
        return true;
    }

    [[nodiscard]] inline const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const {
        need_release = false;
        if (offset + size > this->size_) {
            throw VsagException(
                ErrorType::INTERNAL_ERROR,
                fmt::format("read offset {} + size {} > size {}", offset, size, this->size_));
        }
        return reinterpret_cast<const uint8_t*>(this->start_ + offset);
    }

    inline void
    ReleaseImpl(const uint8_t* data) const {
    }

    inline bool
    MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
        bool ret = true;
        for (uint64_t i = 0; i < count; ++i) {
            ret &= ReadImpl(sizes[i], offsets[i], datas);
            datas += sizes[i];
        }
        return ret;
    }

    inline void
    PrefetchImpl(uint64_t offset, uint64_t cache_line = 64){};

    static inline bool
    InMemoryImpl() {
        return false;
    }

    void
    InitIOImpl(const IOParamPtr& io_param) {
    }

    constexpr static int64_t DEFAULT_INIT_MMAP_SIZE = 4096;

private:
    std::string filepath_{};

    int fd_{-1};

    uint8_t* start_{nullptr};
};
}  // namespace vsag
