
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

namespace vsag {

MMapIO::MMapIO(std::string filename, Allocator* allocator)
    : BasicIO<MMapIO>(allocator), filepath_(std::move(filename)) {
    this->exist_file_ = std::filesystem::exists(this->filepath_);
    if (std::filesystem::is_directory(this->filepath_)) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("{} is a directory", this->filepath_));
    }

    this->fd_ = open(filepath_.c_str(), O_CREAT | O_RDWR, 0644);
    if (this->fd_ < 0) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("open file {} error {}", this->filepath_, strerror(errno)));
    }
    auto mmap_size = this->size_;
    if (this->size_ == 0) {
        mmap_size = DEFAULT_INIT_MMAP_SIZE;
        auto ret = ftruncate64(this->fd_, static_cast<int64_t>(mmap_size));
        if (ret == -1) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "ftruncate64 failed");
        }
    }
    void* addr = mmap(nullptr, mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd_, 0);
    this->start_ = static_cast<uint8_t*>(addr);
}

MMapIO::MMapIO(const MMapIOParameterPtr& io_param, const IndexCommonParam& common_param)
    : MMapIO(io_param->path_, common_param.allocator_.get()){};

MMapIO::MMapIO(const IOParamPtr& param, const IndexCommonParam& common_param)
    : MMapIO(std::dynamic_pointer_cast<MMapIOParameter>(param), common_param){};

MMapIO::~MMapIO() {
    munmap(this->start_, this->size_);
    close(this->fd_);
    // remove file
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
        auto ret = ftruncate64(this->fd_, static_cast<int64_t>(new_size));
        if (ret == -1) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "ftruncate64 failed");
        }
        this->start_ =
            static_cast<uint8_t*>(mremap(this->start_, old_size, new_size, MREMAP_MAYMOVE));
    }
    this->size_ = std::max(this->size_, new_size);
    memcpy(this->start_ + offset, data, size);
}

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
    need_release = false;
    if (offset + size > this->size_) {
        throw VsagException(
            ErrorType::INTERNAL_ERROR,
            fmt::format("read offset {} + size {} > size {}", offset, size, this->size_));
    }
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