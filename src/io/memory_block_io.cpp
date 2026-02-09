
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

#include "memory_block_io.h"

#include <algorithm>
#include <cstring>

#include "common.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "utils/prefetch.h"

namespace vsag {

MemoryBlockIO::MemoryBlockIO(uint64_t block_size, Allocator* allocator)
    : BasicIO<MemoryBlockIO>(allocator),
      block_size_(MemoryBlockIOParameter::NearestPowerOfTwo(block_size)),
      blocks_(0, allocator) {
    this->update_by_block_size();
}

MemoryBlockIO::MemoryBlockIO(const MemoryBlockIOParamPtr& param,
                             const IndexCommonParam& common_param)
    : MemoryBlockIO(param->block_size_, common_param.allocator_.get()) {
}

MemoryBlockIO::MemoryBlockIO(const IOParamPtr& param, const IndexCommonParam& common_param)
    : MemoryBlockIO(std::dynamic_pointer_cast<MemoryBlockIOParameter>(param), common_param) {
}

MemoryBlockIO::~MemoryBlockIO() {
    for (auto* block : blocks_) {
        this->allocator_->Deallocate(block);
    }
}

void
MemoryBlockIO::WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset) {
    check_and_realloc(size + offset);
    uint64_t cur_size = 0;
    auto start_no = offset >> block_bit_;
    auto start_off = offset & in_block_mask_;
    auto max_size = block_size_ - start_off;
    while (cur_size < size) {
        uint8_t* cur_write = blocks_[start_no] + start_off;
        auto cur_length = std::min(size - cur_size, max_size);
        memcpy(cur_write, data + cur_size, cur_length);
        cur_size += cur_length;
        max_size = block_size_;
        ++start_no;
        start_off = 0;
    }
    if (size + offset > this->size_) {
        this->size_ = size + offset;
    }
}

bool
MemoryBlockIO::ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const {
    bool ret = check_valid_offset(size + offset);
    if (ret) {
        uint64_t cur_size = 0;
        auto start_no = offset >> block_bit_;
        auto start_off = offset & in_block_mask_;
        auto max_size = block_size_ - start_off;
        while (cur_size < size) {
            const uint8_t* cur_read = blocks_[start_no] + start_off;
            auto cur_length = std::min(size - cur_size, max_size);
            memcpy(data + cur_size, cur_read, cur_length);
            cur_size += cur_length;
            max_size = block_size_;
            ++start_no;
            start_off = 0;
        }
    }
    return ret;
}

const uint8_t*
MemoryBlockIO::DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const {
    if (check_valid_offset(size + offset)) {
        if (check_in_one_block(offset, size + offset)) {
            need_release = false;
            return this->get_data_ptr(offset);
        }
        need_release = true;
        auto* ptr = reinterpret_cast<uint8_t*>(this->allocator_->Allocate(size));
        this->ReadImpl(size, offset, ptr);
        return ptr;
    }
    return nullptr;
}
bool
MemoryBlockIO::MultiReadImpl(uint8_t* datas,
                             uint64_t* sizes,
                             uint64_t* offsets,
                             uint64_t count) const {
    bool ret = true;
    for (uint64_t i = 0; i < count; ++i) {
        ret &= this->ReadImpl(sizes[i], offsets[i], datas);
        datas += sizes[i];
    }
    return ret;
}
void
MemoryBlockIO::PrefetchImpl(uint64_t offset, uint64_t cache_line) {
    PrefetchLines(get_data_ptr(offset), cache_line);
}

void
MemoryBlockIO::check_and_realloc(uint64_t size) {
    if (size <= (blocks_.size() << block_bit_)) {
        return;
    }
    const uint64_t new_block_count = (size + this->block_size_ - 1) >> block_bit_;
    auto cur_block_size = this->blocks_.size();
    this->blocks_.reserve(new_block_count);
    while (cur_block_size < new_block_count) {
        auto* ptr = static_cast<uint8_t*>(this->allocator_->Allocate(block_size_));
        if (ptr == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "MemoryBlockIO allocation failed");
        }
        this->blocks_.emplace_back(ptr);
        ++cur_block_size;
    }
}

void
MemoryBlockIO::ResizeImpl(uint64_t size) {
    if (size <= this->size_) {
        this->size_ = size;
        return;
    }
    check_and_realloc(size);
    this->size_ = size;
}

static int
countr_zero(uint64_t x) {
    if (x == 0) {
        return 64;
    }
    int count = 0;
    while ((x & 1) == 0) {
        x >>= 1;
        ++count;
    }
    return count;
}

void
MemoryBlockIO::update_by_block_size() {
    this->block_bit_ = countr_zero(this->block_size_);
    this->in_block_mask_ = this->block_size_ - 1;
}

}  // namespace vsag
