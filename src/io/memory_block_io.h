
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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "basic_io.h"
#include "common.h"
#include "index/index_common_param.h"
#include "inner_string_params.h"
#include "memory_block_io_parameter.h"
#include "prefetch.h"
#include "vsag/allocator.h"

namespace vsag {

class MemoryBlockIO : public BasicIO<MemoryBlockIO> {
public:
    explicit MemoryBlockIO(Allocator* allocator, uint64_t block_size)
        : BasicIO<MemoryBlockIO>(allocator),
          block_size_(MemoryBlockIOParameter::NearestPowerOfTwo(block_size)),
          blocks_(0, allocator) {
        this->update_by_block_size();
    }

    explicit MemoryBlockIO(const MemoryBlockIOParamPtr& param, const IndexCommonParam& common_param)
        : MemoryBlockIO(common_param.allocator_.get(), param->block_size_){};

    explicit MemoryBlockIO(const IOParamPtr& param, const IndexCommonParam& common_param)
        : MemoryBlockIO(std::dynamic_pointer_cast<MemoryBlockIOParameter>(param), common_param){};

    ~MemoryBlockIO() override {
        for (auto* block : blocks_) {
            this->allocator_->Deallocate(block);
        }
    }

    inline void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset);

    inline bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const;

    [[nodiscard]] inline const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const;

    inline void
    ReleaseImpl(const uint8_t* data) const {
        auto ptr = const_cast<uint8_t*>(data);
        this->allocator_->Deallocate(ptr);
    };

    inline bool
    MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const;

    inline void
    PrefetchImpl(uint64_t offset, uint64_t cache_line = 64);

    static inline bool
    InMemoryImpl() {
        return true;
    }

private:
    inline int
    countr_zero(uint64_t x) {
        if (x == 0)
            return 64;

        int count = 0;
        while ((x & 1) == 0) {
            x >>= 1;
            ++count;
        }
        return count;
    }

    inline void
    update_by_block_size() {
        block_bit_ = countr_zero(block_size_);
        in_block_mask_ = block_size_ - 1;
    }

    inline void
    check_and_realloc(uint64_t size);

    [[nodiscard]] inline const uint8_t*
    get_data_ptr(uint64_t offset) const {
        auto block_no = offset >> block_bit_;
        auto block_off = offset & in_block_mask_;
        return blocks_[block_no] + block_off;
    }

    [[nodiscard]] inline bool
    check_in_one_block(uint64_t off1, uint64_t off2) const {
        return (off1 ^ off2) < block_size_;
    }

private:
    uint64_t block_size_{DEFAULT_BLOCK_SIZE};

    Vector<uint8_t*> blocks_;

    static constexpr uint64_t DEFAULT_BLOCK_SIZE = 128 * 1024 * 1024;  // 128MB

    static constexpr uint64_t DEFAULT_BLOCK_BIT = 27;

    uint64_t block_bit_{DEFAULT_BLOCK_BIT};

    uint64_t in_block_mask_ = (1 << DEFAULT_BLOCK_BIT) - 1;
};

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
        } else {
            need_release = true;
            auto* ptr = reinterpret_cast<uint8_t*>(this->allocator_->Allocate(size));
            this->ReadImpl(size, offset, ptr);
            return ptr;
        }
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
        this->blocks_.emplace_back((uint8_t*)(this->allocator_->Allocate(block_size_)));
        ++cur_block_size;
    }
}

}  // namespace vsag
