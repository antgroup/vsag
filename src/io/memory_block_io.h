
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

#include "basic_io.h"
#include "memory_block_io_parameter.h"

namespace vsag {
class IndexCommonParam;

class MemoryBlockIO : public BasicIO<MemoryBlockIO> {
public:
    static constexpr bool InMemory = true;
    static constexpr bool SkipDeserialize = false;

public:
    explicit MemoryBlockIO(uint64_t block_size, Allocator* allocator);

    explicit MemoryBlockIO(const MemoryBlockIOParamPtr& param,
                           const IndexCommonParam& common_param);

    explicit MemoryBlockIO(const IOParamPtr& param, const IndexCommonParam& common_param);

    ~MemoryBlockIO() override;

    void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset);

    void
    ResizeImpl(uint64_t size);

    bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const;

    [[nodiscard]] const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const;

    void
    ReleaseImpl(const uint8_t* data) const {
        auto ptr = const_cast<uint8_t*>(data);
        this->allocator_->Deallocate(ptr);
    };

    bool
    MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const;

    void
    PrefetchImpl(uint64_t offset, uint64_t cache_line = 64);

private:
    void
    update_by_block_size();

    void
    check_and_realloc(uint64_t size);

    [[nodiscard]] const uint8_t*
    get_data_ptr(uint64_t offset) const {
        auto block_no = offset >> block_bit_;
        auto block_off = offset & in_block_mask_;
        return blocks_[block_no] + block_off;
    }

    [[nodiscard]] bool
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
}  // namespace vsag
