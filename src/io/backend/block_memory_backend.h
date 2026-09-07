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

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "container_types.h"
#include "io/core/read_lease.h"
#include "io/core/read_operation.h"
#include "io/core/read_request.h"
#include "io/memory_block_io/memory_block_io_parameter.h"
#include "utils/prefetch.h"
#include "vsag/allocator.h"
#include "vsag_exception.h"

namespace vsag {

struct BlockMemoryBackendCapabilities {
    static constexpr bool InMemory = true;
    static constexpr bool RequiresInitialization = false;
    static constexpr bool CanBindSerializedRange = false;
    static constexpr bool LegacyBatchRangeThrows = false;
    static constexpr bool LegacyUncheckedReadable = false;
    static constexpr bool BorrowedReadable = true;
    static constexpr bool BatchReadable = true;
    static constexpr bool AsyncReadable = false;
    static constexpr bool Writable = true;
    static constexpr bool Resizable = true;
};

class BlockMemoryBackend {
public:
    using Capabilities = BlockMemoryBackendCapabilities;
    using Lease = AllocatorLease;
    using Operation = ImmediateOperation;

    explicit BlockMemoryBackend(uint64_t block_size, Allocator* allocator)
        : allocator_(allocator),
          block_size_(MemoryBlockIOParameter::NearestPowerOfTwo(block_size)),
          blocks_(AllocatorWrapper<uint8_t*>(allocator)) {
        if (block_size_ == 0) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "BlockMemoryBackend block size must be greater than zero");
        }
        block_bit_ = CountTrailingZeros(block_size_);
        in_block_mask_ = block_size_ - 1;
    }

    ~BlockMemoryBackend() {
        for (auto* block : blocks_) {
            allocator_->Deallocate(block);
        }
    }

    BlockMemoryBackend(const BlockMemoryBackend&) = delete;
    BlockMemoryBackend&
    operator=(const BlockMemoryBackend&) = delete;

    [[nodiscard]] uint64_t
    InitialLogicalSize() const {
        return 0;
    }

    [[nodiscard]] bool
    ReadAt(uint64_t offset, uint64_t size, uint8_t* destination) const {
        CopyOut(offset, size, destination);
        return true;
    }

    [[nodiscard]] bool
    ReadMany(const ReadRequest* requests, uint64_t count) const {
        for (uint64_t i = 0; i < count; ++i) {
            if (requests[i].size > 0) {
                CopyOut(requests[i].offset, requests[i].size, requests[i].destination);
            }
        }
        return true;
    }

    [[nodiscard]] bool
    ReadManyContiguous(uint8_t* destination,
                       const uint64_t* sizes,
                       const uint64_t* offsets,
                       uint64_t count) const {
        for (uint64_t i = 0; i < count; ++i) {
            if (sizes[i] > 0) {
                CopyOut(offsets[i], sizes[i], destination);
                destination += sizes[i];
            }
        }
        return true;
    }

    [[nodiscard]] Operation
    SubmitReads(const ReadRequest* requests, uint64_t count) const {
        return Operation(ReadMany(requests, count));
    }

    [[nodiscard]] Lease
    Acquire(uint64_t offset, uint64_t size) const {
        if (size == 0) {
            return Lease{};
        }
        if (FitsInOneBlock(offset, size)) {
            return Lease(DataAt(offset), size, AllocatorOwner{});
        }
        auto* data = static_cast<uint8_t*>(allocator_->Allocate(size));
        if (data == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                "BlockMemoryBackend acquire allocation failed");
        }
        CopyOut(offset, size, data);
        return Lease(data, size, AllocatorOwner(allocator_, data));
    }

    [[nodiscard]] const uint8_t*
    LegacyRead(uint64_t offset, uint64_t size, bool& need_release) const {
        if (size == 0) {
            need_release = false;
            return nullptr;
        }
        if (FitsInOneBlock(offset, size)) {
            need_release = false;
            return DataAt(offset);
        }
        auto* data = static_cast<uint8_t*>(allocator_->Allocate(size));
        if (data == nullptr) {
            need_release = false;
            return nullptr;
        }
        CopyOut(offset, size, data);
        need_release = true;
        return data;
    }

    void
    Release(const uint8_t* data) const {
        allocator_->Deallocate(const_cast<uint8_t*>(data));
    }

    void
    WriteAt(uint64_t offset, const uint8_t* source, uint64_t size) {
        EnsureCapacity(offset + size);
        uint64_t copied = 0;
        uint64_t block_index = offset >> block_bit_;
        uint64_t block_offset = offset & in_block_mask_;
        while (copied < size) {
            uint64_t copy_size = std::min(size - copied, block_size_ - block_offset);
            std::memcpy(blocks_[block_index] + block_offset, source + copied, copy_size);
            copied += copy_size;
            ++block_index;
            block_offset = 0;
        }
    }

    void
    ResizePhysical(uint64_t size) {
        EnsureCapacity(size);
    }

    void
    ShrinkPhysical(uint64_t size) {
        uint64_t block_count = size == 0 ? 0 : ((size - 1) >> block_bit_) + 1;
        while (blocks_.size() > block_count) {
            allocator_->Deallocate(blocks_.back());
            blocks_.pop_back();
        }
    }

    void
    Prefetch(uint64_t offset, uint64_t cache_line) {
        PrefetchLines(DataAt(offset),
                      std::min(cache_line, block_size_ - (offset & in_block_mask_)));
    }

    [[nodiscard]] int64_t
    MemoryUsage(uint64_t) const {
        return static_cast<int64_t>(blocks_.size() * block_size_);
    }

    [[nodiscard]] const uint8_t*
    Data() const {
        return nullptr;
    }

    [[nodiscard]] Allocator*
    AllocatorPtr() const {
        return allocator_;
    }

private:
    [[nodiscard]] static uint64_t
    CountTrailingZeros(uint64_t value) {
        uint64_t count = 0;
        while ((value & 1U) == 0) {
            value >>= 1U;
            ++count;
        }
        return count;
    }

    [[nodiscard]] bool
    FitsInOneBlock(uint64_t offset, uint64_t size) const {
        return size <= block_size_ - (offset & in_block_mask_);
    }

    [[nodiscard]] const uint8_t*
    DataAt(uint64_t offset) const {
        return blocks_[offset >> block_bit_] + (offset & in_block_mask_);
    }

    void
    CopyOut(uint64_t offset, uint64_t size, uint8_t* destination) const {
        uint64_t copied = 0;
        uint64_t block_index = offset >> block_bit_;
        uint64_t block_offset = offset & in_block_mask_;
        while (copied < size) {
            uint64_t copy_size = std::min(size - copied, block_size_ - block_offset);
            std::memcpy(destination + copied, blocks_[block_index] + block_offset, copy_size);
            copied += copy_size;
            ++block_index;
            block_offset = 0;
        }
    }

    void
    EnsureCapacity(uint64_t size) {
        uint64_t required_blocks = size == 0 ? 0 : ((size - 1) >> block_bit_) + 1;
        blocks_.reserve(required_blocks);
        while (blocks_.size() < required_blocks) {
            auto* block = static_cast<uint8_t*>(allocator_->Allocate(block_size_));
            if (block == nullptr) {
                throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                    "BlockMemoryBackend allocation failed");
            }
            std::memset(block, 0, block_size_);
            blocks_.emplace_back(block);
        }
    }

    Allocator* allocator_{nullptr};
    uint64_t block_size_{0};
    Vector<uint8_t*> blocks_;
    uint64_t block_bit_{0};
    uint64_t in_block_mask_{0};
};

}  // namespace vsag
