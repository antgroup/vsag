
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

/**
 * @brief In-memory IO implementation using fixed-size memory blocks.
 *
 * This class manages data across multiple fixed-size memory blocks (default 128MB),
 * useful for large datasets that exceed single allocation limits. Each block is
 * independently allocated, allowing efficient memory management and avoiding
 * large contiguous allocations.
 */
class MemoryBlockIO : public BasicIO<MemoryBlockIO> {
public:
    /// Indicates this is an in-memory IO implementation.
    static constexpr bool InMemory = true;

    /// Indicates deserialization is required when loading.
    static constexpr bool SkipDeserialize = false;

public:
    /**
     * @brief Constructs a MemoryBlockIO with a specified block size.
     *
     * @param block_size The size of each memory block (default 128MB).
     * @param allocator A pointer to the Allocator for memory management.
     */
    explicit MemoryBlockIO(uint64_t block_size, Allocator* allocator);

    /**
     * @brief Constructs a MemoryBlockIO from MemoryBlockIOParameter.
     *
     * @param param The IO parameter containing configuration.
     * @param common_param The common index parameters.
     */
    explicit MemoryBlockIO(const MemoryBlockIOParamPtr& param,
                           const IndexCommonParam& common_param);

    /**
     * @brief Constructs a MemoryBlockIO from generic IOParamPtr.
     *
     * @param param The generic IO parameter pointer.
     * @param common_param The common index parameters.
     */
    explicit MemoryBlockIO(const IOParamPtr& param, const IndexCommonParam& common_param);

    /**
     * @brief Destructor that deallocates all memory blocks.
     */
    ~MemoryBlockIO() override;

    /**
     * @brief Writes data to the blocks at a specified offset.
     *
     * @param data A pointer to the data to be written.
     * @param size The size of the data to be written.
     * @param offset The offset at which to write the data.
     */
    void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset);

    /**
     * @brief Resizes the storage to a specified size by adding new blocks if needed.
     *
     * @param size The new total size of the storage.
     */
    void
    ResizeImpl(uint64_t size);

    /**
     * @brief Reads data from the blocks at a specified offset.
     *
     * @param size The size of the data to be read.
     * @param offset The offset at which to read the data.
     * @param data A pointer to the buffer where the read data will be stored.
     * @return True if the read operation was successful, false otherwise.
     */
    bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const;

    /**
     * @brief Reads data directly from blocks, allocating a new buffer if spanning multiple blocks.
     *
     * @param size The size of the data to be read.
     * @param offset The offset at which to read the data.
     * @param need_release A reference to a boolean indicating whether the returned data needs to be released.
     * @return A pointer to the read data.
     */
    [[nodiscard]] const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const;

    /**
     * @brief Releases data previously read via DirectReadImpl.
     *
     * @param data A pointer to the data to be released.
     */
    void
    ReleaseImpl(const uint8_t* data) const {
        auto ptr = const_cast<uint8_t*>(data);
        this->allocator_->Deallocate(ptr);
    };

    /**
     * @brief Reads multiple blocks of data from the storage in a single operation.
     *
     * @param datas A pointer to a contiguous buffer where all read data will be stored sequentially.
     * @param sizes An array of sizes for each block of data to be read.
     * @param offsets An array of offsets for each block of data to be read.
     * @param count The number of blocks of data to be read.
     * @return True if the read operation was successful, false otherwise.
     */
    bool
    MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const;

    /**
     * @brief Prefetches data from the blocks at a specified offset.
     *
     * @param offset The offset at which to prefetch the data.
     * @param cache_line The size of the cache line to prefetch.
     */
    void
    PrefetchImpl(uint64_t offset, uint64_t cache_line = 64);

private:
    /**
     * @brief Updates internal parameters after block size change.
     */
    void
    update_by_block_size();

    /**
     * @brief Checks and reallocates storage by adding new blocks if needed.
     *
     * @param size The required size for the storage.
     */
    void
    check_and_realloc(uint64_t size);

    /**
     * @brief Gets the pointer to data at a specified offset within blocks.
     *
     * @param offset The offset within the storage.
     * @return Pointer to the data at the offset.
     */
    [[nodiscard]] const uint8_t*
    get_data_ptr(uint64_t offset) const {
        auto block_no = offset >> block_bit_;
        auto block_off = offset & in_block_mask_;
        return blocks_[block_no] + block_off;
    }

    /**
     * @brief Checks if two offsets are within the same block.
     *
     * @param off1 First offset.
     * @param off2 Second offset.
     * @return True if both offsets are in the same block.
     */
    [[nodiscard]] bool
    check_in_one_block(uint64_t off1, uint64_t off2) const {
        return (off1 ^ off2) < block_size_;
    }

private:
    /// Size of each memory block (default 128MB).
    uint64_t block_size_{DEFAULT_BLOCK_SIZE};

    /// Vector of pointers to allocated memory blocks.
    Vector<uint8_t*> blocks_;

    /// Default block size: 128MB.
    static constexpr uint64_t DEFAULT_BLOCK_SIZE = 128 * 1024 * 1024;

    /// Default block bit count for bit operations (27 = log2(128MB)).
    static constexpr uint64_t DEFAULT_BLOCK_BIT = 27;

    /// Current block bit count for offset calculation.
    uint64_t block_bit_{DEFAULT_BLOCK_BIT};

    /// Mask for calculating offset within a block (block_size - 1).
    uint64_t in_block_mask_ = (1 << DEFAULT_BLOCK_BIT) - 1;
};
}  // namespace vsag
