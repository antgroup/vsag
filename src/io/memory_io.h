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

#include <cstring>

#include "basic_io.h"
#include "index_common_param.h"
#include "memory_io_parameter.h"
#include "utils/prefetch.h"
#include "vsag_exception.h"

namespace vsag {

/**
 * @brief In-memory IO implementation using a dynamically allocated buffer.
 *
 * This class provides IO operations entirely in memory using a contiguous buffer
 * managed by the allocator. It is suitable for small datasets or testing scenarios
 * where disk persistence is not required.
 */
class MemoryIO : public BasicIO<MemoryIO> {
public:
    /// Indicates this is an in-memory IO implementation.
    static constexpr bool InMemory = true;

    /// Indicates deserialization is required when loading (memory needs to be allocated).
    static constexpr bool SkipDeserialize = false;

public:
    /**
     * @brief Constructs a MemoryIO object with an allocator.
     *
     * Allocates an initial buffer of 1 byte for basic operations.
     *
     * @param allocator A pointer to the Allocator for memory management.
     */
    explicit MemoryIO(Allocator* allocator) : BasicIO<MemoryIO>(allocator) {
        buffer_ = static_cast<uint8_t*>(allocator->Allocate(1));
        if (buffer_ == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                "failed to allocate initial buffer in MemoryIO");
        }
    }

    /**
     * @brief Constructs a MemoryIO object from MemoryIOParameter.
     *
     * @param param The IO parameter containing configuration.
     * @param common_param The common index parameters.
     */
    explicit MemoryIO(const MemoryIOParamPtr& param, const IndexCommonParam& common_param)
        : MemoryIO(common_param.allocator_.get()) {
    }

    /**
     * @brief Constructs a MemoryIO object from generic IOParamPtr.
     *
     * @param param The generic IO parameter pointer.
     * @param common_param The common index parameters.
     */
    explicit MemoryIO(const IOParamPtr& param, const IndexCommonParam& common_param)
        : MemoryIO(std::dynamic_pointer_cast<MemoryIOParameter>(param), common_param) {
    }

    /**
     * @brief Destructor that deallocates the memory buffer.
     */
    ~MemoryIO() override {
        this->allocator_->Deallocate(buffer_);
    }

    /**
     * @brief Writes data to the memory buffer at a specified offset.
     *
     * @param data A pointer to the data to be written.
     * @param size The size of the data to be written.
     * @param offset The offset at which to write the data.
     */
    void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset);

    /**
     * @brief Resizes the memory buffer to a specified size.
     *
     * @param size The new size of the buffer.
     */
    void
    ResizeImpl(uint64_t size);

    /**
     * @brief Reads data from the memory buffer at a specified offset.
     *
     * @param size The size of the data to be read.
     * @param offset The offset at which to read the data.
     * @param data A pointer to the buffer where the read data will be stored.
     * @return True if the read operation was successful, false otherwise.
     */
    bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const;

    /**
     * @brief Reads data directly from the memory buffer without copying.
     *
     * @param size The size of the data to be read.
     * @param offset The offset at which to read the data.
     * @param need_release A reference to a boolean indicating whether the returned data needs to be released.
     * @return A pointer to the read data (direct pointer to internal buffer).
     */
    [[nodiscard]] const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const;

    /**
     * @brief Reads multiple blocks of data into a contiguous buffer.
     *
     * Data blocks are read sequentially and written into a single contiguous buffer.
     * The buffer pointer is advanced after each block read.
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
     * @brief Prefetches data from the memory buffer at a specified offset.
     *
     * @param offset The offset at which to prefetch the data.
     * @param cache_line The size of the cache line to prefetch.
     */
    void
    PrefetchImpl(uint64_t offset, uint64_t cache_line = 64);

private:
    /**
     * @brief Checks the required size and reallocates the buffer if needed.
     *
     * Uses a temporary pointer for Reallocate result to prevent memory leak
     * on allocation failure. The original buffer pointer is only updated
     * after successful reallocation.
     *
     * @param size The required size for the buffer.
     */
    void
    check_and_realloc(uint64_t size) {
        if (size <= this->size_) {
            return;
        }
        uint8_t* new_buffer = static_cast<uint8_t*>(this->allocator_->Reallocate(buffer_, size));
        if (new_buffer == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                "failed to reallocate memory in MemoryIO, requested size: ",
                                size);
        }
        buffer_ = new_buffer;
        this->size_ = size;
    }

private:
    /// Pointer to the start of the allocated memory buffer.
    uint8_t* buffer_{nullptr};
};
}  // namespace vsag
