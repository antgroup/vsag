
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
#include "index_common_param.h"
#include "reader_io_parameter.h"

namespace vsag {

/**
 * @brief Read-only IO implementation using an external Reader interface.
 *
 * This class provides IO operations through an external Reader object,
 * typically used for loading pre-built indexes from disk without modification.
 * The SkipDeserialize=true design indicates that during deserialization,
 * the data is not copied into memory; instead, the Reader directly accesses
 * the serialized data on disk or in memory.
 */
class ReaderIO : public BasicIO<ReaderIO> {
public:
    /// Indicates this is not an in-memory IO implementation.
    static constexpr bool InMemory = false;

    /// Indicates deserialization skips data copying; Reader directly accesses serialized data.
    static constexpr bool SkipDeserialize = true;

public:
    /**
     * @brief Constructs a ReaderIO object with an allocator.
     *
     * @param allocator A pointer to the Allocator for memory management.
     */
    explicit ReaderIO(Allocator* allocator) : BasicIO<ReaderIO>(allocator) {
    }

    /**
     * @brief Constructs a ReaderIO object from ReaderIOParameter.
     *
     * @param param The IO parameter containing configuration.
     * @param common_param The common index parameters.
     */
    explicit ReaderIO(const ReaderIOParamPtr& param, const IndexCommonParam& common_param)
        : ReaderIO(common_param.allocator_.get()) {
    }

    /**
     * @brief Constructs a ReaderIO object from generic IOParamPtr.
     *
     * @param param The generic IO parameter pointer.
     * @param common_param The common index parameters.
     */
    explicit ReaderIO(const IOParamPtr& param, const IndexCommonParam& common_param)
        : ReaderIO(std::dynamic_pointer_cast<ReaderIOParameter>(param), common_param) {
    }

    /**
     * @brief Default destructor.
     */
    ~ReaderIO() override = default;

    /**
     * @brief Writes data to the IO object (no-op for read-only IO).
     *
     * This method is a placeholder that only updates internal size tracking.
     * It does not actually write data as ReaderIO is read-only.
     *
     * @param data Ignored - data pointer (no-op).
     * @param size The size to update internal tracking.
     * @param offset Ignored - offset (no-op).
     */
    void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset);

    /**
     * @brief Initializes the IO object with an external Reader.
     *
     * @param io_param The IO parameter containing the Reader configuration.
     */
    void
    InitIOImpl(const IOParamPtr& io_param);

    /**
     * @brief Reads data from the Reader at a specified offset.
     *
     * @param size The size of the data to be read.
     * @param offset The offset at which to read the data.
     * @param data A pointer to the buffer where the read data will be stored.
     * @return True if the read operation was successful, false otherwise.
     */
    bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const;

    /**
     * @brief Reads data into an allocated buffer and returns a pointer to it.
     *
     * This method allocates a new buffer, reads data into it via the Reader,
     * and returns the pointer. The caller must release the buffer.
     *
     * @param size The size of the data to be read.
     * @param offset The offset at which to read the data.
     * @param need_release Set to true, indicating the returned buffer must be released by caller.
     * @return A pointer to the allocated buffer containing the read data.
     */
    [[nodiscard]] const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const;

    /**
     * @brief Releases data previously read from the Reader.
     *
     * @param data A pointer to the data to be released.
     */
    void
    ReleaseImpl(const uint8_t* data) const;

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
    MultiReadImpl(uint8_t* datas,
                  const uint64_t* sizes,
                  const uint64_t* offsets,
                  uint64_t count) const;

private:
    /// External Reader interface for accessing serialized data without copying.
    std::shared_ptr<Reader> reader_{nullptr};
};

}  // namespace vsag
