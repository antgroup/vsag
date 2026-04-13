
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
#include "mmap_io_parameter.h"

namespace vsag {

class IndexCommonParam;
class Allocator;

/**
 * @brief Memory-mapped file IO implementation.
 *
 * This class provides IO operations using memory mapping (mmap) for efficient
 * file access. It maps file contents directly into memory, allowing fast
 * random access without explicit read/write calls.
 */
class MMapIO : public BasicIO<MMapIO> {
public:
    /// Indicates this is not an in-memory IO implementation.
    static constexpr bool InMemory = false;

    /// Indicates deserialization is required when loading from disk.
    static constexpr bool SkipDeserialize = false;

public:
    /**
     * @brief Constructs a MMapIO object with a filename and allocator.
     *
     * @param filename The path to the file for IO operations.
     * @param allocator A pointer to the Allocator for memory management.
     */
    MMapIO(std::string filename, Allocator* allocator);

    /**
     * @brief Constructs a MMapIO object from MMapIOParameter.
     *
     * @param io_param The IO parameter containing configuration.
     * @param common_param The common index parameters.
     */
    explicit MMapIO(const MMapIOParamPtr& io_param, const IndexCommonParam& common_param);

    /**
     * @brief Constructs a MMapIO object from generic IOParamPtr.
     *
     * @param param The generic IO parameter pointer.
     * @param common_param The common index parameters.
     */
    explicit MMapIO(const IOParamPtr& param, const IndexCommonParam& common_param);

    /**
     * @brief Destructor that unmaps memory and closes file; optionally removes file.
     */
    ~MMapIO() override;

    /**
     * @brief Writes data to the mapped memory at a specified offset.
     *
     * @param data A pointer to the data to be written.
     * @param size The size of the data to be written.
     * @param offset The offset at which to write the data.
     */
    void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset);

    /**
     * @brief Resizes the mapped memory region to a specified size.
     *
     * @param size The new size of the mapped region.
     */
    void
    ResizeImpl(uint64_t size);

    /**
     * @brief Reads data from the mapped memory at a specified offset.
     *
     * @param size The size of the data to be read.
     * @param offset The offset at which to read the data.
     * @param data A pointer to the buffer where the read data will be stored.
     * @return True if the read operation was successful, false otherwise.
     */
    bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const;

    /**
     * @brief Reads data directly from the mapped memory without copying.
     *
     * @param size The size of the data to be read.
     * @param offset The offset at which to read the data.
     * @param need_release A reference to a boolean indicating whether the returned data needs to be released.
     * @return A pointer to the read data (direct pointer to mapped memory).
     */
    [[nodiscard]] const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const;

    /**
     * @brief Reads multiple blocks of data from the mapped memory in a single operation.
     *
     * @param datas A pointer to a contiguous buffer where all read data will be stored sequentially.
     * @param sizes An array of sizes for each block of data to be read.
     * @param offsets An array of offsets for each block of data to be read.
     * @param count The number of blocks of data to be read.
     * @return True if the read operation was successful, false otherwise.
     */
    bool
    MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const;

    /// Default initial size for memory mapping (4KB, one page).
    static constexpr int64_t DEFAULT_INIT_MMAP_SIZE = 4096;

private:
    /// Path to the file used for IO operations.
    std::string filepath_{};

    /// File descriptor for the mapped file.
    int fd_{-1};

    /// Pointer to the start of the memory-mapped region.
    uint8_t* start_{nullptr};

    /// Flag indicating if file existed before opening; false means file will be removed on destruction.
    bool exist_file_{false};
};
}  // namespace vsag
