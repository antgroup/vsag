
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

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <filesystem>

#include "basic_io.h"
#include "buffer_io_parameter.h"
#include "index_common_param.h"

namespace vsag {

/**
 * @brief Buffered file-based IO implementation.
 *
 * This class provides standard read/write operations using POSIX file APIs
 * with a single file descriptor. It manages file lifecycle and removes
 * temporary files on destruction unless the file existed before.
 */
class BufferIO : public BasicIO<BufferIO> {
public:
    /// Indicates this is not an in-memory IO implementation.
    static constexpr bool InMemory = false;

    /// Indicates deserialization is required when loading from disk.
    static constexpr bool SkipDeserialize = false;

public:
    /**
     * @brief Constructs a BufferIO object with a filename and allocator.
     *
     * @param filename The path to the file for IO operations.
     * @param allocator A pointer to the Allocator for memory management.
     */
    BufferIO(std::string filename, Allocator* allocator);

    /**
     * @brief Constructs a BufferIO object from BufferIOParameter.
     *
     * @param io_param The IO parameter containing configuration.
     * @param common_param The common index parameters.
     */
    explicit BufferIO(const BufferIOParameterPtr& io_param, const IndexCommonParam& common_param);

    /**
     * @brief Constructs a BufferIO object from generic IOParamPtr.
     *
     * @param param The generic IO parameter pointer.
     * @param common_param The common index parameters.
     */
    explicit BufferIO(const IOParamPtr& param, const IndexCommonParam& common_param);

    /**
     * @brief Destructor that closes the file descriptor and optionally removes the file.
     *
     * If exist_file_ is false (file was newly created), the file is removed.
     */
    ~BufferIO() override {
#ifdef _WIN32
        _close(this->fd_);
#else
        close(this->fd_);
#endif
        // remove file
        if (not this->exist_file_) {
            std::filesystem::remove(this->filepath_);
        }
    }

    /**
     * @brief Writes data to the file at a specified offset.
     *
     * @param data A pointer to the data to be written.
     * @param size The size of the data to be written.
     * @param offset The offset at which to write the data.
     */
    void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset);

    /**
     * @brief Resizes the file to a specified size.
     *
     * @param size The new size of the file.
     */
    void
    ResizeImpl(uint64_t size);

    /**
     * @brief Reads data from the file at a specified offset.
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
     * This method allocates a new buffer, reads data into it, and returns the pointer.
     * The caller must release the buffer using ReleaseImpl when need_release is true.
     *
     * @param size The size of the data to be read.
     * @param offset The offset at which to read the data.
     * @param need_release Set to true, indicating the returned buffer must be released by caller.
     * @return A pointer to the allocated buffer containing the read data.
     */
    [[nodiscard]] const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const;

    /**
     * @brief Releases data previously read from the file.
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
    MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const;

private:
    /// Path to the file used for IO operations.
    std::string filepath_{};

    /// File descriptor for both reading and writing operations.
    int fd_{-1};

    /// Flag indicating if file existed before opening; false means file will be removed on destruction.
    bool exist_file_{false};
};
}  // namespace vsag
