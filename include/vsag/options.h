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

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include "vsag/allocator.h"
#include "vsag/logger.h"

namespace vsag {

/**
 * @class Options
 * @brief Singleton class for configuring and obtaining settings for threaded operations,
 * block size limits, and logging.
 *
 * The `Options` class provides thread-safe methods to get and set various configuration
 * parameters such as the number of threads used for IO and building operations, the block size
 * limit for memory usage, and a logger instance for recording events.
 */
class Options {
public:
    /**
     * @brief Get the singleton instance of the Options class.
     *
     * @return Options& Reference to the singleton instance.
     */
    static Options&
    Instance();

    /* for testing
public:
    inline bool
    new_version() const {
        return new_version_;
    }

    inline void
    set_new_version(bool new_version) {
        new_version_ = new_version;
    }

private:
    bool new_version_ = true;
*/

public:
    /**
     * @brief Gets the number of threads for IO operations.
     *
     * This function retrieves the number of threads to use for disk index IO during the search process.
     * It is thread-safe, using memory order acquire operations.
     *
     * @return size_t The number of threads for IO operations.
     */
    [[nodiscard]] inline size_t
    num_threads_io() const {
        return num_threads_io_.load(std::memory_order_acquire);
    }

    /**
     * @brief Gets the number of threads for building operations.
     *
     * This function retrieves the number of threads used for constructing indices.
     * It is thread-safe, using memory order acquire operations.
     *
     * @return size_t The number of threads for building operations.
     */
    [[nodiscard]] inline size_t
    num_threads_building() const {
        return num_threads_building_.load(std::memory_order_acquire);
    }

    /**
     * @brief Sets the number of threads for IO operations in diskann.
     *
     * This function sets the number of threads to use for disk index IO during the search process.
     * The specified number of threads should be between 1 and 200.
     *
     * @param num_threads Number of threads for IO operations.
     */
    void
    set_num_threads_io(size_t num_threads);

    /**
     * @brief Sets the number of threads for building operations in diskann.
     *
     * This function sets the number of threads to use for constructing diskann index
     *
     * @param num_threads Number of threads for building operations.
     */
    void
    set_num_threads_building(size_t num_threads);

    /**
     * @brief Gets the limit of block size for memory allocations.
     *
     * This function retrieves the block size limit for memory allocations.
     * It is thread-safe, using memory order acquire operations.
     * The block size should be greater than 2M.
     *
     * @return size_t The block size limit for memory allocations.
     */
    [[nodiscard]] inline size_t
    block_size_limit() const {
        return block_size_limit_.load(std::memory_order_acquire);
    }

    /**
     * @brief Sets the limit of block size for memory allocations.
     *
     * This function sets the block size limit for memory allocations.
     *
     * @param size The size of the block limit, must be greater than 2M.
     */
    void
    set_block_size_limit(size_t size);

    /**
     * @brief Gets the size of direct IO object align bits.
     *
     * This function retrieves the size of direct IO object align bits.
     * It is thread-safe, using memory order acquire operations.
     * The size of direct IO object align bits should be smaller than 21(direct IO object smaller than 2M).
     *
     * @return size_t The size of direct IO object align bits.
     */
    [[nodiscard]] inline size_t
    direct_IO_object_align_bit() const {
        return direct_IO_object_align_bit_.load(std::memory_order_acquire);
    }

    /**
     * @brief Sets the size of direct IO object align bits.
     *
     * This function sets the size of direct IO object align bits.
     *
     * @param align_bit The size of direct IO object align bits.
     */
    void
    set_direct_IO_object_align_bit(size_t align_bit);

    /**
     * @brief Gets the current logger instance.
     *
     * @return Logger* Pointer to the current logger instance.
     */
    Logger*
    logger();

    /**
     * @brief Sets the logger instance.
     *
     * This function sets the logger used for recording events.
     *
     * @param logger Pointer to the logger instance to set.
     * @return bool Returns true if the logger is successfully set.
     */
    inline bool
    set_logger(Logger* logger) {
        logger_ = logger;
        return true;
    }

    // Deleted copy constructor and assignment operator to prevent copies.
    Options(const Options&) = delete;
    Options(const Options&&) = delete;
    Options&
    operator=(const Options&) = delete;

private:
    /// Private default constructor to ensure singleton pattern.
    Options() = default;

    /// Private default destructor.
    ~Options() = default;

private:
    ///< The size of the thread pool for single index I/O during searches.
    std::atomic<size_t> num_threads_io_{8};

    ///< The number of threads used for building a single index.
    std::atomic<size_t> num_threads_building_{4};

    ///< The size of the maximum memory allocated each time (default is 128MB).
    std::atomic<size_t> block_size_limit_{128 * 1024 * 1024};

    ///< The size of the bits used for DirectIOObject align (default is 9).
    std::atomic<size_t> direct_IO_object_align_bit_{9};

    ///< A flag to ensure that the set_direct_IO_object_align_bit() is called only once.
    std::atomic<bool> direct_IO_object_align_bit_flag{false};

    ///< Pointer to the logger instance.
    Logger* logger_ = nullptr;
};

/**
 * @typedef Option
 * @brief Type alias for the Options class, for compatibility.
 */
using Option = Options;
}  // namespace vsag
