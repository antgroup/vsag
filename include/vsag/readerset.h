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
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vsag {

/**
 * @enum IOErrorCode
 * @brief Enumeration for I/O error codes.
 *
 * This enum defines various error codes that can occur during I/O operations.
 */
enum class IOErrorCode {
    IO_SUCCESS = 0,  ///< Operation was successful.
    IO_ERROR = 1,    ///< General I/O error.
    IO_TIMEOUT = 2   ///< I/O operation timed out.
};

/**
 * @typedef CallBack
 * @brief Type alias for a callback function used in asynchronous I/O operations.
 *
 * The callback function takes two parameters: an I/O error code and an error message.
 */
using CallBack = std::function<void(IOErrorCode code, const std::string& message)>;

/**
 * @class Reader
 * @brief An abstract base class for reading data from various sources.
 *
 * The `Reader` class provides a standard interface for reading data synchronously and
 * asynchronously from files or memory. Implementations of this class must provide concrete
 * implementations for the pure virtual functions: Read & AsyncRead & Size.
 */
class Reader {
public:
    /// Default constructor.
    Reader() = default;

    /// Default destructor.
    virtual ~Reader() = default;

public:
    /**
     * @brief Reads a specified number of bytes from the data source.
     *
     * This pure virtual function synchronously reads `len` bytes from the source starting
     * at `offset` and copies them to the memory pointed to by `dest`. This method is thread-safe.
     *
     * @param offset The starting position for reading in the data source.
     * @param len The number of bytes to read.
     * @param dest Pointer to the memory where the read bytes will be copied.
     */
    virtual void
    Read(uint64_t offset, uint64_t len, void* dest) = 0;

    /**
     * @brief Asynchronously reads a specified number of bytes from the data source.
     *
     * This pure virtual function asynchronously reads `len` bytes from the source starting
     * at `offset` and copies them to the memory pointed to by `dest`. Upon completion, the
     * provided callback function is called with the result of the operation.
     *
     * @param offset The starting position for reading in the data source.
     * @param len The number of bytes to read.
     * @param dest Pointer to the memory where the read bytes will be copied.
     * @param callback Function to call upon completion with the result of the operation.
     */
    virtual void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, CallBack callback) = 0;

    /**
     * @brief Performs multiple synchronous read operations from the data source.
     *
     * This virtual function initiates a batch of read operations, each reading a specified
     * number of bytes from the source starting at a given offset and copying them to a
     * corresponding destination buffer. The parameters are provided as arrays where each
     * index corresponds to a single read operation.
     *
     * @param dests    Array of pointers to the memory buffers where the read bytes will be copied.
     *                 Each element corresponds to a specific read operation.
     * @param lens     Array of lengths (in bytes) for each read operation. Must match the
     *                 number of operations specified by `count`.
     * @param offsets  Array of starting positions in the data source for each read operation.
     *                 Must match the number of operations specified by `count`.
     * @param count    The total number of read operations to perform.
     *
     * @return         Returns `true` if all read operations completed successfully,
     *                 `false` otherwise (e.g., due to invalid parameters or I/O errors).
     *
     * @note This function is thread-safe and operates synchronously, blocking until all
     *       read operations are completed.
     * @note The arrays `dests`, `lens`, and `offsets` must be valid and contain exactly `count`
     *       elements. Passing null pointers or mismatched array sizes may result in undefined behavior.
     */
    virtual bool
    MultiRead(uint8_t* dests, const uint64_t* lens, const uint64_t* offsets, uint64_t count);

    /**
     * @brief Returns the size of the data source.
     *
     * This pure virtual function returns the total size of the data source.
     *
     * @return uint64_t The size of the data source.
     */
    [[nodiscard]] virtual uint64_t
    Size() const = 0;
};

/**
 * @class ReaderSet
 * @brief A class for managing a collection of `Reader` objects.
 *
 * The `ReaderSet` class allows associating `Reader` objects with string names for easy retrieval
 * and management. It supports adding, retrieving, and checking the existence of readers in the set.
 */
class ReaderSet {
public:
    /// Default constructor and destructor.
    ReaderSet() = default;
    ~ReaderSet() = default;

    /**
     * @brief Associates a `Reader` with a name and stores it in the set.
     *
     * This function associates a given `Reader` object with a specified name and stores it
     * in the set for future retrieval.
     *
     * @param name The name to associate with the `Reader`.
     * @param reader Shared pointer to the `Reader` to store.
     */
    void
    Set(const std::string& name, std::shared_ptr<Reader> reader) {
        data_[name] = std::move(reader);
    }

    /**
     * @brief Retrieves the `Reader` associated with a given name.
     *
     * This function retrieves the `Reader` object associated with the specified name.
     * If no `Reader` is associated with the name, it returns `nullptr`.
     *
     * @param name The name associated with the `Reader` to retrieve.
     * @return std::shared_ptr<Reader> Shared pointer to the `Reader` associated with the name, or `nullptr`.
     */
    std::shared_ptr<Reader>
    Get(const std::string& name) const {
        if (data_.find(name) == data_.end()) {
            return nullptr;
        }
        return data_.at(name);
    }

    /**
     * @brief Retrieves a list of all names.
     *
     * This function returns a vector containing all the names in the set.
     *
     * @return std::vector<std::string> A vector containing all the names.
     */
    std::vector<std::string>
    GetKeys() const {
        std::vector<std::string> keys;
        keys.resize(data_.size());
        transform(
            data_.begin(),
            data_.end(),
            keys.begin(),
            [](const std::pair<std::string, std::shared_ptr<Reader>>& pair) { return pair.first; });
        return keys;
    }

    /**
     * @brief Checks if a `Reader` is associated with a given name.
     *
     * This function checks if there is a `Reader` object associated with the specified name in the set.
     *
     * @param key The name to check for association with a `Reader`.
     * @return bool Returns `true` if a `Reader` is associated with the name, otherwise `false`.
     */
    bool
    Contains(const std::string& key) const {
        return data_.find(key) != data_.end();
    }

private:
    ///< Map storing `Reader` objects associated with names.
    std::unordered_map<std::string, std::shared_ptr<Reader>> data_;
};
}  // namespace vsag
