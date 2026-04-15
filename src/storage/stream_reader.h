
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

#include <cstdint>
#include <functional>
#include <istream>
#include <stack>

#include "../typing.h"
#include "impl/logger/logger.h"

namespace vsag {

class SliceStreamReader;

/**
 * @brief Abstract base class for stream reading operations.
 *
 * This class provides a unified interface for reading data from various stream sources,
 * including function-based readers, IO streams, and buffered readers.
 */
class StreamReader {
public:
    /**
     * @brief Reads an object of type T from the stream.
     *
     * @tparam T The type of the object to read.
     * @param reader The stream reader to read from.
     * @param val The object to store the read data.
     */
    template <typename T>
    static void
    ReadObj(StreamReader& reader, T& val) {
        reader.Read(reinterpret_cast<char*>(&val), sizeof(val));
    }

    /**
     * @brief Reads a string from the stream.
     *
     * @param reader The stream reader to read from.
     * @return The string read from the stream.
     */
    static std::string
    ReadString(StreamReader& reader) {
        uint64_t length = 0;
        StreamReader::ReadObj(reader, length);
        std::vector<char> buffer(length);
        reader.Read(buffer.data(), length);
        return {buffer.data(), length};
    }

    /**
     * @brief Reads a std::vector from the stream.
     *
     * @tparam T The element type of the vector.
     * @param reader The stream reader to read from.
     * @param val The vector to store the read data.
     */
    template <typename T>
    static void
    ReadVector(StreamReader& reader, std::vector<T>& val) {
        uint64_t size;
        ReadObj(reader, size);
        val.resize(size);
        reader.Read(reinterpret_cast<char*>(val.data()), size * sizeof(T));
    }

    /**
     * @brief Reads a vsag::Vector from the stream.
     *
     * @tparam T The element type of the vector.
     * @param reader The stream reader to read from.
     * @param val The vector to store the read data.
     */
    template <typename T>
    static void
    ReadVector(StreamReader& reader, vsag::Vector<T>& val) {
        uint64_t size;
        ReadObj(reader, size);
        val.resize(size);
        reader.Read(reinterpret_cast<char*>(val.data()), size * sizeof(T));
    }

public:
    /**
     * @brief Reads data from the stream.
     *
     * @param data Buffer to store the read data.
     * @param size Number of bytes to read.
     */
    virtual void
    Read(char* data, uint64_t size) = 0;

    /**
     * @brief Moves the read cursor to the specified position.
     *
     * @param cursor The target position in the stream.
     */
    virtual void
    Seek(uint64_t cursor) = 0;

    /**
     * @brief Gets the current read cursor position.
     *
     * @return Current position in the stream.
     */
    [[nodiscard]] virtual uint64_t
    GetCursor() const = 0;

    /**
     * @brief Gets the total length of the stream.
     *
     * @return Total length in bytes.
     */
    [[nodiscard]] virtual uint64_t
    Length() {
        return length_;
    }

public:
    /**
     * @brief Creates a slice reader from specified position with given length.
     *
     * @param begin Starting position of the slice.
     * @param length Length of the slice.
     * @return SliceStreamReader for the specified range.
     */
    [[nodiscard]] SliceStreamReader
    Slice(uint64_t begin, uint64_t length);

    /**
     * @brief Creates a slice reader from current position with given length.
     *
     * @param length Length of the slice.
     * @return SliceStreamReader for the specified range.
     */
    [[nodiscard]] SliceStreamReader
    Slice(uint64_t length);

    /**
     * @brief Saves current position and moves to specified position.
     *
     * @param cursor Target position to move to.
     */
    void
    PushSeek(uint64_t cursor) {
        positions_.push(this->GetCursor());
        this->Seek(cursor);
    }

    /**
     * @brief Restores the previously saved position.
     */
    void
    PopSeek() {
        this->Seek(positions_.top());
        positions_.pop();
    }

public:
    StreamReader() = default;
    StreamReader(uint64_t length) : length_(length) {
    }

protected:
    /// Total length of the stream in bytes.
    uint64_t length_{0};
    /// Count of IO operations performed.
    uint64_t io_count_{0};

private:
    /// Stack of saved positions for PushSeek/PopSeek operations.
    std::stack<uint64_t> positions_;
};

/**
 * @brief StreamReader implementation using a custom read function.
 *
 * This class wraps a function-based read operation into a stream reader interface.
 */
class ReadFuncStreamReader : public StreamReader {
public:
    void
    Read(char* data, uint64_t size) override;

    void
    Seek(uint64_t cursor) override;

    [[nodiscard]] uint64_t
    GetCursor() const override;

    ~ReadFuncStreamReader();

public:
    /**
     * @brief Constructs a reader with a custom read function.
     *
     * @param read_func Function that reads data from specified position.
     * @param cursor Initial cursor position.
     * @param length Total length of the stream.
     */
    ReadFuncStreamReader(std::function<void(uint64_t, uint64_t, void*)> read_func,
                         uint64_t cursor,
                         uint64_t length);

private:
    /// The custom read function for reading data.
    const std::function<void(uint64_t, uint64_t, void*)> readFunc_;
    /// Current cursor position in the stream.
    uint64_t cursor_{0};
};

/**
 * @brief StreamReader implementation wrapping an std::istream.
 *
 * This class provides a stream reader interface for standard C++ input streams.
 */
class IOStreamReader : public StreamReader {
public:
    void
    Read(char* data, uint64_t size) override;

    void
    Seek(uint64_t cursor) override;

    [[nodiscard]] uint64_t
    GetCursor() const override;

    ~IOStreamReader();

public:
    /**
     * @brief Constructs a reader wrapping an std::istream.
     *
     * @param istream The input stream to read from.
     */
    explicit IOStreamReader(std::istream& istream);

private:
    /// Reference to the underlying input stream.
    std::istream& istream_;
};

/**
 * @brief StreamReader implementation with internal buffering.
 *
 * This class provides buffered reading to improve performance for small reads.
 */
class BufferStreamReader : public StreamReader {
public:
    [[nodiscard]] uint64_t
    Length() override;

    void
    Read(char* data, uint64_t size) override;

    void
    Seek(uint64_t cursor) override;

    [[nodiscard]] uint64_t
    GetCursor() const override;

public:
    /**
     * @brief Constructs a buffered reader with specified maximum size.
     *
     * @param reader The underlying stream reader.
     * @param max_size Maximum size of the actual data stream.
     * @param allocator Allocator for buffer memory management.
     */
    explicit BufferStreamReader(StreamReader* reader, uint64_t max_size, Allocator* allocator);

    ~BufferStreamReader();

private:
    /// Pointer to the underlying stream reader implementation.
    StreamReader* const reader_impl_{nullptr};
    /// Allocator for managing buffer memory.
    vsag::Allocator* allocator_;
    /// Buffer storing cached content.
    char* buffer_{nullptr};
    /// Current read position in the cache.
    uint64_t buffer_cursor_{0};
    /// Size of valid data in the cache.
    uint64_t valid_size_{0};
    /// Maximum capacity of the cache buffer.
    uint64_t buffer_size_{0};
    /// Maximum capacity of the actual data stream.
    uint64_t max_size_{0};
    /// Current read position in the actual data stream.
    uint64_t cursor_{0};
};

/**
 * @brief StreamReader implementation for reading a slice of another stream.
 *
 * This class provides a restricted view of a parent stream within specified boundaries.
 */
class SliceStreamReader : public StreamReader {
public:
    [[nodiscard]] uint64_t
    Length() override;

    void
    Read(char* data, uint64_t size) override;

    void
    Seek(uint64_t cursor) override;

    [[nodiscard]] uint64_t
    GetCursor() const override;

public:
    /**
     * @brief Creates a slice from specified position.
     *
     * @param reader The parent stream reader.
     * @param begin Starting position of the slice.
     * @param length Length of the slice.
     */
    SliceStreamReader(StreamReader* reader, uint64_t begin, uint64_t length);
    /**
     * @brief Creates a slice from current position.
     *
     * @param reader The parent stream reader.
     * @param length Length of the slice.
     */
    SliceStreamReader(StreamReader* reader, uint64_t length);

private:
    /// Pointer to the parent stream reader.
    StreamReader* const reader_impl_{nullptr};
    /// Starting position of the slice in the parent stream.
    uint64_t begin_{0};
    /// Current position within the slice.
    uint64_t cursor_{0};
};

}  // namespace vsag
