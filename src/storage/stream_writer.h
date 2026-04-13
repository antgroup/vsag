
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
#include <ostream>

#include "typing.h"

namespace vsag {

/**
 * @brief Abstract base class for stream writing operations.
 *
 * This class provides a unified interface for writing data to various stream targets,
 * including buffers, IO streams, and function-based writers.
 */
class StreamWriter {
public:
    /**
     * @brief Writes an object of type T to the stream.
     *
     * @tparam T The type of the object to write.
     * @param writer The stream writer to write to.
     * @param val The object to write.
     */
    template <typename T>
    static void
    WriteObj(StreamWriter& writer, const T& val) {
        writer.Write(reinterpret_cast<const char*>(&val), sizeof(val));
    }

    /**
     * @brief Writes a string to the stream.
     *
     * @param writer The stream writer to write to.
     * @param str The string to write.
     */
    static void
    WriteString(StreamWriter& writer, const std::string& str) {
        uint64_t length = str.size();
        StreamWriter::WriteObj(writer, length);
        writer.Write(str.c_str(), length);
    }

    /**
     * @brief Writes a std::vector to the stream.
     *
     * @tparam T The element type of the vector.
     * @param writer The stream writer to write to.
     * @param val The vector to write.
     */
    template <typename T>
    static void
    WriteVector(StreamWriter& writer, const std::vector<T>& val) {
        uint64_t size = val.size();
        WriteObj(writer, size);
        if (size > 0) {
            writer.Write(reinterpret_cast<const char*>(val.data()), size * sizeof(T));
        }
    }

    /**
     * @brief Writes a vsag::Vector to the stream.
     *
     * @tparam T The element type of the vector.
     * @param writer The stream writer to write to.
     * @param val The vector to write.
     */
    template <typename T>
    static void
    WriteVector(StreamWriter& writer, const vsag::Vector<T>& val) {
        uint64_t size = val.size();
        WriteObj(writer, size);
        if (size > 0) {
            writer.Write(reinterpret_cast<const char*>(val.data()), size * sizeof(T));
        }
    }

public:
    /**
     * @brief Writes data to the stream.
     *
     * @param data Pointer to the data to write.
     * @param size Number of bytes to write.
     */
    virtual void
    Write(const char* data, uint64_t size) = 0;

    /**
     * @brief Gets the number of bytes written so far.
     *
     * @return Number of bytes written.
     */
    [[nodiscard]] uint64_t
    GetCursor() const {
        return bytes_written_;
    }

public:
    StreamWriter() = default;

    virtual ~StreamWriter() = default;

protected:
    /// Total number of bytes written to the stream.
    uint64_t bytes_written_{0};
};

/**
 * @brief StreamWriter implementation writing to a memory buffer.
 *
 * This class provides stream writing to a pre-allocated memory buffer.
 */
class BufferStreamWriter : public StreamWriter {
public:
    /**
     * @brief Constructs a writer for the specified buffer.
     *
     * @param buffer Pointer to the memory buffer to write to.
     */
    explicit BufferStreamWriter(char* buffer);

    void
    Write(const char* data, uint64_t size) override;

private:
    /// Pointer to the memory buffer for writing.
    char* buffer_{nullptr};
};

/**
 * @brief StreamWriter implementation wrapping an std::ostream.
 *
 * This class provides stream writing to standard C++ output streams.
 */
class IOStreamWriter : public StreamWriter {
public:
    /**
     * @brief Constructs a writer wrapping an std::ostream.
     *
     * @param ostream The output stream to write to.
     */
    explicit IOStreamWriter(std::ostream& ostream);

    void
    Write(const char* data, uint64_t size) override;

private:
    /// Reference to the underlying output stream.
    std::ostream& ostream_;
    /// Number of bytes written to the stream.
    uint64_t written_bytes_{0};
};

/**
 * @brief StreamWriter implementation using a custom write function.
 *
 * This class wraps a function-based write operation into a stream writer interface.
 */
class WriteFuncStreamWriter : public StreamWriter {
public:
    /**
     * @brief Constructs a writer with a custom write function.
     *
     * @param writeFunc Function that writes data at specified position.
     * @param cursor Initial cursor position.
     */
    explicit WriteFuncStreamWriter(std::function<void(uint64_t, uint64_t, void*)> writeFunc,
                                   uint64_t cursor);

    void
    Write(const char* data, uint64_t size) override;

    /// The custom write function for writing data.
    std::function<void(uint64_t, uint64_t, void*)> writeFunc_;

    /// Current cursor position in the stream.
    uint64_t cursor_{0};
    /// Number of bytes written to the stream.
    uint64_t written_bytes_{0};
};

}  // namespace vsag
