
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

#include <fmt/format.h>

#include <cstdint>

#include "byte_buffer.h"
#include "io_parameter.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/function_exists_check.h"

namespace vsag {

/**
 * @brief A template class for basic input/output operations.
 *
 * This class provides a set of methods for reading, writing, and managing data.
 * The class is templated on the type of the underlying IO object.
 *
 * @tparam IOTmpl The type of the underlying IO object.
 */
template <typename IOTmpl>
class BasicIO {
public:
    /**
     * @brief Constructor that takes an Allocator pointer.
     *
     * The Allocator is used for memory allocation within the class.
     *
     * @param allocator A pointer to the Allocator object.
     */
    explicit BasicIO<IOTmpl>(Allocator* allocator) : allocator_(allocator){};

    /**
     * @brief Virtual destructor to ensure proper cleanup in derived classes.
     */
    virtual ~BasicIO() = default;

    /**
     * @brief Writes data to the IO object at a specified offset.
     *
     * If the IO object has a WriteImpl method, it is called.
     * Otherwise, a runtime error is thrown.
     *
     * @param data A pointer to the data to be written.
     * @param size The size of the data to be written.
     * @param offset The offset at which to write the data.
     */
    inline void
    Write(const uint8_t* data, uint64_t size, uint64_t offset) {
        static_assert(has_WriteImpl<IOTmpl>::value);
        cast().WriteImpl(data, size, offset);
    }

    /**
     * @brief Reads data from the IO object at a specified offset.
     *
     * If the IO object has a ReadImpl method, it is called.
     * Otherwise, a runtime error is thrown.
     *
     * @param size The size of the data to be read.
     * @param offset The offset at which to read the data.
     * @param data A pointer to the buffer where the read data will be stored.
     * @return True if the read operation was successful, false otherwise.
     */
    inline bool
    Read(uint64_t size, uint64_t offset, uint8_t* data) const {
        static_assert(has_ReadImpl<IOTmpl>::value);
        return cast().ReadImpl(size, offset, data);
    }

    /**
     * @brief Reads data directly from the IO object at a specified offset.
     *
     * If the IO object has a DirectReadImpl method, it is called.
     * Otherwise, a runtime error is thrown.
     *
     * @param size The size of the data to be read.
     * @param offset The offset at which to read the data.
     * @param need_release A reference to a boolean indicating whether the returned data needs to be released.
     * @return A pointer to the read data.
     */
    [[nodiscard]] inline const uint8_t*
    Read(uint64_t size, uint64_t offset, bool& need_release) const {
        static_assert(has_DirectReadImpl<IOTmpl>::value);
        return cast().DirectReadImpl(size, offset, need_release);  // TODO(LHT129): use IOReadObject
    }

    /**
     * @brief Reads multiple blocks of data from the IO object.
     *
     * If the IO object has a MultiReadImpl method, it is called.
     * Otherwise, a runtime error is thrown.
     *
     * @param datas An array of pointers to the buffers where the read data will be stored.
     * @param sizes An array of sizes for each block of data to be read.
     * @param offsets An array of offsets for each block of data to be read.
     * @param count The number of blocks of data to be read.
     * @return True if the read operation was successful, false otherwise.
     */
    inline bool
    MultiRead(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
        static_assert(has_MultiReadImpl<IOTmpl>::value);
        return cast().MultiReadImpl(datas, sizes, offsets, count);
    }

    /**
     * @brief Prefetches data from the IO object at a specified offset.
     *
     * If the IO object has a PrefetchImpl method, it is called.
     * Otherwise, a runtime error is thrown.
     *
     * @param offset The offset at which to prefetch the data.
     * @param cache_line The size of the cache line to prefetch.
     */
    inline void
    Prefetch(uint64_t offset, uint64_t cache_line = 64) {
        static_assert(has_PrefetchImpl<IOTmpl>::value);
        return cast().PrefetchImpl(offset, cache_line);
    }

    /**
     * @brief Serializes the IO object to a StreamWriter.
     *
     * @param writer The StreamWriter to which the IO object will be serialized.
     */
    inline void
    Serialize(StreamWriter& writer) {
        StreamWriter::WriteObj(writer, this->size_);
        ByteBuffer buffer(SERIALIZE_BUFFER_SIZE, this->allocator_);
        uint64_t offset = 0;
        while (offset < this->size_) {
            auto cur_size = std::min(SERIALIZE_BUFFER_SIZE, this->size_ - offset);
            this->Read(cur_size, offset, buffer.data);
            writer.Write(reinterpret_cast<const char*>(buffer.data), cur_size);
            offset += cur_size;
        }
    }

    /**
     * @brief Deserializes the IO object from a StreamReader.
     *
     * @param reader The StreamReader from which the IO object will be deserialized.
     */
    inline void
    Deserialize(StreamReader& reader) {
        uint64_t size;
        StreamReader::ReadObj(reader, size);
        ByteBuffer buffer(SERIALIZE_BUFFER_SIZE, this->allocator_);
        uint64_t offset = 0;
        while (offset < size) {
            auto cur_size = std::min(SERIALIZE_BUFFER_SIZE, size - offset);
            reader.Read(reinterpret_cast<char*>(buffer.data), cur_size);
            this->Write(buffer.data, cur_size, offset);
            offset += cur_size;
        }
    }

    /**
     * @brief Releases data previously read from the IO object.
     *
     * Sometimes, new buffer is malloced by read, so need release by use this method.
     *
     * If the IO object has a ReleaseImpl method, it is called.
     * Otherwise, a runtime error is thrown.
     *
     * @param data A pointer to the data to be released.
     */

    inline void
    Release(const uint8_t* data) const {
        static_assert(has_ReleaseImpl<IOTmpl>::value);
        return cast().ReleaseImpl(data);
    }

    /**
     * @brief Checks if the IO object is in-memory.
     *
     * This function checks if the IO object has an InMemoryImpl method.
     * If it does, it calls the method and returns the result.
     * Otherwise, it throws a runtime error.
     *
     * @return True if the IO object is in-memory, false otherwise.
     */
    inline bool
    InMemory() const {
        static_assert(has_InMemoryImpl<IOTmpl>::value);
        return cast().InMemoryImpl();
    }

public:
    /**
     * @brief The size of the IO object.
     */
    uint64_t size_{0};

protected:
    /**
     * @brief Checks if the given offset is valid.
     *
     * This function checks if the given offset is within the bounds of the IO object.
     * If the offset is valid, the function returns true. Otherwise, it returns false.
     *
     * @param size The offset to check.
     * @return True if the offset is valid, false otherwise.
     */
    [[nodiscard]] inline bool
    check_valid_offset(uint64_t size) const {
        // Check if the given offset is within the bounds of the IO object.
        return size <= this->size_;
    }

protected:
    /**
     * @brief A pointer to the Allocator object used for memory allocation.
     *
     * This pointer is used to allocate memory for the IO object.
     * It is a constant pointer, which means that it cannot be modified
     * after it is initialized.
     */
    Allocator* const allocator_;

private:
    /**
     * @brief Casts the current object to the underlying IO object type.
     *
     * @return A reference to the underlying IO object.
     */
    inline IOTmpl&
    cast() {
        return static_cast<IOTmpl&>(*this);
    }

    /**
     * @brief Casts the current object to the underlying IO object type (const version).
     *
     * @return A const reference to the underlying IO object.
     */
    inline const IOTmpl&
    cast() const {
        return static_cast<const IOTmpl&>(*this);
    }

    /**
     * @brief The size of the max buffer used for serialization.
     */
    constexpr static uint64_t SERIALIZE_BUFFER_SIZE = 1024 * 1024 * 2;

private:
    /**
     * @brief Generates a struct to check if a class has a member function with a specific signature.
     *
     */
    GENERATE_HAS_MEMBER_FUNCTION(WriteImpl,
                                 void,
                                 std::declval<const uint8_t*>(),
                                 std::declval<uint64_t>(),
                                 std::declval<uint64_t>())
    GENERATE_HAS_MEMBER_FUNCTION(ReadImpl,
                                 bool,
                                 std::declval<uint64_t>(),
                                 std::declval<uint64_t>(),
                                 std::declval<uint8_t*>())
    GENERATE_HAS_MEMBER_FUNCTION(DirectReadImpl,
                                 const uint8_t*,
                                 std::declval<uint64_t>(),
                                 std::declval<uint64_t>(),
                                 std::declval<bool&>())
    GENERATE_HAS_MEMBER_FUNCTION(MultiReadImpl,
                                 bool,
                                 std::declval<uint8_t*>(),
                                 std::declval<uint64_t*>(),
                                 std::declval<uint64_t*>(),
                                 std::declval<uint64_t>())
    GENERATE_HAS_MEMBER_FUNCTION(PrefetchImpl,
                                 void,
                                 std::declval<uint64_t>(),
                                 std::declval<uint64_t>())
    GENERATE_HAS_MEMBER_FUNCTION(ReleaseImpl, void, std::declval<const uint8_t*>())
    GENERATE_HAS_MEMBER_FUNCTION(InMemoryImpl, bool)
};
}  // namespace vsag
