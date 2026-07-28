
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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>

#include "io/common/io_parameter.h"
#include "io/read_cache/lru_page_cache.h"
#include "io/read_cache/page.h"
#include "io/read_cache/page_cache.h"
#include "io/read_cache/read_cache_parameter.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/byte_buffer.h"
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
    /// Checks if the IO object is in-memory.
    static constexpr bool InMemory = IOTmpl::InMemory;
    static constexpr bool SkipDeserialize = IOTmpl::SkipDeserialize;

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
        InvalidateCacheRange(size, offset);
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
        if (cache_ != nullptr) {
            return ReadCached(size, offset, data);
        }
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
        if (cache_ != nullptr) {
            need_release = false;
            if (size == 0 or not IsValidRange(size, offset)) {
                return nullptr;
            }
            auto* data = static_cast<uint8_t*>(allocator_->Allocate(size));
            if (data == nullptr) {
                return nullptr;
            }
            if (not ReadCached(size, offset, data)) {
                allocator_->Deallocate(data);
                return nullptr;
            }
            need_release = true;
            return data;
        }
        return cast().DirectReadImpl(size, offset, need_release);  // TODO(LHT129): use IOReadObject
    }

    /**
     * @brief Reads multiple blocks of data from the IO object.
     *
     * If the IO object has a MultiReadImpl method, it is called.
     * Otherwise, a runtime error is thrown.
     *
     * @param datas A pointer to a contiguous buffer where all read data will be stored sequentially.
     * @param sizes An array of sizes for each block of data to be read.
     * @param offsets An array of offsets for each block of data to be read.
     * @param count The number of blocks of data to be read.
     * @return True if the read operation was successful, false otherwise.
     */
    inline bool
    MultiRead(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
        static_assert(has_MultiReadImpl<IOTmpl>::value);
        if (cache_ != nullptr) {
            for (uint64_t i = 0; i < count; ++i) {
                if (not ReadCached(sizes[i], offsets[i], datas)) {
                    return false;
                }
                datas += sizes[i];
            }
            return true;
        }
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
        if constexpr (has_PrefetchImpl<IOTmpl>::value) {
            return cast().PrefetchImpl(offset, cache_line);
        }
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
        uint64_t size = 0;
        StreamReader::ReadObj(reader, size);
        this->start_ = reader.GetCursor();
        if constexpr (SkipDeserialize) {
            reader.Seek(reader.GetCursor() + size);
            this->size_ = std::max(this->size_, size);
        } else {
            ByteBuffer buffer(SERIALIZE_BUFFER_SIZE, this->allocator_);
            uint64_t offset = 0;
            while (offset < size) {
                auto cur_size = std::min(SERIALIZE_BUFFER_SIZE, size - offset);
                reader.Read(reinterpret_cast<char*>(buffer.data), cur_size);
                this->Write(buffer.data, cur_size, offset);
                offset += cur_size;
            }
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
        if (cache_ != nullptr) {
            allocator_->Deallocate(const_cast<uint8_t*>(data));
            return;
        }
        if constexpr (has_ReleaseImpl<IOTmpl>::value) {
            return cast().ReleaseImpl(data);
        }
    }

    /**
     * @brief Initializes the IO object with the given IO parameters.
     *
     * This function checks if the IO object has an InitIOImpl method.
     * If it does, it calls the method with the provided IO parameters.
     * Otherwise, it throws a runtime error.
     *
     * @param io_param A pointer to the IO parameters used for initialization.
     */
    inline void
    InitIO(const IOParamPtr& io_param) {
        EnableReadCache(io_param);
        if constexpr (has_InitIOImpl<IOTmpl>::value) {
            return cast().InitIOImpl(UnwrapReadCacheParam(io_param));
        }
    }

    inline void
    Resize(uint64_t size) {
        if constexpr (has_ResizeImpl<IOTmpl>::value) {
            cast().ResizeImpl(size);
        } else {
            if (size <= this->size_) {
                return;
            }
            ByteBuffer buffer(SERIALIZE_BUFFER_SIZE, this->allocator_);
            memset(buffer.data, 0, SERIALIZE_BUFFER_SIZE);
            uint64_t offset = this->size_;
            while (offset < size) {
                auto cur_size = std::min(SERIALIZE_BUFFER_SIZE, size - offset);
                this->Write(buffer.data, cur_size, offset);
                offset += cur_size;
            }
        }
        ClearCache();
    }

    inline void
    Shrink(uint64_t size) {
        if constexpr (has_ShrinkImpl<IOTmpl>::value) {
            cast().ShrinkImpl(size);
        } else {
            if (size <= this->size_) {
                this->size_ = size;
            }
        }
        ClearCache();
    }

    inline int64_t
    GetMemoryUsage() const {
        return this->size_;
    }

    void
    EnableReadCache(const IOParamPtr& io_param) {
        auto cache_param = std::dynamic_pointer_cast<ReadCacheParameter>(io_param);
        if (cache_param == nullptr) {
            return;
        }
        auto page_count = cache_param->total_cache_size_ / Page::DEFAULT_PAGE_SIZE;
        if (page_count == 0) {
            cache_.reset();
            return;
        }
        cache_ = std::make_unique<LRUPageCache>(page_count);
    }

public:
    /**
     * @brief The size of the IO object.
     */
    uint64_t size_{0};
    uint64_t start_{0};

protected:
    /**
     * @brief Protected non-virtual destructor.
     *
     * BasicIO uses CRTP for static polymorphism, so dynamic dispatch is never needed.
     * The destructor is non-virtual to avoid introducing a vptr.
     *
     * The destructor is protected (not public) to prevent deletion through a raw
     * BasicIO pointer at compile time. Note that std::shared_ptr<BasicIO<IOTmpl>>
     * (used in some datacells) remains safe because the deleter is bound to the
     * concrete IOTmpl type at construction time (e.g. via std::make_shared<IOTmpl>).
     */
    ~BasicIO() = default;

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

    [[nodiscard]] bool
    IsValidRange(uint64_t size, uint64_t offset) const {
        return offset <= size_ and size <= size_ - offset;
    }

    bool
    ReadCached(uint64_t size, uint64_t offset, uint8_t* data) const {
        if (not IsValidRange(size, offset)) {
            return false;
        }
        uint64_t copied = 0;
        while (copied < size) {
            uint64_t current_offset = offset + copied;
            uint64_t page_id = current_offset / Page::DEFAULT_PAGE_SIZE;
            uint64_t page_offset = current_offset % Page::DEFAULT_PAGE_SIZE;
            uint64_t copy_size = std::min(size - copied, Page::DEFAULT_PAGE_SIZE - page_offset);
            auto page = GetOrLoadPage(page_id);
            if (page == nullptr) {
                return false;
            }
            std::memcpy(data + copied, page->Data() + page_offset, copy_size);
            copied += copy_size;
        }
        return true;
    }

    PagePtr
    GetOrLoadPage(uint64_t page_id) const {
        if (page_id > UINT64_MAX / Page::DEFAULT_PAGE_SIZE) {
            return nullptr;
        }
        uint64_t offset = page_id * Page::DEFAULT_PAGE_SIZE;
        if (offset >= size_) {
            return nullptr;
        }
        std::scoped_lock<std::mutex> lock(cache_mutex_);
        auto page = cache_->Get(page_id);
        if (page != nullptr) {
            return page;
        }
        auto new_page = std::make_shared<Page>(allocator_);
        if (new_page->Data() == nullptr) {
            return nullptr;
        }
        uint64_t read_size = std::min(Page::DEFAULT_PAGE_SIZE, size_ - offset);
        if (not cast().ReadImpl(read_size, offset, new_page->Data())) {
            return nullptr;
        }
        return cache_->Insert(page_id, std::move(new_page));
    }

    void
    InvalidateCacheRange(uint64_t size, uint64_t offset) {
        if (cache_ == nullptr or size == 0) {
            return;
        }
        std::scoped_lock<std::mutex> lock(cache_mutex_);
        if (offset > UINT64_MAX - (size - 1)) {
            cache_->Clear();
            return;
        }
        uint64_t first_page = offset / Page::DEFAULT_PAGE_SIZE;
        uint64_t last_page = (offset + size - 1) / Page::DEFAULT_PAGE_SIZE;
        for (uint64_t page_id = first_page; page_id <= last_page; ++page_id) {
            cache_->Remove(page_id);
        }
    }

    void
    ClearCache() {
        if (cache_ != nullptr) {
            std::scoped_lock<std::mutex> lock(cache_mutex_);
            cache_->Clear();
        }
    }

    /**
     * @brief The size of the max buffer used for serialization.
     */
    static constexpr uint64_t SERIALIZE_BUFFER_SIZE = 1024 * 1024 * 2;

    mutable std::mutex cache_mutex_;
    mutable std::unique_ptr<PageCache> cache_;

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
    GENERATE_HAS_MEMBER_FUNCTION(InitIOImpl, void, std::declval<const IOParamPtr&>())
    GENERATE_HAS_MEMBER_FUNCTION(ResizeImpl, void, std::declval<uint64_t>())
    GENERATE_HAS_MEMBER_FUNCTION(ShrinkImpl, void, std::declval<uint64_t>())
};
}  // namespace vsag
