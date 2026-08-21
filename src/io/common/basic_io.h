
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
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

#include "hash_types.h"
#include "io/common/io_parameter.h"
#include "io/read_cache/lru_page_cache.h"
#include "io/read_cache/page.h"
#include "io/read_cache/page_cache.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/byte_buffer.h"
#include "utils/function_exists_check.h"

namespace vsag {

template <typename T, typename = void>
struct SupportsZeroSizeResize : std::true_type {};

template <typename T>
struct SupportsZeroSizeResize<T, std::void_t<decltype(T::SupportsZeroSizeResize)>>
    : std::bool_constant<T::SupportsZeroSizeResize> {};

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
private:
    using PageIdList = std::vector<uint64_t, AllocatorWrapper<uint64_t>>;

    class ReadCacheSnapshot {
    public:
        std::shared_ptr<PageCache> cache;
        uint64_t page_id_base{0};
    };

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
    explicit BasicIO<IOTmpl>(Allocator* allocator)
        : allocator_(allocator), cached_direct_reads_(allocator){};

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
        static_assert(HasWriteImpl<IOTmpl>::value);
        cast().WriteImpl(data, size, offset);
        if constexpr (not InMemory) {
            InvalidateCacheRange(size, offset);
        }
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
        static_assert(HasReadImpl<IOTmpl>::value);
        if constexpr (not InMemory) {
            const auto cache = GetReadCacheSnapshot();
            if (cache.cache != nullptr) {
                return ReadCached(size, offset, data, cache);
            }
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
        static_assert(HasDirectReadImpl<IOTmpl>::value);
        if constexpr (not InMemory) {
            const auto cache = GetReadCacheSnapshot();
            if (cache.cache != nullptr) {
                need_release = false;
                if (size == 0 or not IsValidRange(size, offset)) {
                    return nullptr;
                }
                auto* data = static_cast<uint8_t*>(allocator_->Allocate(size));
                if (data == nullptr) {
                    return nullptr;
                }
                if (not ReadCached(size, offset, data, cache)) {
                    allocator_->Deallocate(data);
                    return nullptr;
                }
                need_release = true;
                try {
                    std::scoped_lock<std::mutex> lock(cached_direct_reads_mutex_);
                    cached_direct_reads_.emplace(data);
                } catch (...) {
                    need_release = false;
                    allocator_->Deallocate(data);
                    throw;
                }
                return data;
            }
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
        static_assert(HasMultiReadImpl<IOTmpl>::value);
        if constexpr (not InMemory) {
            const auto cache = GetReadCacheSnapshot();
            if (cache.cache != nullptr) {
                if (count == 0) {
                    return true;
                }
                if (sizes == nullptr or offsets == nullptr) {
                    return false;
                }
                bool has_data = false;
                for (uint64_t i = 0; i < count; ++i) {
                    if (not IsValidRange(sizes[i], offsets[i])) {
                        return false;
                    }
                    has_data = has_data or sizes[i] > 0;
                }
                if (not has_data) {
                    return true;
                }
                if (datas == nullptr) {
                    return false;
                }
                class PageCopy {
                public:
                    uint64_t page_id;
                    uint64_t page_offset;
                    uint64_t output_offset;
                    uint64_t size;
                };
                using PageCopyList = std::vector<PageCopy, AllocatorWrapper<PageCopy>>;
                PageIdList page_ids(allocator_);
                PageCopyList page_copies(allocator_);
                UnorderedSet<uint64_t> seen_page_ids(allocator_);
                uint64_t output_offset = 0;
                for (uint64_t i = 0; i < count; ++i) {
                    if (sizes[i] > UINT64_MAX - output_offset) {
                        return false;
                    }
                    if (sizes[i] == 0) {
                        continue;
                    }
                    uint64_t copied = 0;
                    while (copied < sizes[i]) {
                        const uint64_t current_offset = offsets[i] + copied;
                        const uint64_t page_id = current_offset / Page::DEFAULT_PAGE_SIZE;
                        const uint64_t page_offset = current_offset % Page::DEFAULT_PAGE_SIZE;
                        const uint64_t copy_size =
                            std::min(sizes[i] - copied, Page::DEFAULT_PAGE_SIZE - page_offset);
                        if (seen_page_ids.emplace(page_id).second) {
                            page_ids.emplace_back(page_id);
                        }
                        page_copies.emplace_back(
                            PageCopy{page_id, page_offset, output_offset + copied, copy_size});
                        copied += copy_size;
                    }
                    output_offset += sizes[i];
                }

                std::sort(page_ids.begin(), page_ids.end());
                std::sort(
                    page_copies.begin(), page_copies.end(), [](const auto& lhs, const auto& rhs) {
                        return lhs.page_id < rhs.page_id;
                    });
                // Each miss temporarily needs both a Page and contiguous MultiRead storage.
                // Keep that temporary working set near 8 MiB while preserving batched backend IO.
                constexpr uint64_t MAX_BATCH_READ_TEMP_BYTES = 8ULL * 1024ULL * 1024ULL;
                const uint64_t max_batch_pages = std::max<uint64_t>(
                    1, MAX_BATCH_READ_TEMP_BYTES / (2 * Page::DEFAULT_PAGE_SIZE));
                UnorderedMap<uint64_t, PagePtr> pages(allocator_);
                uint64_t copy_index = 0;
                for (uint64_t batch_begin = 0; batch_begin < page_ids.size();
                     batch_begin += max_batch_pages) {
                    const uint64_t batch_end =
                        std::min<uint64_t>(page_ids.size(), batch_begin + max_batch_pages);
                    PageIdList batch_page_ids(allocator_);
                    batch_page_ids.reserve(batch_end - batch_begin);
                    for (uint64_t i = batch_begin; i < batch_end; ++i) {
                        batch_page_ids.emplace_back(page_ids[i]);
                    }
                    pages.clear();
                    if (not LoadCachedPages(batch_page_ids, pages, true, cache)) {
                        return false;
                    }
                    const uint64_t last_page_id = page_ids[batch_end - 1];
                    while (copy_index < page_copies.size() and
                           page_copies[copy_index].page_id <= last_page_id) {
                        const auto& copy = page_copies[copy_index];
                        const auto iter = pages.find(copy.page_id);
                        if (iter == pages.end() or iter->second == nullptr) {
                            return false;
                        }
                        std::memcpy(datas + copy.output_offset,
                                    iter->second->Data() + copy.page_offset,
                                    copy.size);
                        ++copy_index;
                    }
                }
                return copy_index == page_copies.size();
            }
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
        if constexpr (HasPrefetchImpl<IOTmpl>::value) {
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
        const auto size = this->size_.load(std::memory_order_acquire);
        StreamWriter::WriteObj(writer, size);
        ByteBuffer buffer(SERIALIZE_BUFFER_SIZE, this->allocator_);
        uint64_t offset = 0;
        while (offset < size) {
            auto cur_size = std::min(SERIALIZE_BUFFER_SIZE, size - offset);
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
        has_deserialized_ = true;
        this->start_ = reader.GetCursor();
        if constexpr (SkipDeserialize) {
            reader.Seek(reader.GetCursor() + size);
            this->PublishSize(size);
        } else {
            // Reset the logical and physical extent so a shorter deserialization
            // cannot retain stale bytes from a previously opened file.
            if constexpr (HasResizeImpl<IOTmpl>::value) {
                if (size > 0 or SupportsZeroSizeResize<IOTmpl>::value) {
                    Resize(size);
                } else {
                    this->size_ = 0;
                    ClearCache();
                }
            } else {
                this->size_ = 0;
                ClearCache();
            }
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
        if constexpr (not InMemory) {
            std::scoped_lock<std::mutex> lock(cached_direct_reads_mutex_);
            if (cached_direct_reads_.erase(data) != 0) {
                allocator_->Deallocate(const_cast<uint8_t*>(data));
                return;
            }
        }
        if constexpr (HasReleaseImpl<IOTmpl>::value) {
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
        if constexpr (not InMemory) {
            const auto cache = GetReadCacheSnapshot();
            if (cache.cache == nullptr) {
                EnableReadCache(io_param);
            } else if (io_param != nullptr and io_param->enable_read_cache_) {
                EnableReadCache(io_param);
            } else {
                ClearCache();
            }
        }
        if constexpr (HasInitIOImpl<IOTmpl>::value) {
            return cast().InitIOImpl(io_param);
        }
    }

    inline void
    Resize(uint64_t size) {
        if constexpr (HasResizeImpl<IOTmpl>::value) {
            cast().ResizeImpl(size);
        } else {
            const auto current_size = this->size_.load(std::memory_order_acquire);
            if (size <= current_size) {
                return;
            }
            ByteBuffer buffer(SERIALIZE_BUFFER_SIZE, this->allocator_);
            memset(buffer.data, 0, SERIALIZE_BUFFER_SIZE);
            uint64_t offset = current_size;
            while (offset < size) {
                auto cur_size = std::min(SERIALIZE_BUFFER_SIZE, size - offset);
                this->Write(buffer.data, cur_size, offset);
                offset += cur_size;
            }
        }
        if constexpr (not InMemory) {
            ClearCache();
        }
    }

    /**
     * @brief Grow the storage to size without zero-filling the freshly
     * allocated region, for callers that guarantee they overwrite the whole
     * [0, size) range afterwards (e.g. parallel deserialization).
     *
     * Falls back to Resize when the concrete IO has no dedicated
     * implementation, so it is always safe to call.
     */
    inline void
    ResizeForOverwrite(uint64_t size) {
        if constexpr (has_ResizeForOverwriteImpl<IOTmpl>::value) {
            cast().ResizeForOverwriteImpl(size);
            if constexpr (not InMemory) {
                ClearCache();
            }
        } else {
            Resize(size);
        }
    }

    inline void
    Shrink(uint64_t size) {
        if constexpr (HasShrinkImpl<IOTmpl>::value) {
            cast().ShrinkImpl(size);
        } else {
            if (size <= this->size_.load(std::memory_order_acquire)) {
                this->size_.store(size, std::memory_order_release);
            }
        }
        if constexpr (not InMemory) {
            ClearCache();
        }
    }

    inline int64_t
    GetMemoryUsage() const {
        if constexpr (HasGetMemoryUsageImpl<IOTmpl>::value) {
            return cast().GetMemoryUsageImpl();
        }
        return this->size_.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool
    HasDeserialized() const {
        return has_deserialized_;
    }

    void
    EnableReadCache(const IOParamPtr& io_param) {
        if constexpr (not InMemory) {
            std::scoped_lock<std::mutex> lock(cache_mutex_);
            if (io_param == nullptr or not io_param->enable_read_cache_) {
                cache_.reset();
                cache_page_id_base_ = 0;
                return;
            }
            auto page_count = io_param->read_cache_total_size_ / Page::DEFAULT_PAGE_SIZE;
            if (page_count == 0) {
                cache_.reset();
                cache_page_id_base_ = 0;
                return;
            }
            cache_ = std::make_shared<LRUPageCache>(page_count);
            cache_page_id_base_ = 0;
        }
    }

    /**
     * @brief Replaces this IO's read cache with a shared cache.
     *
     * The page id base gives each logical IO a disjoint key range in the shared cache.
     * Passing nullptr detaches the current cache without clearing caches shared by other IOs.
     * This is intended for read-only logical IO views initialized before queries start.
     */
    void
    SetReadCache(const std::shared_ptr<PageCache>& cache, uint64_t page_id_base = 0) {
        if constexpr (not InMemory) {
            std::scoped_lock<std::mutex> lock(cache_mutex_);
            cache_ = cache;
            cache_page_id_base_ = page_id_base;
        }
    }

public:
    /**
     * @brief The size of the IO object.
     */
    // Writers publish newly initialized storage through size_; readers acquire that publication
    // before validating offsets. Some in-memory IO implementations append concurrently with reads.
    std::atomic<uint64_t> size_{0};
    uint64_t start_{0};

protected:
    void
    PublishSize(uint64_t size) {
        auto current = size_.load(std::memory_order_relaxed);
        while (current < size and
               not size_.compare_exchange_weak(
                   current, size, std::memory_order_release, std::memory_order_relaxed)) {
        }
    }

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
        return size <= this->size_.load(std::memory_order_acquire);
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
        const auto total_size = size_.load(std::memory_order_acquire);
        return offset <= total_size and size <= total_size - offset;
    }

    ReadCacheSnapshot
    GetReadCacheSnapshot() const {
        std::scoped_lock<std::mutex> lock(cache_mutex_);
        // An in-flight operation intentionally keeps using the cache generation it started with.
        return {cache_, cache_page_id_base_};
    }

    bool
    ReadCached(uint64_t size,
               uint64_t offset,
               uint8_t* data,
               const ReadCacheSnapshot& cache) const {
        if (not IsValidRange(size, offset)) {
            return false;
        }
        uint64_t copied = 0;
        while (copied < size) {
            uint64_t current_offset = offset + copied;
            uint64_t page_id = current_offset / Page::DEFAULT_PAGE_SIZE;
            uint64_t page_offset = current_offset % Page::DEFAULT_PAGE_SIZE;
            uint64_t copy_size = std::min(size - copied, Page::DEFAULT_PAGE_SIZE - page_offset);
            auto page = GetOrLoadPage(page_id, cache);
            if (page == nullptr) {
                return false;
            }
            std::memcpy(data + copied, page->Data() + page_offset, copy_size);
            copied += copy_size;
        }
        return true;
    }

    PagePtr
    GetOrLoadPage(uint64_t page_id, const ReadCacheSnapshot& cache) const {
        if (page_id > UINT64_MAX / Page::DEFAULT_PAGE_SIZE) {
            return nullptr;
        }
        const uint64_t offset = page_id * Page::DEFAULT_PAGE_SIZE;
        const auto total_size = size_.load(std::memory_order_acquire);
        if (offset >= total_size) {
            return nullptr;
        }
        if (cache.cache == nullptr or page_id > UINT64_MAX - cache.page_id_base) {
            return nullptr;
        }
        const uint64_t cache_page_id = cache.page_id_base + page_id;
        while (true) {
            auto result = cache.cache->Acquire(cache_page_id);
            if (result.page != nullptr) {
                return result.page;
            }
            if (not result.should_load) {
                auto page = cache.cache->Wait(result.handle);
                if (page != nullptr) {
                    return page;
                }
                if (cache.cache->IsStale(result.handle)) {
                    continue;
                }
                return nullptr;
            }

            PagePtr page = nullptr;
            bool success = false;
            try {
                page = std::make_shared<Page>(allocator_);
                success = page->Data() != nullptr;
                if (success) {
                    const uint64_t read_size =
                        std::min(Page::DEFAULT_PAGE_SIZE, total_size - offset);
                    success = cast().ReadImpl(read_size, offset, page->Data());
                }
            } catch (...) {
                cache.cache->Complete(cache_page_id, result.handle, nullptr, false);
                throw;
            }
            page = cache.cache->Complete(
                cache_page_id, result.handle, success ? std::move(page) : nullptr, success);
            if (page != nullptr) {
                return page;
            }
            if (cache.cache->IsStale(result.handle)) {
                continue;
            }
            return nullptr;
        }
    }

    bool
    LoadCachedPages(const PageIdList& page_ids,
                    UnorderedMap<uint64_t, PagePtr>& pages,
                    bool batch_read,
                    const ReadCacheSnapshot& cache) const {
        PageIdList ordered_page_ids(page_ids.begin(), page_ids.end(), allocator_);
        std::sort(ordered_page_ids.begin(), ordered_page_ids.end());

        while (true) {
            pages.clear();
            bool stale = false;
            const bool loaded =
                LoadCachedPagesOnce(ordered_page_ids, pages, batch_read, stale, cache);
            if (loaded and not stale) {
                return true;
            }
            if (not stale) {
                return false;
            }
        }
    }

    bool
    LoadCachedPagesOnce(const PageIdList& page_ids,
                        UnorderedMap<uint64_t, PagePtr>& pages,
                        bool batch_read,
                        bool& stale,
                        const ReadCacheSnapshot& cache) const {
        stale = false;
        if (cache.cache == nullptr or
            std::any_of(page_ids.begin(), page_ids.end(), [&cache](uint64_t page_id) {
                return page_id > UINT64_MAX - cache.page_id_base;
            })) {
            return false;
        }
        using PageLoadList =
            std::vector<std::pair<uint64_t, PageCache::LoadHandle>,
                        AllocatorWrapper<std::pair<uint64_t, PageCache::LoadHandle>>>;
        PageLoadList owned_loads(allocator_);
        pages.reserve(page_ids.size());
        owned_loads.reserve(page_ids.size());
        const auto abandon_owned_loads = [&cache, &owned_loads]() {
            for (const auto& [page_id, handle] : owned_loads) {
                cache.cache->Complete(cache.page_id_base + page_id, handle, nullptr, false);
            }
            owned_loads.clear();
        };
        try {
            for (const uint64_t page_id : page_ids) {
                auto result = cache.cache->Acquire(cache.page_id_base + page_id);
                if (result.page != nullptr) {
                    pages.emplace(page_id, std::move(result.page));
                } else if (result.should_load) {
                    owned_loads.emplace_back(page_id, std::move(result.handle));
                } else {
                    auto page = cache.cache->Wait(result.handle);
                    stale = stale or cache.cache->IsStale(result.handle);
                    if (page == nullptr or stale) {
                        abandon_owned_loads();
                        return false;
                    }
                    pages.emplace(page_id, std::move(page));
                }
            }
        } catch (...) {
            abandon_owned_loads();
            throw;
        }

        std::vector<PagePtr, AllocatorWrapper<PagePtr>> loaded_pages(allocator_);
        std::vector<uint64_t, AllocatorWrapper<uint64_t>> read_sizes(allocator_);
        std::vector<uint64_t, AllocatorWrapper<uint64_t>> read_offsets(allocator_);
        std::vector<uint8_t, AllocatorWrapper<uint8_t>> read_data(allocator_);
        bool success = true;
        try {
            uint64_t total_size = 0;
            loaded_pages.reserve(owned_loads.size());
            read_sizes.reserve(owned_loads.size());
            read_offsets.reserve(owned_loads.size());
            const auto published_size = size_.load(std::memory_order_acquire);
            for (const auto& [page_id, _] : owned_loads) {
                if (page_id > UINT64_MAX / Page::DEFAULT_PAGE_SIZE) {
                    success = false;
                    break;
                }
                const uint64_t offset = page_id * Page::DEFAULT_PAGE_SIZE;
                if (offset >= published_size) {
                    success = false;
                    break;
                }
                const uint64_t read_size =
                    std::min(Page::DEFAULT_PAGE_SIZE, published_size - offset);
                if (read_size > UINT64_MAX - total_size) {
                    success = false;
                    break;
                }
                auto page = std::make_shared<Page>(allocator_);
                if (page->Data() == nullptr) {
                    success = false;
                    break;
                }
                loaded_pages.emplace_back(std::move(page));
                read_sizes.emplace_back(read_size);
                read_offsets.emplace_back(offset);
                total_size += read_size;
            }
            if (success and not owned_loads.empty()) {
                if (batch_read) {
                    read_data.resize(total_size);
                    success = cast().MultiReadImpl(read_data.data(),
                                                   read_sizes.data(),
                                                   read_offsets.data(),
                                                   owned_loads.size());
                    uint64_t copied = 0;
                    for (uint64_t i = 0; success and i < loaded_pages.size(); ++i) {
                        std::memcpy(
                            loaded_pages[i]->Data(), read_data.data() + copied, read_sizes[i]);
                        copied += read_sizes[i];
                    }
                } else {
                    for (uint64_t i = 0; success and i < loaded_pages.size(); ++i) {
                        success = cast().ReadImpl(
                            read_sizes[i], read_offsets[i], loaded_pages[i]->Data());
                    }
                }
            }
        } catch (...) {
            for (const auto& [page_id, handle] : owned_loads) {
                cache.cache->Complete(cache.page_id_base + page_id, handle, nullptr, false);
            }
            throw;
        }

        for (uint64_t i = 0; i < owned_loads.size(); ++i) {
            const auto& [page_id, handle] = owned_loads[i];
            PagePtr loaded_page = nullptr;
            if (success) {
                loaded_page = std::move(loaded_pages[i]);
            }
            try {
                auto page = cache.cache->Complete(
                    cache.page_id_base + page_id, handle, std::move(loaded_page), success);
                stale = stale or cache.cache->IsStale(handle);
                if (page != nullptr) {
                    pages.emplace(page_id, std::move(page));
                }
            } catch (...) {
                for (uint64_t j = i + 1; j < owned_loads.size(); ++j) {
                    const auto& [remaining_page_id, remaining_handle] = owned_loads[j];
                    cache.cache->Complete(
                        cache.page_id_base + remaining_page_id, remaining_handle, nullptr, false);
                }
                throw;
            }
        }
        if (not success) {
            return false;
        }

        return std::all_of(page_ids.begin(), page_ids.end(), [&pages](uint64_t page_id) {
            const auto iter = pages.find(page_id);
            return iter != pages.end() and iter->second != nullptr;
        });
    }

    void
    InvalidateCacheRange(uint64_t size, uint64_t offset) {
        if (size == 0) {
            return;
        }
        std::shared_ptr<PageCache> cache;
        uint64_t cache_page_id_base;
        {
            std::scoped_lock<std::mutex> lock(cache_mutex_);
            cache = cache_;
            cache_page_id_base = cache_page_id_base_;
        }
        if (cache == nullptr) {
            return;
        }
        if (offset > UINT64_MAX - (size - 1)) {
            cache->Clear();
            return;
        }
        uint64_t first_page = offset / Page::DEFAULT_PAGE_SIZE;
        uint64_t last_page = (offset + size - 1) / Page::DEFAULT_PAGE_SIZE;
        if (last_page > UINT64_MAX - cache_page_id_base) {
            cache->Clear();
            return;
        }
        for (uint64_t page_id = first_page;; ++page_id) {
            cache->Remove(cache_page_id_base + page_id);
            if (page_id == last_page) {
                break;
            }
        }
    }

    void
    ClearCache() {
        std::shared_ptr<PageCache> cache;
        {
            std::scoped_lock<std::mutex> lock(cache_mutex_);
            cache = cache_;
        }
        if (cache != nullptr) {
            cache->Clear();
        }
    }

    /**
     * @brief The size of the max buffer used for serialization.
     */
    static constexpr uint64_t SERIALIZE_BUFFER_SIZE = 1024 * 1024 * 2;

    mutable std::mutex cache_mutex_;
    mutable std::shared_ptr<PageCache> cache_;
    uint64_t cache_page_id_base_{0};
    mutable std::mutex cached_direct_reads_mutex_;
    // Outstanding cached direct reads; each pointer must be released exactly once.
    mutable UnorderedSet<const uint8_t*> cached_direct_reads_;
    bool has_deserialized_{false};

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
    GENERATE_HAS_MEMBER_FUNCTION(ResizeForOverwriteImpl, void, std::declval<uint64_t>())
    GENERATE_HAS_MEMBER_FUNCTION(ShrinkImpl, void, std::declval<uint64_t>())
    GENERATE_HAS_MEMBER_FUNCTION(GetMemoryUsageImpl, int64_t)
};
}  // namespace vsag
