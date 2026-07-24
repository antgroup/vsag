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
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_set>

#include "index_common_param.h"
#include "inner_string_params.h"
#include "io/cache_io/cache_io_parameter.h"
#include "io/cache_io/lru_page_cache.h"
#include "io/cache_io/page.h"
#include "io/cache_io/page_cache.h"
#include "io/cache_io/sharded_lru_page_cache.h"
#include "io/common/basic_io.h"
#include "io/common/io_parameter.h"
#include "io/reader_io/reader_io_parameter.h"

namespace vsag {

template <typename IOTmpl>
class CacheIOTest;

/**
 * @brief Read-cache IO layer wrapping an inner IO implementation.
 *
 * CacheIO provides a page-based read cache in front of any disk-backed IO.
 * All reads go through the cache; writes passthrough to the inner IO directly.
 * The workload is assumed to be non-overlapping between reads and writes.
 *
 * @tparam IOTmpl The underlying disk IO implementation type.
 */
template <typename IOTmpl>
class CacheIO : public BasicIO<CacheIO<IOTmpl>> {
public:
    static constexpr bool InMemory = true;
    static constexpr bool UseNonContinuous = not IOTmpl::InMemory;
    static constexpr bool SkipDeserialize = IOTmpl::SkipDeserialize;

    template <typename... Args>
    CacheIO(const CacheIOParameter& param, Allocator* allocator, Args&&... args)
        : BasicIO<CacheIO<IOTmpl>>(allocator),
          cache_(std::make_unique<ShardedLRUPageCache>(param.total_cache_size_ /
                                                       Page::DEFAULT_PAGE_SIZE)),
          inner_io_(std::make_unique<IOTmpl>(std::forward<Args>(args)...)),
          cache_max_pages_(param.total_cache_size_ / Page::DEFAULT_PAGE_SIZE) {
        this->size_ = inner_io_->size_;
    }

    CacheIO(const IOParamPtr& io_param, const IndexCommonParam& common_param)
        : BasicIO<CacheIO<IOTmpl>>(common_param.allocator_.get()) {
        auto cache_param = std::dynamic_pointer_cast<CacheIOParameter>(io_param);
        if (cache_param == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "CacheIO requires CacheIOParameter");
        }
        cache_max_pages_ = cache_param->total_cache_size_ / Page::DEFAULT_PAGE_SIZE;
        cache_ = std::make_unique<ShardedLRUPageCache>(cache_max_pages_);
        auto inner_io_param = MakeInnerIOParam(cache_param);
        inner_io_ = std::make_unique<IOTmpl>(inner_io_param, common_param);
        this->size_ = inner_io_->size_;
    }

    ~CacheIO() = default;

    void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset) {
        std::scoped_lock<std::mutex> lock(io_mutex_);
        inner_io_->WriteImpl(data, size, offset);
        if (size > 0) {
            InvalidateRange(size, offset);
        }
        if (size > UINT64_MAX - offset) {
            this->size_ = UINT64_MAX;
        } else if (offset + size > this->size_) {
            this->size_ = offset + size;
        }
    }

    bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const {
        if (not is_valid_range(size, offset)) {
            return false;
        }
        if (cache_max_pages_ == 0) {
            return inner_io_->ReadImpl(size, offset, data);
        }
        uint64_t page_size = Page::DEFAULT_PAGE_SIZE;
        uint64_t cur_size = 0;
        while (cur_size < size) {
            uint64_t cur_offset = offset + cur_size;
            uint64_t page_id = cur_offset / page_size;
            uint64_t page_offset = cur_offset % page_size;
            uint64_t copy_size = std::min(size - cur_size, page_size - page_offset);

            PagePtr page = get_or_load_page(page_id);
            if (page == nullptr) {
                return false;
            }
            std::memcpy(data + cur_size, page->Data() + page_offset, copy_size);
            cur_size += copy_size;
        }
        return true;
    }

    [[nodiscard]] const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const {
        need_release = false;
        if (size == 0 or not is_valid_range(size, offset)) {
            return nullptr;
        }
        auto* data = reinterpret_cast<uint8_t*>(this->allocator_->Allocate(size));
        if (data == nullptr) {
            return nullptr;
        }
        try {
            if (not ReadImpl(size, offset, data)) {
                this->allocator_->Deallocate(data);
                return nullptr;
            }
        } catch (...) {
            this->allocator_->Deallocate(data);
            throw;
        }
        need_release = true;
        return data;
    }

    bool
    MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
        if (count == 0) {
            return true;
        }
        if (cache_max_pages_ == 0) {
            return inner_io_->MultiReadImpl(datas, sizes, offsets, count);
        }

        uint64_t page_size = Page::DEFAULT_PAGE_SIZE;

        // Phase 1: scan all blocks, identify cache hits and misses
        // For each block, track which pages it needs and where in the output buffer they go
        struct BlockInfo {
            uint64_t data_offset;  // offset in datas buffer
            uint64_t size;
            uint64_t file_offset;
        };

        std::vector<BlockInfo> blocks(count);
        uint64_t data_pos = 0;
        for (uint64_t i = 0; i < count; ++i) {
            if (not is_valid_range(sizes[i], offsets[i])) {
                return false;
            }
            blocks[i] = {data_pos, sizes[i], offsets[i]};
            data_pos += sizes[i];
        }

        // Collect all needed page_ids and check cache
        // Use a set to deduplicate page_ids for batch loading
        std::vector<uint64_t> miss_page_ids;
        std::unordered_set<uint64_t> miss_page_set;

        for (const auto& block : blocks) {
            uint64_t cur_size = 0;
            while (cur_size < block.size) {
                uint64_t cur_offset = block.file_offset + cur_size;
                uint64_t page_id = cur_offset / page_size;
                uint64_t page_offset = cur_offset % page_size;
                uint64_t chunk = std::min(block.size - cur_size, page_size - page_offset);

                PagePtr page = cache_->Get(page_id);
                if (page == nullptr) {
                    if (miss_page_set.insert(page_id).second) {
                        miss_page_ids.push_back(page_id);
                    }
                }
                cur_size += chunk;
            }
        }

        // Phase 2: batch load all miss pages
        // Use a map to retain loaded pages and prevent eviction before copy
        std::unordered_map<uint64_t, PagePtr> loaded_pages;
        if (not miss_page_ids.empty()) {
            // Sort by page_id for sequential IO
            std::sort(miss_page_ids.begin(), miss_page_ids.end());

            uint64_t miss_count = miss_page_ids.size();
            std::vector<uint64_t> read_sizes(miss_count);
            std::vector<uint64_t> read_offsets(miss_count);
            uint64_t current_size = this->size_;

            // Allocate buffers for batch read
            std::vector<PagePtr> new_pages(miss_count);
            uint64_t total_miss_size = 0;
            for (uint64_t i = 0; i < miss_count; ++i) {
                uint64_t file_offset = miss_page_ids[i] * page_size;
                if (file_offset >= current_size) {
                    return false;
                }
                read_sizes[i] = std::min(page_size, current_size - file_offset);
                read_offsets[i] = file_offset;
                new_pages[i] = std::make_shared<Page>(this->allocator_);
                if (new_pages[i]->Data() == nullptr) {
                    return false;
                }
                total_miss_size += read_sizes[i];
            }

            // Build contiguous buffer for batch read
            std::vector<uint8_t> batch_buf(total_miss_size);

            // Batch load from inner IO using file offsets
            // inner_io_->MultiReadImpl expects (datas, sizes, offsets, count) where
            // datas is contiguous buffer, sizes[] are read sizes, offsets[] are FILE offsets
            // Output is packed contiguously in datas
            if (not inner_io_->MultiReadImpl(
                    batch_buf.data(), read_sizes.data(), read_offsets.data(), miss_count)) {
                return false;
            }

            // Copy from batch buffer to page buffers and retain in map
            uint64_t buf_offset = 0;
            for (uint64_t i = 0; i < miss_count; ++i) {
                std::memcpy(new_pages[i]->Data(), batch_buf.data() + buf_offset, read_sizes[i]);
                buf_offset += read_sizes[i];
                loaded_pages[miss_page_ids[i]] = new_pages[i];
                cache_->Insert(miss_page_ids[i], new_pages[i]);
            }
        }

        // Phase 3: copy all data from cache to output buffer
        // Use loaded_pages map as fallback if cache evicted the page
        for (const auto& block : blocks) {
            uint64_t cur_size = 0;
            while (cur_size < block.size) {
                uint64_t cur_offset = block.file_offset + cur_size;
                uint64_t page_id = cur_offset / page_size;
                uint64_t page_offset = cur_offset % page_size;
                uint64_t chunk = std::min(block.size - cur_size, page_size - page_offset);

                PagePtr page = cache_->Get(page_id);
                if (page == nullptr) {
                    // Fallback to loaded_pages if cache evicted it
                    auto it = loaded_pages.find(page_id);
                    if (it != loaded_pages.end()) {
                        page = it->second;
                    } else {
                        return false;
                    }
                }
                std::memcpy(
                    datas + block.data_offset + cur_size, page->Data() + page_offset, chunk);
                cur_size += chunk;
            }
        }

        return true;
    }

    void
    ReleaseImpl(const uint8_t* data) const {
        this->allocator_->Deallocate(const_cast<uint8_t*>(data));
    }

    void
    InitIOImpl(const IOParamPtr& io_param) {
        std::scoped_lock<std::mutex> lock(io_mutex_);
        auto inner_param = io_param;
        if (auto cache_param = std::dynamic_pointer_cast<CacheIOParameter>(io_param)) {
            inner_param = MakeInnerIOParam(cache_param);
        }
        inner_io_->InitIO(inner_param);
        if (this->size_ == 0) {
            if (auto reader_param = std::dynamic_pointer_cast<ReaderIOParameter>(inner_param);
                reader_param != nullptr and reader_param->reader != nullptr) {
                this->size_ = reader_param->reader->Size();
            } else {
                this->size_ = inner_io_->size_;
            }
        }
        inner_io_->size_ = this->size_;
        inner_io_->start_ = this->start_;
        cache_->Clear();
    }

    void
    ResizeImpl(uint64_t size) {
        std::scoped_lock<std::mutex> lock(io_mutex_);
        inner_io_->Resize(size);
        this->size_ = size;
        cache_->Clear();
    }

    void
    ShrinkImpl(uint64_t size) {
        if (size >= this->size_) {
            return;
        }
        std::scoped_lock<std::mutex> lock(io_mutex_);
        inner_io_->Shrink(size);
        this->size_ = size;
        cache_->Clear();
    }

    int64_t
    GetMemoryUsage() const {
        uint64_t memory = cache_->Size() * Page::DEFAULT_PAGE_SIZE;
        if constexpr (IOTmpl::InMemory) {
            memory += inner_io_->GetMemoryUsage();
        }
        return static_cast<int64_t>(memory);
    }

private:
    static IOParamPtr
    MakeInnerIOParam(const CacheIOParameterPtr& cache_param) {
        if (cache_param == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "CacheIO requires CacheIOParameter");
        }
        if (cache_param->inner_io_type_.empty()) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "CacheIO requires inner_io_type in cache_io params");
        }
        JsonType inner_json = cache_param->original_json_;
        inner_json[TYPE_KEY].SetString(cache_param->inner_io_type_);
        auto inner_param = IOParameter::GetIOParameterByJson(inner_json);
        if (inner_param == nullptr) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "CacheIO inner_io_type is invalid");
        }
        return inner_param;
    }

    void
    InvalidateRange(uint64_t size, uint64_t offset) {
        if (size == 0) {
            return;
        }
        if (offset > UINT64_MAX - (size - 1)) {
            cache_->Clear();
            return;
        }
        uint64_t page_size = Page::DEFAULT_PAGE_SIZE;
        uint64_t first_page = offset / page_size;
        uint64_t last_page = (offset + size - 1) / page_size;
        for (uint64_t page_id = first_page; page_id <= last_page; ++page_id) {
            cache_->Remove(page_id);
        }
    }

    bool
    is_valid_range(uint64_t size, uint64_t offset) const {
        return offset <= this->size_ and size <= this->size_ - offset;
    }

    PagePtr
    get_or_load_page(uint64_t page_id) const {
        uint64_t page_size = Page::DEFAULT_PAGE_SIZE;
        if (page_id > UINT64_MAX / page_size) {
            return nullptr;
        }
        uint64_t file_offset = page_id * page_size;
        uint64_t current_size = this->size_;
        if (file_offset >= current_size) {
            return nullptr;
        }

        PagePtr page = cache_->Get(page_id);
        if (page != nullptr) {
            return page;
        }

        uint64_t read_size = std::min(page_size, current_size - file_offset);

        auto new_page = std::make_shared<Page>(this->allocator_);
        if (new_page->Data() == nullptr) {
            return nullptr;
        }

        if (not inner_io_->ReadImpl(read_size, file_offset, new_page->Data())) {
            return nullptr;
        }

        if (cache_max_pages_ == 0) {
            return new_page;
        }
        return cache_->Insert(page_id, std::move(new_page));
    }

private:
    friend class CacheIOTest<IOTmpl>;

    mutable std::mutex io_mutex_;
    mutable std::unique_ptr<PageCache> cache_;
    std::unique_ptr<IOTmpl> inner_io_;
    uint64_t cache_max_pages_{0};
};

}  // namespace vsag
