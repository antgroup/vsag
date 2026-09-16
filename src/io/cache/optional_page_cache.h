// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
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
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "io/common/io_parameter.h"
#include "io/core/cached_read_lease.h"
#include "io/core/cached_read_operation.h"
#include "io/core/read_lease.h"
#include "io/core/read_operation.h"
#include "io/core/read_request.h"
#include "io/read_cache/lru_page_cache.h"
#include "io/read_cache/page.h"
#include "io/read_cache/page_cache.h"
#include "vsag_exception.h"

namespace vsag {

class OptionalPageCache {
public:
    [[nodiscard]] bool
    Enabled() const {
        return cache_ != nullptr;
    }

    template <typename Backend>
    using Lease = std::conditional_t<std::is_same_v<typename Backend::Lease, AllocatorLease>,
                                     AllocatorLease,
                                     CachedReadLease<typename Backend::Lease>>;

    template <typename Backend>
    using Operation =
        std::conditional_t<std::is_same_v<typename Backend::Operation, ImmediateOperation>,
                           ImmediateOperation,
                           CachedReadOperation<typename Backend::Operation>>;

    template <typename Backend>
    [[nodiscard]] bool
    ReadAt(const Backend& backend, uint64_t logical_size, const ReadRequest& request) const {
        if (cache_ == nullptr) {
            return backend.ReadAt(request.offset, request.size, request.destination);
        }
        uint64_t page_offset = request.offset % Page::DEFAULT_PAGE_SIZE;
        if (request.size <= Page::DEFAULT_PAGE_SIZE - page_offset) {
            return ReadSinglePageCached(backend, logical_size, request, page_offset);
        }
        bool all_hits = false;
        if (not ReadCachedHits(&request, 1, all_hits)) {
            return false;
        }
        if (all_hits) {
            return true;
        }
        return ReadManyCached(backend, logical_size, &request, 1);
    }

    template <typename Backend>
    [[nodiscard]] bool
    ReadMany(const Backend& backend,
             uint64_t logical_size,
             const ReadRequest* requests,
             uint64_t count) const {
        if (cache_ == nullptr) {
            return backend.ReadMany(requests, count);
        }
        if (count == 1) {
            if (requests[0].size == 0) {
                return true;
            }
            uint64_t page_offset = requests[0].offset % Page::DEFAULT_PAGE_SIZE;
            if (requests[0].size <= Page::DEFAULT_PAGE_SIZE - page_offset) {
                return ReadSinglePageCached(backend, logical_size, requests[0], page_offset);
            }
        }
        bool all_hits = false;
        if (not ReadCachedHits(requests, count, all_hits)) {
            return false;
        }
        if (all_hits) {
            return true;
        }
        return ReadManyCached(backend, logical_size, requests, count);
    }

    template <typename Backend>
    [[nodiscard]] bool
    ReadManyContiguous(const Backend& backend,
                       uint64_t logical_size,
                       uint8_t* destination,
                       const uint64_t* sizes,
                       const uint64_t* offsets,
                       uint64_t count) const {
        if (cache_ == nullptr) {
            return backend.ReadManyContiguous(destination, sizes, offsets, count);
        }
        if (count == 1) {
            if (sizes[0] == 0) {
                return true;
            }
            uint64_t page_offset = offsets[0] % Page::DEFAULT_PAGE_SIZE;
            if (sizes[0] <= Page::DEFAULT_PAGE_SIZE - page_offset) {
                return ReadSinglePageCached(backend,
                                            logical_size,
                                            ReadRequest{destination, offsets[0], sizes[0]},
                                            page_offset);
            }
        }
        bool all_hits = false;
        if (not ReadContiguousCachedHits(destination, sizes, offsets, count, all_hits)) {
            return false;
        }
        if (all_hits) {
            return true;
        }
        std::vector<ReadRequest> requests;
        requests.reserve(count);
        for (uint64_t i = 0; i < count; ++i) {
            requests.emplace_back(ReadRequest{destination, offsets[i], sizes[i]});
            destination += sizes[i];
        }
        return ReadManyCached(backend, logical_size, requests.data(), requests.size());
    }

    template <typename Backend>
    [[nodiscard]] Operation<Backend>
    SubmitReads(const Backend& backend,
                uint64_t logical_size,
                const ReadRequest* requests,
                uint64_t count) const {
        if (cache_ == nullptr) {
            return WrapBackendOperation<Backend>(backend.SubmitReads(requests, count));
        }
        return WrapCachedOperation<Backend>(
            ImmediateOperation(ReadMany(backend, logical_size, requests, count)));
    }

    template <typename Backend>
    [[nodiscard]] Lease<Backend>
    Acquire(const Backend& backend, uint64_t logical_size, uint64_t offset, uint64_t size) const {
        if (cache_ == nullptr) {
            return WrapBackendLease<Backend>(backend.Acquire(offset, size));
        }
        if (size == 0) {
            return Lease<Backend>{};
        }
        Allocator* allocator = backend.AllocatorPtr();
        auto* data = static_cast<uint8_t*>(allocator->Allocate(size));
        if (data == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                "OptionalPageCache acquire allocation failed");
        }
        AllocatorOwner owner(allocator, data);
        if (not ReadAt(backend, logical_size, ReadRequest{data, offset, size})) {
            return Lease<Backend>{};
        }
        return WrapCachedLease<Backend>(AllocatorLease(data, size, std::move(owner)));
    }

    template <typename Backend>
    [[nodiscard]] const uint8_t*
    LegacyRead(const Backend& backend,
               uint64_t logical_size,
               uint64_t offset,
               uint64_t size,
               bool& need_release) const {
        if (cache_ == nullptr) {
            return backend.LegacyRead(offset, size, need_release);
        }
        need_release = false;
        if (size == 0) {
            return nullptr;
        }
        Allocator* allocator = backend.AllocatorPtr();
        auto* data = static_cast<uint8_t*>(allocator->Allocate(size));
        if (data == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                "OptionalPageCache legacy allocation failed");
        }
        try {
            if (not ReadAt(backend, logical_size, ReadRequest{data, offset, size})) {
                allocator->Deallocate(data);
                return nullptr;
            }
        } catch (...) {
            allocator->Deallocate(data);
            throw;
        }
        need_release = true;
        return data;
    }

    template <typename Backend>
    void
    Release(const Backend& backend, const uint8_t* data) const {
        if (cache_ == nullptr) {
            backend.Release(data);
            return;
        }
        backend.AllocatorPtr()->Deallocate(const_cast<uint8_t*>(data));
    }

    template <typename Backend>
    void
    Invalidate(const Backend&, uint64_t offset, uint64_t size) {
        if (cache_ == nullptr or size == 0) {
            return;
        }
        const uint64_t tail = size - 1;
        const uint64_t last_byte = offset > UINT64_MAX - tail ? UINT64_MAX : offset + tail;
        uint64_t first_page = offset / Page::DEFAULT_PAGE_SIZE;
        uint64_t last_page = last_byte / Page::DEFAULT_PAGE_SIZE;
        if (first_page > UINT64_MAX - page_id_base_) {
            return;
        }
        last_page = std::min(last_page, UINT64_MAX - page_id_base_);
        const uint64_t page_count = last_page - first_page + 1;
        constexpr uint64_t per_page_invalidation_limit = 64;
        if (page_count > per_page_invalidation_limit) {
            cache_->RemoveRange(page_id_base_ + first_page, page_id_base_ + last_page);
            return;
        }
        for (uint64_t page_id = first_page;; ++page_id) {
            cache_->Remove(page_id_base_ + page_id);
            if (page_id == last_page) {
                break;
            }
        }
    }

    void
    Clear(uint64_t logical_size) {
        if (cache_ == nullptr or logical_size == 0) {
            return;
        }
        uint64_t page_count = logical_size / Page::DEFAULT_PAGE_SIZE +
                              static_cast<uint64_t>(logical_size % Page::DEFAULT_PAGE_SIZE != 0);
        if (page_id_base_ != 0) {
            page_count = std::min(page_count, UINT64_MAX - page_id_base_ + 1);
        }
        cache_->RemoveRange(page_id_base_, page_id_base_ + page_count - 1);
    }

    void
    Configure(const IOParamPtr& io_param) {
        if (io_param == nullptr or not io_param->enable_read_cache_) {
            cache_owner_.reset();
            cache_ = nullptr;
            page_id_base_ = 0;
            shared_cache_ = false;
            return;
        }
        uint64_t page_count = io_param->read_cache_total_size_ / Page::DEFAULT_PAGE_SIZE;
        if (page_count == 0) {
            cache_owner_.reset();
            cache_ = nullptr;
            page_id_base_ = 0;
            shared_cache_ = false;
            return;
        }
        cache_owner_ = std::make_shared<LRUPageCache>(page_count);
        cache_ = cache_owner_.get();
        page_id_base_ = 0;
        shared_cache_ = false;
    }

    void
    SetShared(std::shared_ptr<PageCache> cache, uint64_t page_id_base) {
        // Cache configuration is a setup-time operation and must not race with reads.
        cache_owner_ = std::move(cache);
        cache_ = cache_owner_.get();
        page_id_base_ = cache_ == nullptr ? 0 : page_id_base;
        shared_cache_ = cache_ != nullptr;
    }

private:
    [[nodiscard]] bool
    ReadCachedHits(const ReadRequest* requests, uint64_t count, bool& all_hits) const {
        for (uint64_t request_index = 0; request_index < count; ++request_index) {
            uint64_t copied = 0;
            while (copied < requests[request_index].size) {
                uint64_t current_offset = requests[request_index].offset + copied;
                uint64_t logical_page_id = current_offset / Page::DEFAULT_PAGE_SIZE;
                if (logical_page_id > UINT64_MAX - page_id_base_) {
                    return false;
                }
                uint64_t page_offset = current_offset % Page::DEFAULT_PAGE_SIZE;
                uint64_t fragment_size = std::min(requests[request_index].size - copied,
                                                  Page::DEFAULT_PAGE_SIZE - page_offset);
                PagePtr page = cache_->Get(page_id_base_ + logical_page_id);
                if (page == nullptr) {
                    all_hits = false;
                    return true;
                }
                std::memcpy(requests[request_index].destination + copied,
                            page->Data() + page_offset,
                            fragment_size);
                copied += fragment_size;
            }
        }
        all_hits = true;
        return true;
    }

    [[nodiscard]] bool
    ReadContiguousCachedHits(uint8_t* destination,
                             const uint64_t* sizes,
                             const uint64_t* offsets,
                             uint64_t count,
                             bool& all_hits) const {
        for (uint64_t request_index = 0; request_index < count; ++request_index) {
            uint64_t copied = 0;
            while (copied < sizes[request_index]) {
                uint64_t current_offset = offsets[request_index] + copied;
                uint64_t logical_page_id = current_offset / Page::DEFAULT_PAGE_SIZE;
                if (logical_page_id > UINT64_MAX - page_id_base_) {
                    return false;
                }
                uint64_t page_offset = current_offset % Page::DEFAULT_PAGE_SIZE;
                uint64_t fragment_size =
                    std::min(sizes[request_index] - copied, Page::DEFAULT_PAGE_SIZE - page_offset);
                PagePtr page = cache_->Get(page_id_base_ + logical_page_id);
                if (page == nullptr) {
                    all_hits = false;
                    return true;
                }
                std::memcpy(destination + copied, page->Data() + page_offset, fragment_size);
                copied += fragment_size;
            }
            destination += sizes[request_index];
        }
        all_hits = true;
        return true;
    }

    template <typename Backend>
    [[nodiscard]] static Lease<Backend>
    WrapBackendLease(typename Backend::Lease lease) {
        if constexpr (std::is_same_v<typename Backend::Lease, AllocatorLease>) {
            return std::move(lease);
        } else {
            return Lease<Backend>(std::move(lease));
        }
    }

    template <typename Backend>
    [[nodiscard]] static Lease<Backend>
    WrapCachedLease(AllocatorLease lease) {
        if constexpr (std::is_same_v<typename Backend::Lease, AllocatorLease>) {
            return std::move(lease);
        } else {
            return Lease<Backend>(std::move(lease));
        }
    }

    template <typename Backend>
    [[nodiscard]] static Operation<Backend>
    WrapBackendOperation(typename Backend::Operation operation) {
        if constexpr (std::is_same_v<typename Backend::Operation, ImmediateOperation>) {
            return std::move(operation);
        } else {
            return Operation<Backend>(std::move(operation));
        }
    }

    template <typename Backend>
    [[nodiscard]] static Operation<Backend>
    WrapCachedOperation(ImmediateOperation operation) {
        if constexpr (std::is_same_v<typename Backend::Operation, ImmediateOperation>) {
            return std::move(operation);
        } else {
            return Operation<Backend>(std::move(operation));
        }
    }

    struct PageFragment {
        uint8_t* destination{nullptr};
        uint32_t page_offset{0};
        uint32_t size{0};
        uint32_t page_index{0};
    };

    struct PlannedPage {
        uint64_t logical_page_id{0};
        uint64_t cache_page_id{0};
        PagePtr page;
        PagePtr candidate;
        PageCache::LoadHandle load_handle;
        bool should_load{false};
        bool load_completed{false};
    };

    template <typename Backend>
    [[nodiscard]] bool
    ReadSinglePageCached(const Backend& backend,
                         uint64_t logical_size,
                         const ReadRequest& request,
                         uint64_t page_offset) const {
        uint64_t logical_page_id = request.offset / Page::DEFAULT_PAGE_SIZE;
        if (logical_page_id > UINT64_MAX - page_id_base_) {
            return false;
        }
        uint64_t cache_page_id = page_id_base_ + logical_page_id;
        PagePtr page = cache_->Get(cache_page_id);
        if (page != nullptr) {
            std::memcpy(request.destination, page->Data() + page_offset, request.size);
            return true;
        }

        std::unique_lock<std::mutex> private_load_lock(private_load_mutex_, std::defer_lock);
        if (not shared_cache_) {
            private_load_lock.lock();
        }
        while (true) {
            auto load = cache_->Acquire(cache_page_id);
            if (load.page != nullptr) {
                page = std::move(load.page);
                break;
            }
            if (not load.should_load) {
                page = cache_->Wait(load.handle);
                if (page != nullptr) {
                    break;
                }
                if (cache_->IsStale(load.handle)) {
                    continue;
                }
                return false;
            }

            PagePtr candidate;
            bool success = false;
            try {
                uint64_t physical_page_offset = request.offset - page_offset;
                if (physical_page_offset < logical_size) {
                    candidate = std::make_shared<Page>(backend.AllocatorPtr());
                    success = candidate->Data() != nullptr;
                    if (success) {
                        uint64_t read_size =
                            std::min(Page::DEFAULT_PAGE_SIZE, logical_size - physical_page_offset);
                        success =
                            backend.ReadAt(physical_page_offset, read_size, candidate->Data());
                    }
                }
            } catch (...) {
                cache_->Complete(cache_page_id, load.handle, nullptr, false);
                throw;
            }
            page = cache_->Complete(
                cache_page_id, load.handle, success ? std::move(candidate) : nullptr, success);
            if (page != nullptr) {
                break;
            }
            if (cache_->IsStale(load.handle)) {
                continue;
            }
            return false;
        }
        std::memcpy(request.destination, page->Data() + page_offset, request.size);
        return true;
    }

    template <typename Backend>
    [[nodiscard]] bool
    ReadManyCached(const Backend& backend,
                   uint64_t logical_size,
                   const ReadRequest* requests,
                   uint64_t count) const {
        std::unique_lock<std::mutex> private_load_lock(private_load_mutex_, std::defer_lock);
        if (not shared_cache_) {
            private_load_lock.lock();
        }
        std::vector<PageFragment> fragments;
        std::vector<PlannedPage> pages;
        std::unordered_map<uint64_t, uint32_t> page_indices;
        fragments.reserve(count);
        pages.reserve(std::min<uint64_t>(count, LINEAR_PAGE_THRESHOLD));
        bool use_hash_index = false;

        for (uint64_t request_index = 0; request_index < count; ++request_index) {
            uint64_t copied = 0;
            while (copied < requests[request_index].size) {
                uint64_t current_offset = requests[request_index].offset + copied;
                uint64_t logical_page_id = current_offset / Page::DEFAULT_PAGE_SIZE;
                uint64_t page_offset = current_offset % Page::DEFAULT_PAGE_SIZE;
                uint64_t fragment_size = std::min(requests[request_index].size - copied,
                                                  Page::DEFAULT_PAGE_SIZE - page_offset);
                if (logical_page_id > UINT64_MAX - page_id_base_) {
                    return false;
                }
                uint64_t cache_page_id = page_id_base_ + logical_page_id;
                if (pages.size() >= UINT32_MAX) {
                    return false;
                }
                uint32_t page_index = UINT32_MAX;
                if (not use_hash_index) {
                    for (uint32_t i = 0; i < pages.size(); ++i) {
                        if (pages[i].cache_page_id == cache_page_id) {
                            page_index = i;
                            break;
                        }
                    }
                    if (page_index == UINT32_MAX and pages.size() >= LINEAR_PAGE_THRESHOLD) {
                        page_indices.reserve(pages.size() * 2);
                        for (uint32_t i = 0; i < pages.size(); ++i) {
                            page_indices.emplace(pages[i].cache_page_id, i);
                        }
                        use_hash_index = true;
                    }
                }
                if (use_hash_index) {
                    auto it = page_indices.find(cache_page_id);
                    if (it != page_indices.end()) {
                        page_index = it->second;
                    }
                }
                if (page_index == UINT32_MAX) {
                    page_index = static_cast<uint32_t>(pages.size());
                    pages.emplace_back(PlannedPage{logical_page_id,
                                                   cache_page_id,
                                                   cache_->Get(cache_page_id),
                                                   nullptr,
                                                   {},
                                                   false});
                    if (use_hash_index) {
                        page_indices.emplace(cache_page_id, page_index);
                    }
                }
                fragments.emplace_back(PageFragment{requests[request_index].destination + copied,
                                                    static_cast<uint32_t>(page_offset),
                                                    static_cast<uint32_t>(fragment_size),
                                                    page_index});
                copied += fragment_size;
            }
        }

        while (true) {
            bool stale = false;
            std::vector<uint32_t> missing_page_indices;
            std::vector<ReadRequest> backend_requests;
            std::vector<uint32_t> leader_indices;
            missing_page_indices.reserve(pages.size());
            backend_requests.reserve(pages.size());
            leader_indices.reserve(pages.size());

            for (uint32_t page_index = 0; page_index < pages.size(); ++page_index) {
                PlannedPage& page = pages[page_index];
                page.page = cache_->Get(page.cache_page_id);
                page.candidate.reset();
                page.load_handle = {};
                page.should_load = false;
                page.load_completed = false;
                if (page.page == nullptr) {
                    missing_page_indices.emplace_back(page_index);
                }
            }

            try {
                for (uint32_t page_index : missing_page_indices) {
                    PlannedPage& page = pages[page_index];
                    auto load = cache_->Acquire(page.cache_page_id);
                    page.page = std::move(load.page);
                    page.load_handle = std::move(load.handle);
                    page.should_load = load.should_load;
                }

                for (uint32_t page_index : missing_page_indices) {
                    PlannedPage& page = pages[page_index];
                    if (page.page != nullptr or not page.should_load) {
                        continue;
                    }
                    if (page.logical_page_id > UINT64_MAX / Page::DEFAULT_PAGE_SIZE) {
                        FailIncompleteLoads(pages);
                        return false;
                    }
                    uint64_t page_offset = page.logical_page_id * Page::DEFAULT_PAGE_SIZE;
                    if (page_offset >= logical_size) {
                        FailIncompleteLoads(pages);
                        return false;
                    }
                    page.candidate = std::make_shared<Page>(backend.AllocatorPtr());
                    if (page.candidate->Data() == nullptr) {
                        FailIncompleteLoads(pages);
                        return false;
                    }
                    uint64_t read_size =
                        std::min(Page::DEFAULT_PAGE_SIZE, logical_size - page_offset);
                    backend_requests.emplace_back(
                        ReadRequest{page.candidate->Data(), page_offset, read_size});
                    leader_indices.emplace_back(page_index);
                }

                if (not backend_requests.empty() and
                    not backend.ReadMany(backend_requests.data(), backend_requests.size())) {
                    FailIncompleteLoads(pages);
                    return false;
                }

                for (uint32_t page_index : leader_indices) {
                    PlannedPage& page = pages[page_index];
                    page.page = cache_->Complete(
                        page.cache_page_id, page.load_handle, std::move(page.candidate), true);
                    page.load_completed = true;
                    stale = stale or cache_->IsStale(page.load_handle);
                }
            } catch (...) {
                FailIncompleteLoads(pages);
                throw;
            }

            for (uint32_t page_index : missing_page_indices) {
                PlannedPage& page = pages[page_index];
                if (page.page != nullptr or page.should_load) {
                    continue;
                }
                page.page = cache_->Wait(page.load_handle);
                stale = stale or cache_->IsStale(page.load_handle);
            }
            if (stale) {
                continue;
            }

            for (const PageFragment& fragment : fragments) {
                const PagePtr& page = pages[fragment.page_index].page;
                if (page == nullptr) {
                    return false;
                }
                std::memcpy(
                    fragment.destination, page->Data() + fragment.page_offset, fragment.size);
            }
            return true;
        }
    }

    void
    FailIncompleteLoads(std::vector<PlannedPage>& pages) const {
        for (PlannedPage& page : pages) {
            if (page.should_load and not page.load_completed) {
                cache_->Complete(page.cache_page_id, page.load_handle, nullptr, false);
                page.load_completed = true;
            }
        }
    }

    std::shared_ptr<PageCache> cache_owner_;
    PageCache* cache_{nullptr};
    uint64_t page_id_base_{0};
    bool shared_cache_{false};
    mutable std::mutex private_load_mutex_;
    // Linear lookup avoids hash allocation for the common tiny-batch case; switch early enough
    // that the bounded O(N^2) prefix remains cheaper than constructing the hash table.
    static constexpr uint64_t LINEAR_PAGE_THRESHOLD = 8;
};

}  // namespace vsag
