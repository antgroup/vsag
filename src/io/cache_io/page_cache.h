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
#include <mutex>
#include <unordered_map>

#include "io/cache_io/page.h"

namespace vsag {

/**
 * @brief Page storage manager with pluggable eviction policy.
 *
 * PageCache owns the cached pages and decides which page to evict
 * when the capacity is reached. Pages are managed by shared_ptr:
 * a page returned to the caller stays valid even if it is evicted
 * from the cache concurrently.
 */
class PageCache {
public:
    explicit PageCache(uint64_t max_pages);

    virtual ~PageCache() = default;

    /**
     * @brief Look up a page, marking it as recently accessed.
     *
     * @param page_id The page to look up.
     * @return The cached page, or nullptr on miss.
     */
    virtual PagePtr
    Get(uint64_t page_id);

    /**
     * @brief Insert a page, evicting victims first if the cache is full.
     *
     * If the page already exists, the existing page is returned instead.
     *
     * @param page_id The page id.
     * @param page The page to insert.
     * @return The cached page.
     */
    virtual PagePtr
    Insert(uint64_t page_id, PagePtr page);

    /**
     * @brief Remove a page from the cache.
     *
     * @param page_id The page to remove.
     */
    virtual void
    Remove(uint64_t page_id);

    /**
     * @brief Clear all cached pages.
     */
    virtual void
    Clear();

    /**
     * @brief Number of pages currently cached.
     */
    virtual uint64_t
    Size() const;

protected:
    virtual void
    OnAccess(uint64_t page_id) = 0;

    virtual void
    OnInsert(uint64_t page_id) = 0;

    virtual void
    OnRemove(uint64_t page_id) = 0;

    virtual uint64_t
    PickVictim() = 0;

protected:
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, PagePtr> pages_;
    uint64_t max_pages_{0};
};

}  // namespace vsag
