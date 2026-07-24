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
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "io/cache_io/page_cache.h"

namespace vsag {

/**
 * @brief Sharded LRU PageCache for better concurrent performance.
 *
 * Divides the cache into N independent shards, each with its own lock.
 * Routes operations to shard[page_id % num_shards] to reduce lock contention.
 */
class ShardedLRUPageCache : public PageCache {
public:
    explicit ShardedLRUPageCache(uint64_t max_pages, uint64_t num_shards = 16);

    PagePtr
    Get(uint64_t page_id) override;

    PagePtr
    Insert(uint64_t page_id, PagePtr page) override;

    void
    Remove(uint64_t page_id) override;

    void
    Clear() override;

    uint64_t
    Size() const override;

protected:
    // These are not used in sharded implementation, but required by base class
    void
    OnAccess(uint64_t page_id) override {
    }

    void
    OnInsert(uint64_t page_id) override {
    }

    void
    OnRemove(uint64_t page_id) override {
    }

    uint64_t
    PickVictim() override {
        return UINT64_MAX;
    }

private:
    struct Shard {
        mutable std::mutex mutex;
        std::unordered_map<uint64_t, PagePtr> pages;
        std::list<uint64_t> order;
        std::unordered_map<uint64_t, std::list<uint64_t>::iterator> iters;
        uint64_t max_pages;

        explicit Shard(uint64_t max_pages_per_shard) : max_pages(max_pages_per_shard) {
        }
    };

    std::vector<std::unique_ptr<Shard>> shards_;
    uint64_t num_shards_;

    uint64_t
    GetShardIndex(uint64_t page_id) const {
        return page_id % num_shards_;
    }
};

}  // namespace vsag
