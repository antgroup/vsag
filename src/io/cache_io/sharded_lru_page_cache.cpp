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

#include "io/cache_io/sharded_lru_page_cache.h"

#include <algorithm>

namespace vsag {

ShardedLRUPageCache::ShardedLRUPageCache(uint64_t max_pages, uint64_t num_shards)
    : PageCache(max_pages), num_shards_(num_shards) {
    shards_.reserve(num_shards);
    // Distribute pages evenly across shards
    uint64_t base_pages = max_pages / num_shards;
    uint64_t extra_pages = max_pages % num_shards;
    for (uint64_t i = 0; i < num_shards; ++i) {
        uint64_t shard_capacity = base_pages + (i < extra_pages ? 1 : 0);
        shards_.push_back(std::make_unique<Shard>(shard_capacity));
    }
}

PagePtr
ShardedLRUPageCache::Get(uint64_t page_id) {
    uint64_t shard_idx = GetShardIndex(page_id);
    auto& shard = *shards_[shard_idx];
    std::scoped_lock<std::mutex> lock(shard.mutex);

    auto it = shard.pages.find(page_id);
    if (it == shard.pages.end()) {
        return nullptr;
    }

    // Update LRU order
    auto order_it = shard.iters.find(page_id);
    if (order_it != shard.iters.end()) {
        shard.order.splice(shard.order.begin(), shard.order, order_it->second);
    }

    return it->second;
}

PagePtr
ShardedLRUPageCache::Insert(uint64_t page_id, PagePtr page) {
    uint64_t shard_idx = GetShardIndex(page_id);
    auto& shard = *shards_[shard_idx];
    std::scoped_lock<std::mutex> lock(shard.mutex);

    // Check if already exists
    auto existing = shard.pages.find(page_id);
    if (existing != shard.pages.end()) {
        // Update LRU order
        auto order_it = shard.iters.find(page_id);
        if (order_it != shard.iters.end()) {
            shard.order.splice(shard.order.begin(), shard.order, order_it->second);
        }
        return existing->second;
    }

    // Skip insert if shard has zero capacity
    if (shard.max_pages == 0) {
        return page;
    }
    // Evict if necessary
    {
        while (shard.pages.size() >= shard.max_pages) {
            if (shard.order.empty()) {
                break;
            }
            uint64_t victim = shard.order.back();
            shard.order.pop_back();
            shard.iters.erase(victim);
            shard.pages.erase(victim);
        }
    }

    // Insert new page
    shard.order.push_front(page_id);
    shard.iters[page_id] = shard.order.begin();
    shard.pages[page_id] = std::move(page);

    return shard.pages[page_id];
}

void
ShardedLRUPageCache::Remove(uint64_t page_id) {
    uint64_t shard_idx = GetShardIndex(page_id);
    auto& shard = *shards_[shard_idx];
    std::scoped_lock<std::mutex> lock(shard.mutex);

    auto it = shard.pages.find(page_id);
    if (it != shard.pages.end()) {
        auto order_it = shard.iters.find(page_id);
        if (order_it != shard.iters.end()) {
            shard.order.erase(order_it->second);
            shard.iters.erase(order_it);
        }
        shard.pages.erase(it);
    }
}

void
ShardedLRUPageCache::Clear() {
    for (const auto& shard_ptr : shards_) {
        auto& shard = *shard_ptr;
        std::scoped_lock<std::mutex> lock(shard.mutex);
        shard.pages.clear();
        shard.order.clear();
        shard.iters.clear();
    }
}

uint64_t
ShardedLRUPageCache::Size() const {
    uint64_t total = 0;
    for (const auto& shard_ptr : shards_) {
        auto& shard = *shard_ptr;
        std::scoped_lock<std::mutex> lock(shard.mutex);
        total += shard.pages.size();
    }
    return total;
}

}  // namespace vsag
