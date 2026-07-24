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

#include "io/cache_io/page_cache.h"

#include <algorithm>

namespace vsag {

PageCache::PageCache(uint64_t max_pages) : max_pages_(max_pages) {
}

PagePtr
PageCache::Get(uint64_t page_id) {
    std::scoped_lock<std::mutex> lock(mutex_);
    auto it = pages_.find(page_id);
    if (it == pages_.end()) {
        return nullptr;
    }
    OnAccess(page_id);
    return it->second;
}

PagePtr
PageCache::Insert(uint64_t page_id, PagePtr page) {
    std::scoped_lock<std::mutex> lock(mutex_);
    auto existing = pages_.find(page_id);
    if (existing != pages_.end()) {
        OnAccess(page_id);
        return existing->second;
    }
    if (max_pages_ == 0) {
        return page;
    }
    while (pages_.size() >= max_pages_) {
        uint64_t victim = PickVictim();
        if (victim == UINT64_MAX or pages_.find(victim) == pages_.end()) {
            victim = pages_.begin()->first;
        }
        OnRemove(victim);
        pages_.erase(victim);
    }
    pages_[page_id] = std::move(page);
    OnInsert(page_id);
    return pages_[page_id];
}

void
PageCache::Remove(uint64_t page_id) {
    std::scoped_lock<std::mutex> lock(mutex_);
    auto it = pages_.find(page_id);
    if (it != pages_.end()) {
        OnRemove(page_id);
        pages_.erase(it);
    }
}

void
PageCache::Clear() {
    std::scoped_lock<std::mutex> lock(mutex_);
    for (const auto& page_pair : pages_) {
        OnRemove(page_pair.first);
    }
    pages_.clear();
}

uint64_t
PageCache::Size() const {
    std::scoped_lock<std::mutex> lock(mutex_);
    return pages_.size();
}

}  // namespace vsag
