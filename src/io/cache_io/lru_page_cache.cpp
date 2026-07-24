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

#include "io/cache_io/lru_page_cache.h"

namespace vsag {

LRUPageCache::LRUPageCache(uint64_t max_pages) : PageCache(max_pages) {
}

void
LRUPageCache::OnAccess(uint64_t page_id) {
    auto it = iters_.find(page_id);
    if (it != iters_.end()) {
        order_.splice(order_.begin(), order_, it->second);
    }
}

void
LRUPageCache::OnInsert(uint64_t page_id) {
    order_.push_front(page_id);
    iters_[page_id] = order_.begin();
}

void
LRUPageCache::OnRemove(uint64_t page_id) {
    auto it = iters_.find(page_id);
    if (it != iters_.end()) {
        order_.erase(it->second);
        iters_.erase(it);
    }
}

uint64_t
LRUPageCache::PickVictim() {
    if (order_.empty()) {
        return UINT64_MAX;
    }
    return order_.back();
}

}  // namespace vsag
