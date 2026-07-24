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
#include <unordered_map>

#include "io/cache_io/page_cache.h"

namespace vsag {

/**
 * @brief PageCache with Least-Recently-Used eviction policy.
 */
class LRUPageCache : public PageCache {
public:
    explicit LRUPageCache(uint64_t max_pages);

protected:
    void
    OnAccess(uint64_t page_id) override;

    void
    OnInsert(uint64_t page_id) override;

    void
    OnRemove(uint64_t page_id) override;

    uint64_t
    PickVictim() override;

private:
    std::list<uint64_t> order_;
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> iters_;
};

}  // namespace vsag
