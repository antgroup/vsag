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
#include <deque>

#include "impl/allocator/safe_allocator.h"
#include "unittest.h"

using namespace vsag;

namespace {

class FifoPageCache : public PageCache {
public:
    explicit FifoPageCache(uint64_t max_pages) : PageCache(max_pages) {
    }

protected:
    void
    OnAccess(uint64_t page_id) override {
        last_access_ = page_id;
    }

    void
    OnInsert(uint64_t page_id) override {
        order_.push_back(page_id);
    }

    void
    OnRemove(uint64_t page_id) override {
        auto it = std::find(order_.begin(), order_.end(), page_id);
        if (it != order_.end()) {
            order_.erase(it);
        }
    }

    uint64_t
    PickVictim() override {
        if (order_.empty()) {
            return UINT64_MAX;
        }
        return order_.front();
    }

public:
    uint64_t last_access_{UINT64_MAX};

private:
    std::deque<uint64_t> order_;
};

PagePtr
MakePage(Allocator* allocator, uint8_t value) {
    auto page = std::make_shared<Page>(allocator);
    page->Data()[0] = value;
    return page;
}

}  // namespace

TEST_CASE("PageCache Insert Get Remove Test", "[PageCache][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    FifoPageCache cache(2);
    auto page = MakePage(allocator.get(), 7);

    REQUIRE(cache.Insert(1, page) == page);
    REQUIRE(cache.Size() == 1);
    REQUIRE(cache.Get(1) == page);
    REQUIRE(cache.last_access_ == 1);
    REQUIRE(cache.Get(2) == nullptr);

    cache.Remove(1);
    REQUIRE(cache.Size() == 0);
    REQUIRE(cache.Get(1) == nullptr);
}

TEST_CASE("PageCache Duplicate Insert Test", "[PageCache][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    FifoPageCache cache(2);
    auto first = MakePage(allocator.get(), 1);
    auto second = MakePage(allocator.get(), 2);

    REQUIRE(cache.Insert(10, first) == first);
    REQUIRE(cache.Insert(10, second) == first);
    REQUIRE(cache.Size() == 1);
    REQUIRE(cache.Get(10)->Data()[0] == 1);
}

TEST_CASE("PageCache Eviction Test", "[PageCache][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    FifoPageCache cache(2);
    cache.Insert(1, MakePage(allocator.get(), 1));
    cache.Insert(2, MakePage(allocator.get(), 2));
    cache.Insert(3, MakePage(allocator.get(), 3));

    REQUIRE(cache.Size() == 2);
    REQUIRE(cache.Get(1) == nullptr);
    REQUIRE(cache.Get(2) != nullptr);
    REQUIRE(cache.Get(3) != nullptr);
}
