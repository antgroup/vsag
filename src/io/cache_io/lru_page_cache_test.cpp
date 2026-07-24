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

#include "impl/allocator/safe_allocator.h"
#include "unittest.h"

using namespace vsag;

namespace {

PagePtr
MakePage(Allocator* allocator, uint8_t value) {
    auto page = std::make_shared<Page>(allocator);
    page->Data()[0] = value;
    return page;
}

}  // namespace

TEST_CASE("LRUPageCache Evicts Least Recently Used Page", "[LRUPageCache][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    LRUPageCache cache(2);
    cache.Insert(1, MakePage(allocator.get(), 1));
    cache.Insert(2, MakePage(allocator.get(), 2));

    REQUIRE(cache.Get(1) != nullptr);
    cache.Insert(3, MakePage(allocator.get(), 3));

    REQUIRE(cache.Size() == 2);
    REQUIRE(cache.Get(1) != nullptr);
    REQUIRE(cache.Get(2) == nullptr);
    REQUIRE(cache.Get(3) != nullptr);
}

TEST_CASE("LRUPageCache Duplicate Insert Keeps Original Page", "[LRUPageCache][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    LRUPageCache cache(2);
    auto first = MakePage(allocator.get(), 1);
    auto second = MakePage(allocator.get(), 2);

    REQUIRE(cache.Insert(1, first) == first);
    REQUIRE(cache.Insert(1, second) == first);
    REQUIRE(cache.Size() == 1);
    REQUIRE(cache.Get(1)->Data()[0] == 1);
}

TEST_CASE("LRUPageCache Remove Updates Eviction State", "[LRUPageCache][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    LRUPageCache cache(2);
    cache.Insert(1, MakePage(allocator.get(), 1));
    cache.Insert(2, MakePage(allocator.get(), 2));
    cache.Remove(1);
    cache.Insert(3, MakePage(allocator.get(), 3));

    REQUIRE(cache.Size() == 2);
    REQUIRE(cache.Get(1) == nullptr);
    REQUIRE(cache.Get(2) != nullptr);
    REQUIRE(cache.Get(3) != nullptr);
}
