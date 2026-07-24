
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

#include "visited_list.h"

#include <algorithm>
#include <limits>
#include <random>
#include <thread>

#include "impl/allocator/default_allocator.h"
#include "unittest.h"
using namespace vsag;

namespace {
class CountingAllocator : public DefaultAllocator {
public:
    void*
    Allocate(uint64_t size) override {
        ++allocate_count_;
        return DefaultAllocator::Allocate(size);
    }

    void
    Deallocate(void* p) override {
        ++deallocate_count_;
        DefaultAllocator::Deallocate(p);
    }

    uint64_t allocate_count_{0};
    uint64_t deallocate_count_{0};
};
}  // namespace

TEST_CASE("VisitedList Basic Test", "[ut][VisitedList]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    auto size = 10000;
    auto vl_ptr = std::make_shared<VisitedList>(size, allocator.get());

    SECTION("test set & get normal") {
        int count = 500;
        std::unordered_set<InnerIdType> ids;
        for (int i = 0; i < count; ++i) {
            auto id = random() % size;
            ids.insert(id);
            vl_ptr->Set(id);
        }
        for (auto& id : ids) {
            REQUIRE(vl_ptr->Get(id));
        }

        for (int i = 0; i < size; ++i) {
            if (ids.count(i) == 0) {
                REQUIRE(vl_ptr->Get(i) == false);
            }
        }
    }

    SECTION("test reset") {
        int count = 500;
        std::unordered_set<InnerIdType> ids;
        for (int i = 0; i < count; ++i) {
            auto id = random() % size;
            ids.insert(id);
            vl_ptr->Set(id);
        }
        vl_ptr->Reset();
        for (auto& id : ids) {
            REQUIRE(vl_ptr->Get(id) == false);
        }
    }

    SECTION("test word boundaries and repeated sets") {
        const std::vector<InnerIdType> ids = {0, 1, 63, 64, 65, static_cast<InnerIdType>(size - 1)};
        for (const auto id : ids) {
            vl_ptr->Set(id);
            vl_ptr->Set(id);
            REQUIRE(vl_ptr->Get(id));
        }

        vl_ptr->Reset();
        for (const auto id : ids) {
            REQUIRE_FALSE(vl_ptr->Get(id));
        }

        vl_ptr->Set(64);
        REQUIRE(vl_ptr->Get(64));
        vl_ptr->Reset();
        REQUIRE_FALSE(vl_ptr->Get(64));
    }

    SECTION("test memory usage") {
        const auto word_count = (static_cast<uint64_t>(size) + VisitedList::kBitsPerWord - 1) /
                                VisitedList::kBitsPerWord;
        const auto expected = sizeof(VisitedList) + word_count * (sizeof(VisitedList::WordType) +
                                                                  sizeof(VisitedList::TagType));
        REQUIRE(vl_ptr->GetMemoryUsage() == expected);
    }

    SECTION("test tag overflow") {
        vl_ptr->Set(0);
        for (uint64_t i = 0; i < std::numeric_limits<VisitedList::TagType>::max(); ++i) {
            vl_ptr->Reset();
        }
        REQUIRE_FALSE(vl_ptr->Get(0));
        vl_ptr->Set(0);
        REQUIRE(vl_ptr->Get(0));
    }
}

TEST_CASE("VisitedList Zero Size Test", "[ut][VisitedList]") {
    CountingAllocator allocator;
    {
        VisitedList visited_list(0, &allocator);
        REQUIRE(visited_list.GetMemoryUsage() == sizeof(VisitedList));
        for (uint64_t i = 0; i < std::numeric_limits<VisitedList::TagType>::max(); ++i) {
            visited_list.Reset();
        }
    }
    REQUIRE(allocator.allocate_count_ == 0);
    REQUIRE(allocator.deallocate_count_ == 0);
}

TEST_CASE("VisitedList Randomized Differential Test", "[ut][VisitedList]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    std::mt19937_64 random_generator(20260715);
    const std::vector<InnerIdType> sizes = {1, 2, 5, 63, 64, 65, 127, 128, 129, 10000};

    for (const auto size : sizes) {
        VisitedList visited_list(size, allocator.get());
        std::vector<bool> reference(static_cast<uint64_t>(size), false);
        std::uniform_int_distribution<uint64_t> id_distribution(0, static_cast<uint64_t>(size) - 1);
        std::uniform_int_distribution<uint64_t> operation_distribution(0, 15);

        for (uint64_t operation = 0; operation < 20000; ++operation) {
            const auto id = static_cast<InnerIdType>(id_distribution(random_generator));
            const auto operation_type = operation_distribution(random_generator);
            if (operation_type == 0) {
                visited_list.Reset();
                std::fill(reference.begin(), reference.end(), false);
            } else if (operation_type <= 8) {
                visited_list.Set(id);
                reference[static_cast<uint64_t>(id)] = true;
            } else {
                REQUIRE(visited_list.Get(id) == reference[static_cast<uint64_t>(id)]);
            }
        }

        for (InnerIdType id = 0; id < size; ++id) {
            REQUIRE(visited_list.Get(id) == reference[static_cast<uint64_t>(id)]);
        }
    }
}

TEST_CASE("VisitedListPool Basic Test", "[ut][VisitedListPool]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    auto init_size = 10;
    auto vl_size = 1000;
    std::vector<Allocator*> allocators = {allocator.get(), nullptr};
    for (auto allocator_ptr : allocators) {
        auto pool =
            std::make_shared<VisitedListPool>(init_size, allocator_ptr, vl_size, allocator.get());

        auto TestVL = [&](std::shared_ptr<VisitedList>& vl_ptr) {
            int count = 500;
            std::unordered_set<InnerIdType> ids;
            for (int i = 0; i < count; ++i) {
                auto id = random() % vl_size;
                ids.insert(id);
                vl_ptr->Set(id);
            }
            for (auto& id : ids) {
                REQUIRE(vl_ptr->Get(id) == true);
            }

            for (InnerIdType i = 0; i < vl_size; ++i) {
                if (ids.count(i) == 0) {
                    REQUIRE(vl_ptr->Get(i) == false);
                }
            }
        };

        SECTION("test concurrency") {
            auto func = [&]() {
                int count = 10;
                int max_operators = 20;
                std::vector<std::shared_ptr<VisitedList>> results;
                for (int i = 0; i < count; ++i) {
                    auto opt = random() % max_operators + 1;
                    for (auto j = 0; j < opt; ++j) {
                        results.emplace_back(pool->TakeOne());
                    }
                    for (auto& result : results) {
                        pool->ReturnOne(result);
                    }
                    results.clear();
                }
            };
            std::vector<std::shared_ptr<std::thread>> threads;
            auto thread_count = 5;
            threads.reserve(thread_count);
            for (auto i = 0; i < thread_count; ++i) {
                threads.emplace_back((std::make_shared<std::thread>(func)));
            }
            for (auto& thread : threads) {
                thread->join();
            }
            for (int i = 0; i < 10; ++i) {
                auto vl = pool->TakeOne();
                TestVL(vl);
                pool->ReturnOne(vl);
            }
        }
    }
}
