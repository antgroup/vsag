// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include "search_lock.h"

#include <atomic>
#include <stdexcept>
#include <thread>
#include <vector>

#include "impl/allocator/default_allocator.h"
#include "unittest.h"
#include "utils/visited_list.h"

using namespace vsag;

TEST_CASE("Search lock policy restores scopes and is thread local", "[ut][search_no_lock]") {
    struct CountingMutex {
        int held{0};
        void
        lock_shared() {
            ++held;
        }
        void
        unlock_shared() {
            --held;
        }
    } mutex;
    REQUIRE_FALSE(SearchLockScope::IsNoLock());
    {
        auto lock = MakeSearchReadLock(mutex);
        REQUIRE(lock.owns_lock());
        REQUIRE(mutex.held == 1);
    }
    REQUIRE(mutex.held == 0);
    try {
        const SearchLockScope outer(true);
        auto empty = MakeSearchReadLock(mutex);
        REQUIRE_FALSE(empty.owns_lock());
        REQUIRE(mutex.held == 0);
        bool worker_default = false;
        std::thread worker([&] { worker_default = not SearchLockScope::IsNoLock(); });
        worker.join();
        REQUIRE(worker_default);
        {
            const SearchLockScope inner(false);
            auto lock = MakeSearchReadLock(mutex);
            auto moved = std::move(lock);
            REQUIRE(moved.owns_lock());
            REQUIRE_FALSE(lock.owns_lock());
            REQUIRE(mutex.held == 1);
        }
        REQUIRE(SearchLockScope::IsNoLock());
        REQUIRE(mutex.held == 0);
        throw std::runtime_error("restore on exception");
    } catch (const std::runtime_error&) {
    }
    REQUIRE_FALSE(SearchLockScope::IsNoLock());
}

TEST_CASE("Visited lists can be returned across lock policy scopes", "[ut][search_no_lock]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    VisitedListPool pool(2, allocator.get(), 128, allocator.get());
    VisitedListPtr first;
    {
        const SearchLockScope scope(true);
        first = pool.TakeOne();
        first->Set(7);
        auto second = pool.TakeOne();
        REQUIRE(first != second);
        REQUIRE_FALSE(second->Get(7));
        pool.ReturnOne(second);
    }
    pool.ReturnOne(first);
    auto locked = pool.TakeOne();
    {
        const SearchLockScope scope(true);
        pool.ReturnOne(locked);
        auto reused = pool.TakeOne();
        REQUIRE_FALSE(reused->Get(7));
        pool.ReturnOne(reused);
    }
}

TEST_CASE("Mixed concurrent queries own distinct reusable visited lists", "[ut][search_no_lock]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    VisitedListPool pool(2, allocator.get(), 128, allocator.get());
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::atomic<int> errors{0};
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 16; ++worker) {
        workers.emplace_back([&, worker] {
            ready.fetch_add(1);
            while (not start.load()) {
            }
            for (int rep = 0; rep < 2000; ++rep) {
                const SearchLockScope scope((worker + rep) % 2 == 0);
                auto list = pool.TakeOne();
                for (int id = 0; id < 16; ++id) {
                    if (list->Get(id)) {
                        errors.fetch_add(1);
                    }
                }
                list->Set(worker);
                std::this_thread::yield();
                if (not list->Get(worker)) {
                    errors.fetch_add(1);
                }
                for (int id = 0; id < 16; ++id) {
                    if (id != worker and list->Get(id)) {
                        errors.fetch_add(1);
                    }
                }
                pool.ReturnOne(list);
            }
        });
    }
    while (ready.load() != 16) {
    }
    start.store(true);
    for (auto& worker : workers) {
        worker.join();
    }
    REQUIRE(errors.load() == 0);
}

TEST_CASE("Query-private mutex guards select an empty lock", "[ut][search_no_lock]") {
    std::mutex mutex;
    {
        auto lock = MakeSearchMutexLock(mutex);
        REQUIRE(lock.owns_lock());
    }
    {
        const SearchLockScope scope(true);
        auto lock = MakeSearchMutexLock(mutex);
        REQUIRE_FALSE(lock.owns_lock());
        REQUIRE(mutex.try_lock());
        mutex.unlock();
    }
    auto lock = MakeSearchMutexLock(mutex);
    REQUIRE(lock.owns_lock());
}
