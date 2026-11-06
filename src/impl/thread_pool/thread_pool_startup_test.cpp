// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "thread_pool_startup.h"

#include <atomic>
#include <limits>
#include <set>
#include <stdexcept>
#include <thread>

#include "default_thread_pool.h"
#include "safe_thread_pool.h"
#include "unittest.h"
using namespace vsag;

namespace {
class StartupProbePool final : public ThreadPool {
public:
    StartupProbePool() : pool(4) {
    }
    std::future<void>
    Enqueue(std::function<void()> task) override {
        if (submitted++ == reject_after)
            throw std::runtime_error("injected startup submit failure");
        return pool.Enqueue([this, task = std::move(task)] {
            {
                std::lock_guard<std::mutex> guard(mutex);
                threads.insert(std::this_thread::get_id());
            }
            ++active;
            task();
            --active;
        });
    }
    void
    WaitUntilEmpty() override {
        pool.WaitUntilEmpty();
    }
    void
    SetQueueSizeLimit(uint64_t value) override {
        pool.SetQueueSizeLimit(value);
    }
    void
    SetPoolSize(uint64_t value) override {
        pool.SetPoolSize(value);
    }
    DefaultThreadPool pool;
    uint32_t submitted{0};
    uint32_t reject_after{2};
    std::atomic<uint32_t> active{0};
    std::mutex mutex;
    std::set<std::thread::id> threads;
};
}  // namespace

TEST_CASE("Thread-pool startup confirms every worker and drains failed submissions",
          "[ut][ThreadPoolStartup]") {
    auto probe = std::make_shared<StartupProbePool>();
    auto pool = std::make_shared<SafeThreadPool>(probe);
    REQUIRE_THROWS(ConfirmThreadPoolWorkers(*pool, 4));
    probe->WaitUntilEmpty();
    REQUIRE(probe->active == 0);
    REQUIRE(probe->threads.size() <= 2);
    probe->threads.clear();
    probe->submitted = 0;
    probe->reject_after = std::numeric_limits<uint32_t>::max();
    REQUIRE_NOTHROW(ConfirmThreadPoolWorkers(*pool, 4));
    probe->WaitUntilEmpty();
    REQUIRE(probe->threads.size() == 4);
    REQUIRE(probe->active == 0);
    REQUIRE_NOTHROW(ConfirmThreadPoolWorkers(*pool, 0));
}
