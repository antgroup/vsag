
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

#include "ring_buffer_thread_pool.h"

#include <algorithm>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace vsag;

// ---------------------------------------------------------------------------
// Helper: a promise-based gate so a test can hold a callback mid-execution
// without busy-waiting, then release it when ready.
// ---------------------------------------------------------------------------
struct PromiseGate {
    std::promise<void> enter;
    std::promise<void> leave;

    // Shared state so the test can call .get_future() before enqueuing.
    std::shared_future<void> entered = enter.get_future().share();
    std::shared_future<void> exit = leave.get_future().share();
};

// ---------------------------------------------------------------------------
// Invalid construction
// ---------------------------------------------------------------------------
TEST_CASE("RingBufferThreadPool rejects zero threads", "[ut][RingBufferThreadPool]") {
    REQUIRE_THROWS_AS(RingBufferThreadPool(0), std::invalid_argument);
}

TEST_CASE("RingBufferThreadPool rejects non-positive queue capacity",
          "[ut][RingBufferThreadPool]") {
    REQUIRE_THROWS_AS(RingBufferThreadPool(1, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(RingBufferThreadPool(1, -1), std::invalid_argument);
}

TEST_CASE("RingBufferThreadPool rejects non-power-of-two queue capacity",
          "[ut][RingBufferThreadPool]") {
    REQUIRE_THROWS_AS(RingBufferThreadPool(1, 3), std::invalid_argument);
    REQUIRE_THROWS_AS(RingBufferThreadPool(1, 100), std::invalid_argument);
}

TEST_CASE("RingBufferThreadPool accepts valid constructor args", "[ut][RingBufferThreadPool]") {
    REQUIRE_NOTHROW(RingBufferThreadPool(1));
    REQUIRE_NOTHROW(RingBufferThreadPool(1, 16));
    REQUIRE_NOTHROW(RingBufferThreadPool(4, 64));
}

// ---------------------------------------------------------------------------
// Exactly-once counting: each ID must be incremented exactly once across
// multiple producers, verifying no slot is skipped or double-executed.
// ---------------------------------------------------------------------------
TEST_CASE("RingBufferThreadPool exactly once per-ID with capacity 1",
          "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(4, 1);
    constexpr int kIDs = 200;
    std::vector<std::atomic<int>> counters(kIDs);
    for (auto& counter : counters) {
        counter.store(0);
    }

    auto producer = [&](int parity) {
        for (int i = parity; i < kIDs; i += 2) {
            pool.Enqueue([&counters, i]() { counters[i].fetch_add(1); });
        }
    };

    std::thread t1(producer, 0);
    std::thread t2(producer, 1);
    t1.join();
    t2.join();
    pool.WaitUntilEmpty();

    for (int i = 0; i < kIDs; ++i) {
        INFO("ID " << i << " was incremented " << counters[i].load() << " times, expected 1");
        REQUIRE(counters[i].load() == 1);
    }
}

TEST_CASE("RingBufferThreadPool exactly once per-ID with capacity 2",
          "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(4, 2);
    constexpr int kIDs = 500;
    std::vector<std::atomic<int>> counters(kIDs);
    for (auto& counter : counters) {
        counter.store(0);
    }

    auto producer = [&](int parity) {
        for (int i = parity; i < kIDs; i += 2) {
            pool.Enqueue([&counters, i]() { counters[i].fetch_add(1); });
        }
    };

    std::thread t1(producer, 0);
    std::thread t2(producer, 1);
    t1.join();
    t2.join();
    pool.WaitUntilEmpty();

    for (int i = 0; i < kIDs; ++i) {
        INFO("ID " << i << " was incremented " << counters[i].load() << " times, expected 1");
        REQUIRE(counters[i].load() == 1);
    }
}

TEST_CASE("RingBufferThreadPool exactly once default capacity multi-producer",
          "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(8);
    std::atomic<int64_t> counter{0};
    constexpr int kTasksPerProducer = 1000;
    constexpr int kProducers = 4;

    std::vector<std::thread> producers;
    for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&pool, &counter]() {
            for (int i = 0; i < kTasksPerProducer; ++i) {
                pool.Enqueue([&counter]() { counter.fetch_add(1); });
            }
        });
    }
    for (auto& t : producers) {
        t.join();
    }
    pool.WaitUntilEmpty();
    REQUIRE(counter == kTasksPerProducer * kProducers);
}

// ---------------------------------------------------------------------------
// GeneralEnqueue with mixed return types
// ---------------------------------------------------------------------------
TEST_CASE("RingBufferThreadPool GeneralEnqueue return value", "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(4);

    auto fut_int = pool.GeneralEnqueue([]() -> int { return 42; });
    REQUIRE(fut_int.get() == 42);

    auto fut_double = pool.GeneralEnqueue([](double x) -> double { return x * 2; }, 3.5);
    REQUIRE(fut_double.get() == 7.0);

    auto fut_void = pool.GeneralEnqueue([]() {});
    fut_void.get();
}

// ---------------------------------------------------------------------------
// Nested GeneralEnqueue: a worker can enqueue more work inline and the
// in-flight counter / WaitUntilEmpty still works correctly.
// ---------------------------------------------------------------------------
TEST_CASE("RingBufferThreadPool nested GeneralEnqueue inline", "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(4);
    std::atomic<int> outer{0};
    std::atomic<int> inner{0};

    pool.Enqueue([&]() {
        outer.fetch_add(1);
        // Enqueue inner work from within a worker — must not deadlock or
        // corrupt the in-flight counter.
        pool.Enqueue([&]() { inner.fetch_add(1); });
    });

    pool.WaitUntilEmpty();
    REQUIRE(outer.load() == 1);
    REQUIRE(inner.load() == 1);
}

// ---------------------------------------------------------------------------
// WaitUntilEmpty must wait for execution completion, not just enqueue.
// Defect: REQUIRE_FALSE in the waiter thread before release/join could
//         cause std::terminate.  Record the observation, then release,
//         join, and assert.
// ---------------------------------------------------------------------------
TEST_CASE("RingBufferThreadPool WaitUntilEmpty waits for execution completion",
          "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(2);
    PromiseGate gate;

    std::atomic<bool> started{false};
    std::atomic<bool> finished{false};
    pool.Enqueue([&]() {
        started.store(true);
        gate.exit.wait();
        finished.store(true);
    });

    // Wait for the task to actually start executing.
    while (!started.load()) {
        std::this_thread::yield();
    }

    // WaitUntilEmpty must block while a task is mid-execution.
    std::atomic<bool> empty_returned{false};
    std::thread waiter([&]() {
        pool.WaitUntilEmpty();
        empty_returned.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Record observation BEFORE releasing the gate or joining.
    bool blocked_as_expected = !empty_returned.load();

    gate.leave.set_value();
    waiter.join();

    bool returned_after_release = empty_returned.load();
    bool task_finished = finished.load();

    REQUIRE(blocked_as_expected);
    REQUIRE(returned_after_release);
    REQUIRE(task_finished);
}

// ---------------------------------------------------------------------------
// WaitUntilEmpty with promise-based gate for new-submission-during-drain
// ---------------------------------------------------------------------------
TEST_CASE("RingBufferThreadPool WaitUntilEmpty re-waits after new submission",
          "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(4);

    for (int round = 0; round < 5; ++round) {
        const int kTasks = 50;
        std::atomic<int> counter{0};
        for (int i = 0; i < kTasks; ++i) {
            pool.Enqueue([&counter]() { counter.fetch_add(1); });
        }
        pool.WaitUntilEmpty();
        REQUIRE(counter == kTasks);

        // Submit more work after empty — WaitUntilEmpty must re-block.
        counter = 0;
        for (int i = 0; i < kTasks; ++i) {
            pool.Enqueue([&counter]() { counter.fetch_add(1); });
        }
        pool.WaitUntilEmpty();
        REQUIRE(counter == kTasks);
    }
}

// ---------------------------------------------------------------------------
// Idle wakeup: a sleeping worker must wake when work is enqueued
// ---------------------------------------------------------------------------
TEST_CASE("RingBufferThreadPool idle worker wakes up on enqueue", "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(1);
    // Let the single worker settle into idle.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::atomic<bool> done{false};
    auto start = std::chrono::steady_clock::now();
    pool.Enqueue([&done]() { done.store(true); });
    pool.WaitUntilEmpty();
    auto elapsed = std::chrono::steady_clock::now() - start;
    REQUIRE(done.load());
    // Must complete well under 500 ms — a stuck idle poll would take much longer.
    REQUIRE(elapsed < std::chrono::milliseconds(500));
}

// ---------------------------------------------------------------------------
// Destructor drain: in-flight tasks must complete before dtor returns.
//   - Gate must outlive the pool (task references gate.exit).
//   - Task must signal gate.enter when it starts, so the test can gate.entered.wait().
//   - An independent thread deletes the pool; the destructor must block on
//     the in-flight worker.  Verify it hasn't returned before release, then
//     release + join + assert.
// ---------------------------------------------------------------------------
TEST_CASE("RingBufferThreadPool destructor drains in-flight tasks", "[ut][RingBufferThreadPool]") {
    // Gate declared FIRST — outlives the heap pool.
    PromiseGate gate;
    std::atomic<bool> task_finished{false};

    auto* raw_pool = new RingBufferThreadPool(2);
    raw_pool->Enqueue([&]() {
        gate.enter.set_value();  // signal "I have started"
        gate.exit.wait();        // block until test releases
        task_finished.store(true);
    });

    // Wait until the task has definitely started executing.
    gate.entered.wait();

    // Independent thread deletes the pool.  The destructor blocks on
    // joining the worker, which is blocked on gate.exit.
    std::atomic<bool> destroy_returned{false};
    std::thread destroyer([raw_pool, &destroy_returned]() {
        delete raw_pool;
        destroy_returned.store(true);
    });

    // Give the destroyer a moment — it should still be blocked.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    bool destroy_blocked = !destroy_returned.load();

    // Release the gate so the task finishes and the destructor can complete.
    gate.leave.set_value();
    destroyer.join();

    bool destroy_finished = destroy_returned.load();
    bool task_done = task_finished.load();

    REQUIRE(destroy_blocked);
    REQUIRE(destroy_finished);
    REQUIRE(task_done);
}

// ---------------------------------------------------------------------------
// Resize: grow
// ---------------------------------------------------------------------------
TEST_CASE("RingBufferThreadPool grow pool", "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(1);
    pool.SetPoolSize(4);

    std::atomic<int> counter{0};
    constexpr int kTasks = 200;

    std::vector<std::future<void>> futures;
    for (int i = 0; i < kTasks; ++i) {
        futures.push_back(pool.Enqueue([&counter]() { counter.fetch_add(1); }));
    }
    pool.WaitUntilEmpty();
    REQUIRE(counter == kTasks);
}

// ---------------------------------------------------------------------------
// Resize: shrink
// ---------------------------------------------------------------------------
TEST_CASE("RingBufferThreadPool shrink pool", "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(4);
    pool.SetPoolSize(1);

    std::atomic<int> counter{0};
    constexpr int kTasks = 200;

    for (int i = 0; i < kTasks; ++i) {
        pool.Enqueue([&counter]() { counter.fetch_add(1); });
    }
    pool.WaitUntilEmpty();
    REQUIRE(counter == kTasks);
}

// ---------------------------------------------------------------------------
// Resize: grow then shrink
// ---------------------------------------------------------------------------
TEST_CASE("RingBufferThreadPool grow then shrink", "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(1);

    // Grow to 8
    pool.SetPoolSize(8);
    {
        std::atomic<int> counter{0};
        for (int i = 0; i < 100; ++i) {
            pool.Enqueue([&counter]() { counter.fetch_add(1); });
        }
        pool.WaitUntilEmpty();
        REQUIRE(counter == 100);
    }

    // Shrink to 1
    pool.SetPoolSize(1);
    {
        std::atomic<int> counter{0};
        for (int i = 0; i < 100; ++i) {
            pool.Enqueue([&counter]() { counter.fetch_add(1); });
        }
        pool.WaitUntilEmpty();
        REQUIRE(counter == 100);
    }

    // Grow back to 4
    pool.SetPoolSize(4);
    {
        std::atomic<int> counter{0};
        for (int i = 0; i < 100; ++i) {
            pool.Enqueue([&counter]() { counter.fetch_add(1); });
        }
        pool.WaitUntilEmpty();
        REQUIRE(counter == 100);
    }
}

// ---------------------------------------------------------------------------
// WaitUntilEmpty / SetPoolSize called from worker throws logic_error
// ---------------------------------------------------------------------------
TEST_CASE("RingBufferThreadPool WaitUntilEmpty from worker throws", "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(2);
    std::atomic<bool> caught{false};

    auto fut = pool.Enqueue([&]() {
        try {
            pool.WaitUntilEmpty();
        } catch (const std::logic_error&) {
            caught.store(true);
        }
    });
    fut.get();
    REQUIRE(caught.load());
}

TEST_CASE("RingBufferThreadPool SetPoolSize from worker throws", "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(2);
    std::atomic<bool> caught{false};

    auto fut = pool.Enqueue([&]() {
        try {
            pool.SetPoolSize(4);
        } catch (const std::logic_error&) {
            caught.store(true);
        }
    });
    fut.get();
    REQUIRE(caught.load());
}

// ---------------------------------------------------------------------------
// SetQueueSizeLimit
// ---------------------------------------------------------------------------
TEST_CASE("RingBufferThreadPool SetQueueSizeLimit applies a smaller logical limit",
          "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(1, 16);
    pool.SetQueueSizeLimit(2);
    PromiseGate gate;
    auto running = pool.Enqueue([&] {
        gate.enter.set_value();
        gate.exit.wait();
    });
    gate.entered.wait();
    auto first = pool.Enqueue([] {});
    auto second = pool.Enqueue([] {});
    std::promise<void> entered;
    auto entered_future = entered.get_future();
    auto producer = std::async(std::launch::async, [&] {
        entered.set_value();
        return pool.Enqueue([] {});
    });
    entered_future.wait();
    const auto before = producer.wait_for(std::chrono::milliseconds(100));
    gate.leave.set_value();
    const auto after = producer.wait_for(std::chrono::seconds(2));
    running.get();
    first.get();
    second.get();
    auto submitted = producer.get();
    submitted.get();
    pool.WaitUntilEmpty();
    REQUIRE(before == std::future_status::timeout);
    REQUIRE(after == std::future_status::ready);
}

TEST_CASE("RingBufferThreadPool SetQueueSizeLimit zero clamped to one",
          "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(4, 16);
    pool.SetQueueSizeLimit(0);

    std::atomic<int> counter{0};
    for (int i = 0; i < 50; ++i) {
        pool.Enqueue([&counter]() { counter.fetch_add(1); });
    }
    pool.WaitUntilEmpty();
    REQUIRE(counter == 50);
}

TEST_CASE("RingBufferThreadPool wakes a producer when the limit increases",
          "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(1, 4);
    pool.SetQueueSizeLimit(1);
    PromiseGate gate;
    auto running = pool.Enqueue([&] {
        gate.enter.set_value();
        gate.exit.wait();
    });
    gate.entered.wait();
    auto queued = pool.Enqueue([] {});
    std::promise<void> entered;
    auto entered_future = entered.get_future();
    auto producer = std::async(std::launch::async, [&] {
        entered.set_value();
        return pool.Enqueue([] {});
    });
    entered_future.wait();
    auto before = producer.wait_for(std::chrono::milliseconds(100));
    pool.SetQueueSizeLimit(2);
    auto after = producer.wait_for(std::chrono::seconds(2));
    gate.leave.set_value();
    running.get();
    queued.get();
    auto submitted = producer.get();
    submitted.get();
    pool.WaitUntilEmpty();
    REQUIRE(before == std::future_status::timeout);
    REQUIRE(after == std::future_status::ready);
}

TEST_CASE("RingBufferThreadPool wakes a producer on slot reclamation",
          "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(1, 1);
    PromiseGate gate;
    auto running = pool.Enqueue([&] {
        gate.enter.set_value();
        gate.exit.wait();
    });
    gate.entered.wait();
    auto queued = pool.Enqueue([] {});
    std::promise<void> entered;
    auto entered_future = entered.get_future();
    auto producer = std::async(std::launch::async, [&] {
        entered.set_value();
        return pool.Enqueue([] {});
    });
    entered_future.wait();
    auto before = producer.wait_for(std::chrono::milliseconds(100));
    gate.leave.set_value();
    auto after = producer.wait_for(std::chrono::seconds(2));
    running.get();
    queued.get();
    auto submitted = producer.get();
    submitted.get();
    pool.WaitUntilEmpty();
    REQUIRE(before == std::future_status::timeout);
    REQUIRE(after == std::future_status::ready);
}

// ---------------------------------------------------------------------------
// Exception propagation through futures
// ---------------------------------------------------------------------------
TEST_CASE("RingBufferThreadPool exceptions propagate through futures",
          "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(2);

    auto fut = pool.Enqueue([]() { throw std::runtime_error("test error"); });
    REQUIRE_THROWS_AS(fut.get(), std::runtime_error);

    auto fut_int = pool.GeneralEnqueue([]() -> int { throw std::logic_error("logic"); });
    REQUIRE_THROWS_AS(fut_int.get(), std::logic_error);
}

// ---------------------------------------------------------------------------
// Exception does not break subsequent tasks
// ---------------------------------------------------------------------------
TEST_CASE("RingBufferThreadPool exception does not break subsequent tasks",
          "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(2);
    std::atomic<int> counter{0};

    pool.Enqueue([]() { throw std::runtime_error("boom"); });
    pool.Enqueue([&counter]() { counter.fetch_add(1); });
    pool.Enqueue([&counter]() { counter.fetch_add(1); });

    pool.WaitUntilEmpty();
    REQUIRE(counter == 2);
}

// ---------------------------------------------------------------------------
// Enqueue return type: std::future<void>
// ---------------------------------------------------------------------------
TEST_CASE("RingBufferThreadPool Enqueue returns valid future", "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(2);

    std::atomic<bool> called{false};
    auto fut = pool.Enqueue([&called]() { called.store(true); });
    REQUIRE(fut.valid());
    fut.get();
    REQUIRE(called.load());
}

// ---------------------------------------------------------------------------
// SetPoolSize with zero/negative is a no-op
// ---------------------------------------------------------------------------
TEST_CASE("RingBufferThreadPool SetPoolSize zero does not crash", "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(4);
    pool.SetPoolSize(0);

    std::atomic<int> counter{0};
    for (int i = 0; i < 50; ++i) {
        pool.Enqueue([&counter]() { counter.fetch_add(1); });
    }
    pool.WaitUntilEmpty();
    REQUIRE(counter == 50);
}

// ---------------------------------------------------------------------------
// Large task count stress test
// ---------------------------------------------------------------------------
TEST_CASE("RingBufferThreadPool stress many tasks", "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(8);
    std::atomic<int64_t> counter{0};
    constexpr int kTasks = 10000;

    for (int i = 0; i < kTasks; ++i) {
        pool.Enqueue([&counter]() { counter.fetch_add(1); });
    }
    pool.WaitUntilEmpty();
    REQUIRE(counter == kTasks);
}

// ---------------------------------------------------------------------------
// IsCurrentThread identity
// ---------------------------------------------------------------------------
TEST_CASE("RingBufferThreadPool IsCurrentThread", "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(2);

    // Main thread is NOT a pool thread.
    REQUIRE_FALSE(pool.IsCurrentThread());

    // A task running on the pool IS on a pool thread.
    bool is_current = false;
    pool.Enqueue([&]() { is_current = pool.IsCurrentThread(); }).get();
    REQUIRE(is_current);

    // After the task, main thread is still NOT a pool thread.
    REQUIRE_FALSE(pool.IsCurrentThread());
}

TEST_CASE("RingBufferThreadPool default logical limit exceeds physical ring",
          "[ut][RingBufferThreadPool]") {
    RingBufferThreadPool pool(1);
    PromiseGate gate;
    auto running = pool.Enqueue([&] {
        gate.enter.set_value();
        gate.exit.wait();
    });
    gate.entered.wait();
    std::vector<uint64_t> order;
    order.reserve(100002);
    for (uint64_t i = 0; i < 100000; ++i) {
        pool.Enqueue([&, i] { order.push_back(i); });
    }
    std::promise<void> attempting;
    auto entered = attempting.get_future();
    auto producer = std::async(std::launch::async, [&] {
        attempting.set_value();
        return pool.Enqueue([&] { order.push_back(100000); });
    });
    entered.wait();
    auto blocked = producer.wait_for(std::chrono::milliseconds(100));
    pool.SetQueueSizeLimit(std::numeric_limits<uint64_t>::max());
    auto admitted = producer.wait_for(std::chrono::seconds(2));
    // Always release callbacks before assertions so a failure cannot strand joins.
    gate.leave.set_value();
    running.get();
    producer.get().get();
    pool.Enqueue([&] { order.push_back(100001); }).get();
    pool.WaitUntilEmpty();
    REQUIRE(blocked == std::future_status::timeout);
    REQUIRE(admitted == std::future_status::ready);
    REQUIRE(order.size() == 100002);
    for (uint64_t i = 0; i < order.size(); ++i) {
        REQUIRE(order[i] == i);
    }
}

TEST_CASE("RingBufferThreadPool overflow drains during destruction", "[ut][RingBufferThreadPool]") {
    std::atomic<uint64_t> completed{0};
    {
        RingBufferThreadPool pool(1, 2);
        pool.SetQueueSizeLimit(1024);
        PromiseGate gate;
        auto running = pool.Enqueue([&] {
            gate.enter.set_value();
            gate.exit.wait();
        });
        gate.entered.wait();
        for (uint64_t i = 0; i < 1000; ++i) {
            pool.Enqueue([&] { completed.fetch_add(1); });
        }
        gate.leave.set_value();
        running.get();
    }
    REQUIRE(completed == 1000);
}
