
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

#include "safe_thread_pool.h"

#include "unittest.h"
TEST_CASE("SafeThreadPool Basic Test", "[ut][SafeThreadPool]") {
    auto thread_pool = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    int data = 0;
    std::mutex m;
    thread_pool->SetPoolSize(4);
    thread_pool->SetQueueSizeLimit(6);
    int round = 10;
    for (int i = 0; i < round; ++i) {
        thread_pool->GeneralEnqueue(
            [&data, &m](int i) -> int {
                std::lock_guard lock(m);
                vsag::logger::info("current data:{}", data);
                data++;
                return i * i;
            },
            i);
        thread_pool->GeneralEnqueue(
            []() { throw std::runtime_error("throw a error in thread pool"); });
    }
    thread_pool->WaitUntilEmpty();
    REQUIRE(data == round);
}

namespace {

// Hide the default pool's identity behind the public custom-pool interface.
class CustomThreadPool : public vsag::ThreadPool {
public:
    std::future<void>
    Enqueue(std::function<void(void)> task) override {
        return pool_.Enqueue(std::move(task));
    }

    void
    WaitUntilEmpty() override {
        pool_.WaitUntilEmpty();
    }

    void
    SetQueueSizeLimit(std::uint64_t limit) override {
        pool_.SetQueueSizeLimit(limit);
    }

    void
    SetPoolSize(std::uint64_t limit) override {
        pool_.SetPoolSize(limit);
    }

private:
    vsag::DefaultThreadPool pool_{1};
};

class InlineThreadPool : public CustomThreadPool {
public:
    std::future<void>
    Enqueue(std::function<void(void)> task) override {
        std::packaged_task<void()> packaged_task(std::move(task));
        auto result = packaged_task.get_future();
        packaged_task();
        return result;
    }
};

}  // namespace

TEST_CASE("SafeThreadPool Custom Worker Identity", "[ut][SafeThreadPool]") {
    CustomThreadPool custom;
    CustomThreadPool other_custom;
    vsag::SafeThreadPool pool(&custom, false);
    vsag::SafeThreadPool other_pool(&other_custom, false);
    vsag::SafeThreadPool nested(&pool, false);
    vsag::SafeThreadPool sibling(&custom, false);

    // An external custom pool must not force callers of IsCurrentThread() to run inline.
    REQUIRE_FALSE(pool.IsCurrentThread());
    REQUIRE_FALSE(nested.IsCurrentThread());
    REQUIRE(pool.RequiresInlineExecution());
    REQUIRE(nested.RequiresInlineExecution());

    bool raw_worker_is_current = true;
    custom.Enqueue([&]() { raw_worker_is_current = pool.IsCurrentThread(); }).get();
    REQUIRE_FALSE(raw_worker_is_current);

    auto check_identity = [&]() {
        return pool.IsCurrentThread() && nested.IsCurrentThread() && sibling.IsCurrentThread() &&
               not other_pool.IsCurrentThread();
    };
    REQUIRE(pool.GeneralEnqueue(check_identity).get());
    REQUIRE(nested.GeneralEnqueue(check_identity).get());
    REQUIRE(sibling.GeneralEnqueue(check_identity).get());
    REQUIRE_FALSE(pool.IsCurrentThread());
}

TEST_CASE("SafeThreadPool Native Default Worker Identity", "[ut][SafeThreadPool]") {
    vsag::DefaultThreadPool native(1);
    vsag::DefaultThreadPool other_native(1);
    vsag::SafeThreadPool pool(&native, false);
    vsag::SafeThreadPool nested(&pool, false);
    vsag::SafeThreadPool other_pool(&other_native, false);
    vsag::SafeThreadPool empty(nullptr, false);

    REQUIRE_FALSE(pool.IsCurrentThread());
    REQUIRE_FALSE(pool.RequiresInlineExecution());
    REQUIRE_FALSE(empty.IsCurrentThread());
    bool is_current = false;
    native
        .Enqueue([&]() {
            is_current = pool.IsCurrentThread() && nested.IsCurrentThread() &&
                         not other_pool.IsCurrentThread() && pool.RequiresInlineExecution();
        })
        .get();
    REQUIRE(is_current);
}

TEST_CASE("SafeThreadPool Worker Identity Exception Restoration", "[ut][SafeThreadPool]") {
    CustomThreadPool custom;
    InlineThreadPool inline_custom;
    vsag::SafeThreadPool pool(&custom, false);
    vsag::SafeThreadPool inline_pool(&inline_custom, false);

    bool inside_inline = false;
    bool restored_outer = false;
    pool.Enqueue([&]() {
            inline_pool
                .Enqueue([&]() {
                    // A synchronous inner pool must not hide the occupied outer worker.
                    inside_inline = inline_pool.IsCurrentThread() && pool.IsCurrentThread();
                    throw std::runtime_error("restore safe pool identity");
                })
                .get();
            restored_outer = pool.IsCurrentThread() && not inline_pool.IsCurrentThread();
            throw std::runtime_error("restore custom worker identity");
        })
        .get();
    REQUIRE(inside_inline);
    REQUIRE(restored_outer);

    bool restored_worker = false;
    custom
        .Enqueue([&]() {
            restored_worker = not pool.IsCurrentThread() && not inline_pool.IsCurrentThread();
        })
        .get();
    REQUIRE(restored_worker);

    // Non-standard exceptions bypass SafeThreadPool's catch but must still unwind the scope.
    auto failed = inline_pool.Enqueue([]() { throw 42; });
    REQUIRE_THROWS_AS(failed.get(), int);
    REQUIRE_FALSE(inline_pool.IsCurrentThread());
    REQUIRE_FALSE(pool.IsCurrentThread());
}
