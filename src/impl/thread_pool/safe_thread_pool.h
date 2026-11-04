
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

#include "default_thread_pool.h"
#include "impl/logger/logger.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER(SafeThreadPool);

class SafeThreadPool : public ThreadPool {
public:
    static std::shared_ptr<SafeThreadPool>
    FactoryDefaultThreadPool() {
        return std::make_shared<SafeThreadPool>(
            new DefaultThreadPool(Options::Instance().num_threads_building()), true);
    }

public:
    SafeThreadPool(ThreadPool* thread_pool, bool owner) : pool_(thread_pool), owner_(owner) {
    }

    SafeThreadPool(const std::shared_ptr<ThreadPool>& thread_pool)
        : pool_ptr_(thread_pool), pool_(thread_pool.get()) {
    }

    ~SafeThreadPool() override {
        if (owner_) {
            delete pool_;
        }
    }

    template <class F, class... Args>
    auto
    GeneralEnqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F&&, Args&&...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<return_type> res = task->get_future();
        Enqueue([task]() { (*task)(); });
        return res;  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
    }

    std::future<void>
    Enqueue(std::function<void(void)> task) override {
        auto func_wrapper = [pool = GetUnderlyingPool(), task = std::move(task)]() {
            PoolThreadScope scope(pool);
            try {
                task();
            } catch (std::exception& e) {
                logger::error("error in thread pool: " + std::string(e.what()));
            }
        };
        return pool_->Enqueue(func_wrapper);
    }
    void
    WaitUntilEmpty() override {
        pool_->WaitUntilEmpty();
    }
    void
    SetQueueSizeLimit(std::uint64_t limit) override {
        pool_->SetQueueSizeLimit(limit);
    }
    void
    SetPoolSize(std::uint64_t limit) override {
        pool_->SetPoolSize(limit);
    }

    // Unlike RequiresInlineExecution(), unknown external pools are not assumed to be current.
    bool
    IsCurrentThread() const {
        const auto* pool = GetUnderlyingPool();
        if (pool == nullptr) {
            return false;
        }
        for (auto* scope = current_scope_; scope != nullptr; scope = scope->previous_) {
            if (scope->pool_ == pool) {
                return true;
            }
        }
        if (const auto* identity = dynamic_cast<const DefaultThreadPool*>(pool)) {
            return identity->IsCurrentThread();
        }
        return false;
    }

    // Unknown external pools may already be occupied by the calling query. Submitting
    // and joining on such a pool can deadlock, including when a bounded queue is full.
    bool
    RequiresInlineExecution() const {
        if (pool_ == nullptr) {
            return true;
        }
        if (const auto* pool = dynamic_cast<const SafeThreadPool*>(pool_)) {
            return pool->RequiresInlineExecution();
        }
        // Even if some workers are idle now, the pool reserves none for this query.
        // Concurrent nested queries could occupy them all before submit/join finishes;
        // execute same-pool queries inline rather than relying on transient spare capacity.
        if (const auto* identity = dynamic_cast<const DefaultThreadPool*>(pool_)) {
            return identity->IsCurrentThread();
        }
        return true;
    }

private:
    const ThreadPool*
    GetUnderlyingPool() const {
        const ThreadPool* pool = pool_;
        while (const auto* wrapper = dynamic_cast<const SafeThreadPool*>(pool)) {
            pool = wrapper->pool_;
        }
        return pool;
    }

    class PoolThreadScope {
    public:
        explicit PoolThreadScope(const ThreadPool* pool) : pool_(pool), previous_(current_scope_) {
            current_scope_ = this;
        }

        ~PoolThreadScope() {
            current_scope_ = previous_;
        }

        const ThreadPool* pool_;
        PoolThreadScope* previous_;
    };

    inline static thread_local PoolThreadScope* current_scope_{nullptr};

    ThreadPool* pool_{nullptr};
    std::shared_ptr<ThreadPool> pool_ptr_{nullptr};
    bool owner_{false};
};

}  // namespace vsag
