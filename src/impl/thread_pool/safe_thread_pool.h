
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
        auto func_wrapper = [task = std::move(task)]() {
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
    SetQueueSizeLimit(std::size_t limit) override {
        pool_->SetQueueSizeLimit(limit);
    }
    void
    SetPoolSize(std::size_t limit) override {
        pool_->SetPoolSize(limit);
    }

private:
    ThreadPool* pool_{nullptr};
    std::shared_ptr<ThreadPool> pool_ptr_{nullptr};
    bool owner_{false};
};

}  // namespace vsag
