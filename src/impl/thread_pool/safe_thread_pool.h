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

/**
 * @file safe_thread_pool.h
 * @brief Thread-safe thread pool wrapper with exception handling.
 */

#pragma once

#include "default_thread_pool.h"
#include "impl/logger/logger.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER(SafeThreadPool);

/**
 * @brief Thread-safe thread pool wrapper that provides exception handling.
 *
 * This class wraps a ThreadPool and adds exception safety by catching and
 * logging exceptions thrown by tasks. It supports both owning and non-owning
 * semantics for the underlying thread pool.
 */
class SafeThreadPool : public ThreadPool {
public:
    /**
     * @brief Creates a default safe thread pool with DefaultThreadPool.
     *
     * @return A shared pointer to a new SafeThreadPool instance.
     */
    static std::shared_ptr<SafeThreadPool>
    FactoryDefaultThreadPool() {
        return std::make_shared<SafeThreadPool>(
            new DefaultThreadPool(Options::Instance().num_threads_building()), true);
    }

public:
    /**
     * @brief Constructs a SafeThreadPool with a raw pointer.
     *
     * @param thread_pool Pointer to the underlying thread pool.
     * @param owner Whether this wrapper owns the thread pool.
     */
    SafeThreadPool(ThreadPool* thread_pool, bool owner) : pool_(thread_pool), owner_(owner) {
    }

    /**
     * @brief Constructs a SafeThreadPool with a shared pointer.
     *
     * @param thread_pool Shared pointer to the underlying thread pool.
     */
    SafeThreadPool(const std::shared_ptr<ThreadPool>& thread_pool)
        : pool_ptr_(thread_pool), pool_(thread_pool.get()) {
    }

    /**
     * @brief Destructor that deletes the pool if owned.
     */
    ~SafeThreadPool() override {
        if (owner_) {
            delete pool_;
        }
    }

    /**
     * @brief Enqueues a task with arguments and returns a future.
     *
     * @tparam F The function type.
     * @tparam Args The argument types.
     * @param f The function to execute.
     * @param args The arguments to pass to the function.
     * @return A future for the result.
     */
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

    /**
     * @brief Enqueues a task for execution with exception safety.
     *
     * @param task The task function to execute.
     * @return A future that completes when the task finishes.
     */
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

    /**
     * @brief Waits until all tasks in the queue are completed.
     */
    void
    WaitUntilEmpty() override {
        pool_->WaitUntilEmpty();
    }

    /**
     * @brief Sets the maximum queue size.
     *
     * @param limit The maximum number of tasks in the queue.
     */
    void
    SetQueueSizeLimit(std::uint64_t limit) override {
        pool_->SetQueueSizeLimit(limit);
    }

    /**
     * @brief Sets the number of threads in the pool.
     *
     * @param limit The number of threads.
     */
    void
    SetPoolSize(std::uint64_t limit) override {
        pool_->SetPoolSize(limit);
    }

private:
    /// Raw pointer to the underlying thread pool.
    ThreadPool* pool_{nullptr};
    /// Shared pointer for shared ownership.
    std::shared_ptr<ThreadPool> pool_ptr_{nullptr};
    /// Whether this wrapper owns the pool.
    bool owner_{false};
};

}  // namespace vsag