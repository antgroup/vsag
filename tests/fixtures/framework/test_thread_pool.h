
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
 * @file test_thread_pool.h
 * @brief Simple thread pool implementation for testing.
 */

#pragma once

#include <functional>
#include <future>
#include <iostream>
#include <queue>
#include <type_traits>

namespace fixtures {

/**
 * @class ThreadPool
 * @brief Basic thread pool for parallel task execution in tests.
 * Supports enqueueing tasks and returns futures for result retrieval.
 */
class ThreadPool {
public:
    /**
     * @brief Constructs a thread pool with the specified number of threads.
     * @param threads Number of worker threads to create.
     */
    ThreadPool(uint64_t threads) : stop(false) {
        for (uint64_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                                             [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    /**
     * @brief Enqueues a task for execution by a worker thread.
     * @tparam F Function type.
     * @tparam Args Argument types.
     * @param f The function to execute.
     * @param args Arguments to pass to the function.
     * @return Future for retrieving the result.
     */
    template <class F, class... Args>
    auto
    enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    /**
     * @brief Destructor that stops all threads and waits for completion.
     */
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) worker.join();
    }

private:
    std::vector<std::thread> workers;         // Worker threads.
    std::queue<std::function<void()>> tasks;  // Queue of pending tasks.
    std::mutex queue_mutex;                   // Mutex for task queue access.
    std::condition_variable condition;        // Condition variable for thread synchronization.
    bool stop;                                // Flag to signal thread termination.
};
}  // namespace fixtures
