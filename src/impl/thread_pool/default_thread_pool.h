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
 * @file default_thread_pool.h
 * @brief Default thread pool implementation.
 */

#pragma once

#include <ThreadPool.h>

#include <functional>
#include <future>

#include "vsag/thread_pool.h"

namespace vsag {

/**
 * @brief Default thread pool implementation using progschj/ThreadPool.
 *
 * This class wraps the third-party ThreadPool library to provide
 * thread pool functionality for the vsag library.
 */
class DefaultThreadPool : public ThreadPool {
public:
    /**
     * @brief Constructs a DefaultThreadPool with the specified number of threads.
     *
     * @param threads The number of worker threads to create.
     */
    explicit DefaultThreadPool(std::uint64_t threads);

    /**
     * @brief Enqueues a task for execution.
     *
     * @param task The task function to execute.
     * @return A future that completes when the task finishes.
     */
    std::future<void>
    Enqueue(std::function<void(void)> task) override;

    /**
     * @brief Waits until all tasks in the queue are completed.
     */
    void
    WaitUntilEmpty() override;

    /**
     * @brief Sets the maximum queue size.
     *
     * @param limit The maximum number of tasks in the queue.
     */
    void
    SetQueueSizeLimit(std::uint64_t limit) override;

    /**
     * @brief Sets the number of threads in the pool.
     *
     * @param limit The number of threads.
     */
    void
    SetPoolSize(std::uint64_t limit) override;

private:
    /// Pointer to the underlying thread pool implementation.
    std::unique_ptr<progschj::ThreadPool> pool_;
};

}  // namespace vsag