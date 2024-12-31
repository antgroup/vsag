
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

#include <functional>
#include <future>

namespace vsag {

class ThreadPool {
public:
    /**
      * Blocks until all tasks in the thread pool have completed.
      *
      * This function will wait until all tasks that have been
      * enqueued to the thread pool are finished executing.
      */
    virtual void
    WaitUntilEmpty() = 0;

    /**
      * Sets the limit on the size of the task queue.
      *
      * @param limit The maximum size of the task queue.
      *              Tasks exceeding this size may be rejected or blocked,
      *              depending on the specific implementation.
      */
    virtual void
    SetQueueSizeLimit(std::size_t limit) = 0;

    /**
      * Sets the limit on the size of the thread pool.
      *
      * @param limit The maximum number of worker threads in the pool.
      *              No additional threads will be created beyond this limit.
      */
    virtual void
    SetPoolSize(std::size_t limit) = 0;

    /**
      * Destructor.
      *
      * Cleans up resources used by the thread pool.
      */
    virtual ~ThreadPool() = default;

    /**
      * Enqueues a new task to be executed by the thread pool.
      *
      * @param task A callable object that takes no parameters and
      *             represents the task to be executed.
      * @return std::future<void> A future object representing the
      *                           asynchronous execution of the task,
      *                           which can be used to obtain the task's
      *                           result and status.
      */
    virtual std::future<void>
    Enqueue(std::function<void(void)> task) = 0;
};

}  // namespace vsag
