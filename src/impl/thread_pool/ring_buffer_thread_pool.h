
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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "vsag/thread_pool.h"

namespace vsag {

class RingBufferThreadPool : public ThreadPool {
public:
    static constexpr int kDefaultQueueCapacity = 4096;

    explicit RingBufferThreadPool(uint64_t num_threads, int queue_capacity = kDefaultQueueCapacity);

    ~RingBufferThreadPool() override;

    // ThreadPool interface
    std::future<void>
    Enqueue(std::function<void(void)> task) override;

    void
    WaitUntilEmpty() override;

    void
    SetPoolSize(uint64_t limit) override;

    void
    SetQueueSizeLimit(uint64_t limit) override;

    // GeneralEnqueue — primary interface used by VSAG call sites
    template <class F, class... Args>
    auto
    GeneralEnqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    // Thread identity
    bool
    IsCurrentThread() const;

    // Thread affinity
    void
    SetAffinity(const std::vector<int>& cpu_ids);

private:
    void
    worker_loop(int worker_id);

    // Ring buffer
    std::vector<std::function<void()>> task_buffer_;
    const int queue_capacity_;
    alignas(64) std::atomic<uint64_t> write_idx_{0};
    alignas(64) std::atomic<uint64_t> read_idx_{0};
    std::mutex write_mutex_;

    // Worker state
    struct Worker {
        std::thread thread;
        int index;
    };
    std::vector<std::unique_ptr<Worker>> workers_;
    std::atomic<int> worker_count_{0};

    // Synchronization
    std::mutex sleep_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
    std::atomic<size_t> in_flight_{0};
    std::condition_variable empty_cv_;

    // Queue limit
    std::atomic<uint64_t> max_queue_size_{static_cast<uint64_t>(kDefaultQueueCapacity)};
};

// --- Template implementation ---

template <class F, class... Args>
auto
RingBufferThreadPool::GeneralEnqueue(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();

    in_flight_.fetch_add(1, std::memory_order_relaxed);

    // PRODUCER: acquire write_mutex_ then CAS the write index
    for (;;) {
        {
            std::lock_guard<std::mutex> lock(write_mutex_);
            uint64_t w = write_idx_.load(std::memory_order_relaxed);
            uint64_t r = read_idx_.load(std::memory_order_acquire);
            uint64_t effective_cap = static_cast<uint64_t>(std::min(static_cast<uint64_t>(queue_capacity_),
                                         max_queue_size_.load(std::memory_order_relaxed)));
            if (w - r < effective_cap) {
                // Construct the lambda BEFORE updating write_idx_, so the
                // consumer always sees a fully-written slot.  If the lambda
                // allocation throws std::bad_alloc, we must roll back the
                // in_flight counter, because no task was actually published.
                try {
                    task_buffer_[(w & static_cast<uint64_t>(queue_capacity_ - 1))] =
                        [task]() { (*task)(); };
                } catch (...) {
                    in_flight_.fetch_sub(1, std::memory_order_release);
                    throw;
                }
                write_idx_.store(w + 1, std::memory_order_release);
                break;
            }
        }
        cv_.notify_one();
        std::this_thread::yield();
    }
    cv_.notify_one();
    return res;  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
}

}  // namespace vsag
