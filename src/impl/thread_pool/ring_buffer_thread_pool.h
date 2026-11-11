
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

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "vsag/thread_pool.h"

namespace vsag {

// Ring plus on-demand FIFO overflow, serialized publication and CAS ring consumers.
// This is not a lock-free pool: publication and parking share a mutex.
// Completion uses an atomic counter and a separate mutex for zero-transition wakeups.
// Nested submissions execute inline; worker-side waits and management are rejected.
class RingBufferThreadPool : public ThreadPool {
public:
    static constexpr int kDefaultQueueCapacity = 4096;

    static constexpr uint64_t kDefaultQueueSizeLimit = 100000;

    // Default construction separates physical capacity (4096) from logical limit (100000).
    explicit RingBufferThreadPool(uint64_t num_threads);

    // Explicit capacities retain their initial logical limit for small-capacity tests.
    RingBufferThreadPool(uint64_t num_threads, int queue_capacity);

    ~RingBufferThreadPool() override;

    // ThreadPool interface
    std::future<void>
    Enqueue(std::function<void(void)> task) override;

    void
    WaitUntilEmpty() override;

    void
    SetPoolSize(uint64_t limit) override;

    // Logical queued-task limit, independent of physical capacity; 0 selects 1.
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

private:
    void
    worker_loop(int worker_id);

    // Ring buffer
    struct Slot {
        std::atomic<bool> ready{false};
        std::function<void()> task;
    };
    std::unique_ptr<Slot[]> task_buffer_;
    const int queue_capacity_;
    alignas(64) std::atomic<uint64_t> write_idx_{0};
    alignas(64) std::atomic<uint64_t> read_idx_{0};
    std::mutex write_mutex_;
    // Accessed only under write_mutex_; producers never bypass an existing backlog.
    std::deque<std::function<void()>> overflow_;

    // Worker state
    struct Worker {
        std::thread thread;
        int index;
    };
    std::vector<std::unique_ptr<Worker>> workers_;
    std::atomic<int> worker_count_{0};

    // Synchronization
    std::mutex management_mutex_;
    std::condition_variable cv_;
    std::condition_variable not_full_cv_;
    std::atomic<bool> stop_{false};
    alignas(64) std::atomic<uint64_t> in_flight_{0};
    std::mutex completion_mutex_;
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

    // Nested submissions must not spin on a queue that only this pool can drain.
    if (IsCurrentThread()) {
        (*task)();
        return res;
    }

    // Reclamation and limit changes use the same mutex as this wait.
    std::unique_lock<std::mutex> lock(write_mutex_);
    not_full_cv_.wait(lock, [this] {
        if (stop_.load(std::memory_order_relaxed)) {
            return true;
        }
        const uint64_t w = write_idx_.load(std::memory_order_relaxed);
        const uint64_t r = read_idx_.load(std::memory_order_acquire);
        const uint64_t ring_queued = w - r;
        const uint64_t limit = max_queue_size_.load(std::memory_order_relaxed);
        // Subtract before comparing: ring_queued + overflow_.size() could overflow.
        // Reserve a representable in-flight count before publishing either source.
        return ring_queued < limit && overflow_.size() < limit - ring_queued &&
               in_flight_.load(std::memory_order_relaxed) < std::numeric_limits<uint64_t>::max();
    });
    if (stop_.load(std::memory_order_relaxed)) {
        throw std::runtime_error("enqueue on stopped thread pool");
    }
    const uint64_t w = write_idx_.load(std::memory_order_relaxed);
    const uint64_t r = read_idx_.load(std::memory_order_acquire);
    auto& slot = task_buffer_[w & static_cast<uint64_t>(queue_capacity_ - 1)];
    // Allocate callable and any deque block before incrementing in_flight_.
    std::function<void()> queued_task = [task]() { (*task)(); };
    if (!overflow_.empty() || w - r >= static_cast<uint64_t>(queue_capacity_) ||
        slot.ready.load(std::memory_order_acquire)) {
        overflow_.push_back(std::move(queued_task));
        in_flight_.fetch_add(1, std::memory_order_relaxed);
    } else {
        slot.task = std::move(queued_task);
        in_flight_.fetch_add(1, std::memory_order_relaxed);
        slot.ready.store(true, std::memory_order_release);
        write_idx_.store(w + 1, std::memory_order_release);
    }
    cv_.notify_one();
    return res;  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
}

}  // namespace vsag
