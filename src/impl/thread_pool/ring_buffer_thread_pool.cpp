
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

#include <stdexcept>

namespace vsag {
namespace {
thread_local const RingBufferThreadPool* current_pool = nullptr;

class PoolThreadScope {
public:
    explicit PoolThreadScope(const RingBufferThreadPool* pool) : previous_(current_pool) {
        current_pool = pool;
    }

    ~PoolThreadScope() {
        current_pool = previous_;
    }

private:
    const RingBufferThreadPool* previous_;
};
}  // namespace

RingBufferThreadPool::RingBufferThreadPool(uint64_t num_threads, int queue_capacity)
    : queue_capacity_(queue_capacity), task_buffer_(queue_capacity) {
    if (num_threads == 0) {
        throw std::invalid_argument("num_threads must be > 0");
    }
    if (queue_capacity_ <= 0 || (queue_capacity_ & (queue_capacity_ - 1)) != 0) {
        throw std::invalid_argument("queue_capacity must be a positive power of two");
    }
    worker_count_.store(static_cast<int>(num_threads), std::memory_order_relaxed);
    workers_.reserve(num_threads);
    for (uint64_t i = 0; i < num_threads; ++i) {
        auto w = std::make_unique<Worker>();
        w->index = static_cast<int>(i);
        w->thread = std::thread(&RingBufferThreadPool::worker_loop, this, w->index);
        workers_.push_back(std::move(w));
    }
}

RingBufferThreadPool::~RingBufferThreadPool() {
    stop_.store(true, std::memory_order_release);
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w->thread.joinable()) {
            w->thread.join();
        }
    }
}

std::future<void>
RingBufferThreadPool::Enqueue(std::function<void(void)> task) {
    return GeneralEnqueue([this, task = std::move(task)]() {
        PoolThreadScope scope(this);
        task();
    });
}

void
RingBufferThreadPool::WaitUntilEmpty() {
    std::unique_lock<std::mutex> lock(sleep_mutex_);
    empty_cv_.wait(lock, [this] { return in_flight_.load(std::memory_order_acquire) == 0; });
}

void
RingBufferThreadPool::SetPoolSize(uint64_t limit) {
    int current = worker_count_.load(std::memory_order_relaxed);
    int target = static_cast<int>(limit);
    if (target <= 0) {
        return;
    }
    if (target > current) {
        // Grow: publish the target count BEFORE creating threads.  New threads
        // check worker_count_ on entry and would otherwise observe the old
        // count and exit immediately.
        worker_count_.store(target, std::memory_order_release);
        for (int i = current; i < target; ++i) {
            auto w = std::make_unique<Worker>();
            w->index = i;
            w->thread = std::thread(&RingBufferThreadPool::worker_loop, this, w->index);
            workers_.push_back(std::move(w));
        }
    } else if (target < current) {
        // Shrink: signal excess workers to exit
        // Excess workers detect that their index >= pool_size and return from worker_loop
        worker_count_.store(target, std::memory_order_release);
        cv_.notify_all();
        for (int i = target; i < static_cast<int>(workers_.size()); ++i) {
            if (workers_[i] && workers_[i]->thread.joinable()) {
                workers_[i]->thread.join();
            }
            workers_[i].reset();
        }
        workers_.resize(target);
    }
}

void
RingBufferThreadPool::SetQueueSizeLimit(uint64_t limit) {
    // Clamp to the physical ring capacity before storing to avoid narrowing
    // a large limit to a negative or zero effective capacity.
    uint64_t clamped = std::min(limit, static_cast<uint64_t>(queue_capacity_));
    if (clamped == 0) {
        clamped = 1;
    }
    max_queue_size_.store(clamped, std::memory_order_release);
}

bool
RingBufferThreadPool::IsCurrentThread() const {
    return current_pool == this;
}

void
RingBufferThreadPool::SetAffinity(const std::vector<int>& cpu_ids) {
    // Platform-dependent: only supported on Linux via pthread_setaffinity_np
#if defined(__linux__)
    for (size_t idx = 0; idx < cpu_ids.size() && idx < workers_.size(); ++idx) {
        if (!workers_[idx]) {
            continue;
        }
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_ids[idx], &cpuset);
        pthread_setaffinity_np(workers_[idx]->thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }
#else
    (void)cpu_ids;
#endif
}

void
RingBufferThreadPool::worker_loop(int worker_id) {
    for (;;) {
        uint64_t r = read_idx_.load(std::memory_order_relaxed);
        uint64_t w = write_idx_.load(std::memory_order_acquire);

        if (r < w) {
            // CONSUMER: claim the next slot
            uint64_t expected = r;
            if (read_idx_.compare_exchange_weak(
                    expected, r + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
                // Move the task BEFORE signalling empty: the producer uses read_idx_
                // to decide which slots are reusable, so the slot must be finished
                // being read before another producer can claim it.
                auto task = std::move(task_buffer_[(r & (queue_capacity_ - 1))]);

                // Signal empty if last in-flight task BEFORE executing the task.
                // This lets WaitUntilEmpty() observe that no task remains, then
                // any new submission wakes the consumer.  Submitting while the
                // last task is still executing would re-increment in_flight and
                // WaitUntilEmpty() would wait for it.
                if (in_flight_.fetch_sub(1, std::memory_order_release) == 1) {
                    std::lock_guard<std::mutex> lock(sleep_mutex_);
                    empty_cv_.notify_all();
                }

                {
                    PoolThreadScope scope(this);
                    task();
                }
                continue;
            }
            // CAS failed — another worker claimed it; retry
            continue;
        }

        // Check for stop
        if (stop_.load(std::memory_order_relaxed)) {
            return;
        }

        // Check for pool shrink
        int limit = worker_count_.load(std::memory_order_acquire);
        if (worker_id >= limit) {
            return;
        }

        // Wait for work
        std::unique_lock<std::mutex> lock(sleep_mutex_);
        if (stop_.load(std::memory_order_relaxed)) {
            return;
        }
        // Recheck limit under lock to avoid missing a shrink signal
        limit = worker_count_.load(std::memory_order_acquire);
        if (worker_id >= limit) {
            return;
        }
        if (read_idx_.load(std::memory_order_acquire) >=
            write_idx_.load(std::memory_order_acquire)) {
            cv_.wait(lock);
        }
