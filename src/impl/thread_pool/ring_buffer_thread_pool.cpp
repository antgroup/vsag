
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

#include <limits>
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

RingBufferThreadPool::RingBufferThreadPool(uint64_t num_threads)
    : RingBufferThreadPool(num_threads, kDefaultQueueCapacity) {
    RingBufferThreadPool::SetQueueSizeLimit(kDefaultQueueSizeLimit);
}

RingBufferThreadPool::RingBufferThreadPool(uint64_t num_threads, int queue_capacity)
    : queue_capacity_(queue_capacity) {
    if (num_threads == 0 || num_threads > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("invalid thread count");
    }
    if (queue_capacity_ <= 0 || (queue_capacity_ & (queue_capacity_ - 1)) != 0) {
        throw std::invalid_argument("queue_capacity must be a positive power of two");
    }
    task_buffer_ = std::make_unique<Slot[]>(queue_capacity_);
    max_queue_size_.store(static_cast<uint64_t>(queue_capacity_));
    try {
        RingBufferThreadPool::SetPoolSize(num_threads);
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(write_mutex_);
            stop_.store(true, std::memory_order_release);
        }
        cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker->thread.joinable()) {
                worker->thread.join();
            }
        }
        throw;
    }
}

RingBufferThreadPool::~RingBufferThreadPool() {
    {
        std::lock_guard<std::mutex> lock(write_mutex_);
        stop_.store(true, std::memory_order_release);
    }
    cv_.notify_all();
    not_full_cv_.notify_all();
    for (auto& worker : workers_) {
        if (worker->thread.joinable()) {
            worker->thread.join();
        }
    }
}

std::future<void>
RingBufferThreadPool::Enqueue(std::function<void(void)> task) {
    return GeneralEnqueue(std::move(task));
}

void
RingBufferThreadPool::WaitUntilEmpty() {
    if (IsCurrentThread()) {
        throw std::logic_error("cannot wait for a thread pool from its worker");
    }
    std::unique_lock<std::mutex> lock(completion_mutex_);
    empty_cv_.wait(lock, [this] { return in_flight_.load(std::memory_order_acquire) == 0; });
}

void
RingBufferThreadPool::SetPoolSize(uint64_t limit) {
    if (IsCurrentThread()) {
        throw std::logic_error("cannot resize a thread pool from its worker");
    }
    if (limit == 0) {
        return;
    }
    if (limit > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("invalid thread count");
    }
    std::lock_guard<std::mutex> management_lock(management_mutex_);
    int current = worker_count_.load(std::memory_order_acquire);
    int target = static_cast<int>(limit);
    if (target > current) {
        workers_.reserve(limit);
        try {
            for (int i = current; i < target; ++i) {
                auto worker = std::make_unique<Worker>();
                worker->index = i;
                {
                    std::lock_guard<std::mutex> lock(write_mutex_);
                    worker_count_.store(i + 1, std::memory_order_release);
                }
                worker->thread = std::thread(&RingBufferThreadPool::worker_loop, this, i);
                workers_.push_back(std::move(worker));
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(write_mutex_);
                worker_count_.store(current, std::memory_order_release);
            }
            cv_.notify_all();
            for (uint64_t i = current; i < workers_.size(); ++i) {
                workers_[i]->thread.join();
            }
            workers_.resize(current);
            throw;
        }
    } else if (target < current) {
        {
            std::lock_guard<std::mutex> lock(write_mutex_);
            worker_count_.store(target, std::memory_order_release);
        }
        cv_.notify_all();
        for (int i = target; i < current; ++i) {
            workers_[i]->thread.join();
        }
        workers_.resize(target);
    }
}

void
RingBufferThreadPool::SetQueueSizeLimit(uint64_t limit) {
    {
        std::lock_guard<std::mutex> lock(write_mutex_);
        max_queue_size_.store(std::max(limit, uint64_t{1}), std::memory_order_release);
    }
    not_full_cv_.notify_all();
}

bool
RingBufferThreadPool::IsCurrentThread() const {
    return current_pool == this;
}

void
RingBufferThreadPool::worker_loop(int worker_id) {
    PoolThreadScope scope(this);
    for (;;) {
        if (worker_id >= worker_count_.load(std::memory_order_acquire)) {
            return;
        }
        std::function<void()> task;
        uint64_t r = read_idx_.load(std::memory_order_relaxed);
        const uint64_t w = write_idx_.load(std::memory_order_acquire);
        // A stale write snapshot must not let a consumer claim unpublished work.
        const uint64_t available = w - r;
        if (available != 0 && available <= static_cast<uint64_t>(queue_capacity_)) {
            if (!read_idx_.compare_exchange_weak(r, r + 1, std::memory_order_relaxed)) {
                continue;
            }
            auto& slot = task_buffer_[r & static_cast<uint64_t>(queue_capacity_ - 1)];
            task = std::move(slot.task);
            // Claim != reclamation: spill rather than overwrite a claimed slot.
            {
                std::lock_guard<std::mutex> lock(write_mutex_);
                slot.ready.store(false, std::memory_order_release);
            }
            not_full_cv_.notify_all();
        } else {
            std::unique_lock<std::mutex> lock(write_mutex_);
            // Recheck under the publisher mutex: stale empty must not let overflow
            // overtake unclaimed ring work. Retry CAS if any ring work remains.
            if (read_idx_.load(std::memory_order_relaxed) !=
                write_idx_.load(std::memory_order_relaxed)) {
                continue;
            }
            if (!overflow_.empty()) {
                task = std::move(overflow_.front());
                overflow_.pop_front();
                not_full_cv_.notify_all();
            } else {
                if (stop_.load(std::memory_order_relaxed)) {
                    return;
                }
                cv_.wait(lock, [this, worker_id] {
                    return stop_.load(std::memory_order_relaxed) || !overflow_.empty() ||
                           worker_id >= worker_count_.load(std::memory_order_relaxed) ||
                           read_idx_.load(std::memory_order_relaxed) !=
                               write_idx_.load(std::memory_order_relaxed);
                });
                continue;
            }
        }
        task();
        // Both sources share completion accounting and publish completed task effects.
        const uint64_t previous = in_flight_.fetch_sub(1, std::memory_order_acq_rel);
        // Admission permits MAX-1 -> MAX. Completion MAX -> MAX-1 must wake
        // producers parked on the representable-count guard (not an underflow).
        if (previous == std::numeric_limits<uint64_t>::max()) {
            std::lock_guard<std::mutex> lock(write_mutex_);
            not_full_cv_.notify_all();
        }
        if (previous == 1) {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            empty_cv_.notify_all();
        }
    }
}

}  // namespace vsag
