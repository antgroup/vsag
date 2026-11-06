
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
#include <system_error>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

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
        SetPoolSize(num_threads);
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
    for (auto& worker : workers_) {
        if (worker->thread.joinable()) {
            worker->thread.join();
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
    if (IsCurrentThread()) {
        throw std::logic_error("cannot wait for a thread pool from its worker");
    }
    std::unique_lock<std::mutex> lock(write_mutex_);
    empty_cv_.wait(lock, [this] { return in_flight_ == 0; });
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
    if (IsCurrentThread()) {
        throw std::logic_error("cannot change pool affinity from its worker");
    }
    std::lock_guard<std::mutex> management_lock(management_mutex_);
    // Platform-dependent: only supported on Linux via pthread_setaffinity_np
#if defined(__linux__)
    for (uint64_t idx = 0; idx < cpu_ids.size() && idx < workers_.size(); ++idx) {
        if (!workers_[idx]) {
            continue;
        }
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (cpu_ids[idx] < 0 || cpu_ids[idx] >= CPU_SETSIZE) {
            throw std::invalid_argument("invalid CPU id");
        }
        CPU_SET(cpu_ids[idx], &cpuset);
        int error = pthread_setaffinity_np(
            workers_[idx]->thread.native_handle(), sizeof(cpu_set_t), &cpuset);
        if (error != 0) {
            throw std::system_error(error, std::generic_category());
        }
    }
#else
    (void)cpu_ids;
#endif
}

void
RingBufferThreadPool::worker_loop(int worker_id) {
    PoolThreadScope scope(this);
    for (;;) {
        if (worker_id >= worker_count_.load(std::memory_order_acquire)) {
            return;
        }
        uint64_t r = read_idx_.load(std::memory_order_relaxed);
        uint64_t w = write_idx_.load(std::memory_order_acquire);
        // A stale write snapshot must not let a consumer claim unpublished work.
        uint64_t available = w - r;
        if (available != 0 && available <= static_cast<uint64_t>(queue_capacity_)) {
            if (!read_idx_.compare_exchange_weak(r, r + 1, std::memory_order_relaxed)) {
                continue;
            }
            auto& slot = task_buffer_[r & static_cast<uint64_t>(queue_capacity_ - 1)];
            auto task = std::move(slot.task);
            // A claimed index is not a reclaimed slot: publish reuse only after moving.
            slot.ready.store(false, std::memory_order_release);
            task();
            {
                std::lock_guard<std::mutex> lock(write_mutex_);
                if (--in_flight_ == 0) {
                    empty_cv_.notify_all();
                }
            }
            continue;
        }
        std::unique_lock<std::mutex> lock(write_mutex_);
        cv_.wait(lock, [this, worker_id] {
            return stop_.load(std::memory_order_relaxed) ||
                   worker_id >= worker_count_.load(std::memory_order_relaxed) ||
                   read_idx_.load(std::memory_order_relaxed) !=
                       write_idx_.load(std::memory_order_relaxed);
        });
        if (stop_.load(std::memory_order_relaxed) &&
            read_idx_.load(std::memory_order_relaxed) ==
                write_idx_.load(std::memory_order_relaxed)) {
            return;
        }
    }
}

}  // namespace vsag
