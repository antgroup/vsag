// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <condition_variable>
#include <cstdint>
#include <future>
#include <mutex>
#include <vector>

#include "vsag/thread_pool.h"
namespace vsag {
inline void
ConfirmThreadPoolWorkers(ThreadPool& pool, uint32_t workers) {
    if (workers == 0) {
        return;
    }
    // Block one task on each worker, so completion proves every worker has started.
    // Keep this inside construction: no query or initializer can observe
    // a pool whose readiness check is still pending.
    std::promise<void> release;
    auto gate = release.get_future().share();
    std::mutex ready_mutex;
    std::condition_variable ready;
    uint32_t started = 0;
    std::vector<std::future<void>> tasks;
    tasks.reserve(workers);
    try {
        for (uint32_t i = 0; i < workers; ++i) {
            tasks.push_back(pool.Enqueue([&ready_mutex, &ready, &started, gate] {
                {
                    std::lock_guard<std::mutex> guard(ready_mutex);
                    ++started;
                }
                ready.notify_one();
                gate.wait();
            }));
        }
        std::unique_lock<std::mutex> guard(ready_mutex);
        ready.wait(guard, [&] { return started == workers; });
    } catch (...) {
        release.set_value();
        for (auto& task : tasks) {
            task.wait();
        }
        throw;
    }
    release.set_value();
    for (auto& task : tasks) {
        task.get();
    }
}

}  // namespace vsag
