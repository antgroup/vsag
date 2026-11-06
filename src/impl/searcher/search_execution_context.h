// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <algorithm>
#include <cstdint>
#include <mutex>
#include <optional>

#include "impl/thread_pool/safe_thread_pool.h"

namespace vsag {
// One explicitly shared query-resource group. No index types or query parameters.
class SearchExecutionContext {
public:
    explicit SearchExecutionContext(std::optional<uint32_t> maximum = {});
    uint32_t
    TryReserve(uint32_t wanted) {
        std::lock_guard<std::mutex> guard(mutex);
        const auto available = slots > callers + busy ? slots - callers - busy : 0;
        const auto count = std::min({wanted, limit - busy, available});
        busy += count;
        return count;
    }
    uint32_t slots;
    uint32_t limit;
    // Both counters are guarded by mutex, including reservation release.
    // Admission maintains busy <= limit; limit is fixed at construction.
    uint32_t callers{0};
    uint32_t busy{0};
    std::mutex mutex;
    SafeThreadPoolPtr pool;
};
}  // namespace vsag
