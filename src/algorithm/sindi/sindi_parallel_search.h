// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <vector>

#include "impl/searcher/sindi_searcher.h"
#include "vsag/concurrent_filter.h"

namespace vsag {

struct SindiParallelPlan {
    static SindiParallelPlan
    Make(uint64_t windows, uint64_t candidates, double ratio);

    uint64_t candidates{0};
    uint64_t capacity{0};
    std::vector<uint64_t> boundaries;
    std::vector<uint64_t> quotas;
};

// One invocation, shared only until its workers have joined. Never shared with another query.
struct SindiParallelState {
    using Clock = std::chrono::steady_clock;
    enum class Stop { NONE, DEADLINE, CALLBACK_LIMIT, ERROR };

    bool
    Stopped();
    void
    RequestStop(Stop reason);
    bool
    ReserveCallback();

    FilterPtr user_filter;
    std::unique_ptr<ConcurrentFilterLease> lease;
    std::function<FilterPtr(const FilterPtr&)> map_filter;
    double ratio{1};
    uint32_t requested{1};
    // Internal A/B/C test control, never parsed from request JSON.
    uint32_t execution_limit{16};
    uint32_t granted{0};
    uint64_t callback_limit{0};
    std::atomic<uint64_t> callbacks{0};
    std::atomic<uint64_t> completed_windows{0};
    std::atomic<Stop> stop{Stop::NONE};
    Clock::time_point deadline{Clock::time_point::max()};
    SindiParallelPlan plan;
};

void
SearchSindiPartitions(const SindiWindowSearchRequest& request,
                      SindiParallelState& state,
                      MaxHeap& output);

}  // namespace vsag
