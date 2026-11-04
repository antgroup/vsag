// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "algorithm/sindi_metadata_filter.h"
#include "datacell/sindi_search_term_datacell.h"
#include "query_context.h"
#include "vsag/concurrent_filter.h"

namespace vsag {

class SearchExecutionContext;

using SindiCandidate = std::pair<float, InnerIdType>;
using SindiCandidateHeap = MaxHeap;

// Construct with CompareByFirst{true}: admission and eviction must use the same
// distance/id ordering, including ties.
inline void
SindiMergeCandidate(SindiCandidateHeap& heap, const SindiCandidate& candidate, uint64_t capacity) {
    if (capacity == 0) {
        return;
    }
    if (heap.size() < capacity or candidate < heap.top()) {
        heap.push(candidate);
        if (heap.size() > capacity) {
            heap.pop();
        }
    }
}

// Immutable query inputs; each slot owns its mutable scratch and evaluator.
struct SindiPartitionSearchRequest {
    const std::vector<uint32_t>& windows;
    const SindiSearchTermDataCellPtr& datacell;
    const SparseTermComputerPtr& computer;
    const InnerSearchParam& search;
    const SindiMetadataFilter& metadata_filter;
    const SindiMetadataSearchRoute& route;
    Allocator* allocator;
    uint64_t window_size;
    uint64_t element_count;
    bool use_term_lists;
    SearchStatistics* statistics;
};

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

    std::shared_ptr<SearchExecutionContext> resources;
    FilterPtr user_filter;
    std::unique_ptr<ConcurrentFilterLease> lease;
    std::function<FilterPtr(const FilterPtr&)> map_filter;
    double ratio{2};
    uint32_t requested{1};
    // Internal A/B/C test control, never parsed from request JSON.
    uint32_t execution_limit{16};
    uint32_t granted{0};
    uint32_t filter_slots{0};
    uint32_t cpu_slots{0};
    uint32_t worker_limit{0};
    bool resources_ready{false};
    bool profile_cpu{false};
    bool worker_cpu_available{true};
    uint64_t worker_cpu_ns{0};
    uint64_t callback_limit{0};
    std::atomic<uint64_t> callbacks{0};
    std::atomic<uint64_t> completed_windows{0};
    std::atomic<Stop> stop{Stop::NONE};
    Clock::time_point deadline{Clock::time_point::max()};
    SindiParallelPlan plan;
};

struct SindiCpuTime;
std::string
SindiExecutionStatistics(std::string summary,
                         const SindiParallelState* parallel,
                         const SindiCpuTime& caller_cpu,
                         bool profile_cpu,
                         uint32_t requested_slots = 1);

void
SearchSindiPartitions(const SindiPartitionSearchRequest& request,
                      SindiParallelState& state,
                      MaxHeap& output);

}  // namespace vsag
