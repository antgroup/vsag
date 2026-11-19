// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "datacell/sindi_search_term_datacell.h"
#include "json_types.h"
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
    // Read-only policy, invoked concurrently; captures outlive all workers.
    // Empty means posting pruning is allowed for every selected window.
    std::function<bool(uint32_t)> term_prune_enabled;
    Allocator* allocator;
    uint64_t window_size;
    uint64_t element_count;
    bool use_term_lists;
    SearchStatistics* statistics;
};

struct SindiParallelPlan {
    static SindiParallelPlan
    Make(uint64_t windows, uint64_t candidates, double ratio, uint64_t topk);

    uint64_t candidates{0};
    uint64_t capacity{0};
    std::vector<uint64_t> boundaries;
    std::vector<uint64_t> quotas;
};

struct SindiSlotStatistics {
    uint64_t computed{0};
    uint64_t attempted{0};
    uint64_t consumed{0};
    uint64_t compute_ns{0};
    uint64_t consume_ns{0};
    uint64_t clear_ns{0};
    uint64_t filter_ns{0};
    uint64_t filter_calls{0};
    uint64_t postings{0};
    uint64_t compute_max_ns{0};
    uint32_t slowest_window{0};
    uint64_t slowest_postings{0};
    std::array<uint64_t, 6> histogram{};
};

struct SindiPartitionStatistics {
    uint64_t windows{0};
    uint64_t candidates{0};
    uint64_t wall_ns{0};
};

// Optional, invocation-owned diagnostics. The index chooses whether to collect
// basic counters and whether to enable detailed timing. Never shared by queries.
struct SindiParallelStatistics {
    bool profile_search{false};
    uint64_t admission_ns{0};
    uint64_t slot_prepare_ns{0};
    uint64_t wait_ns{0};
    uint64_t merge_ns{0};
    std::vector<SindiSlotStatistics> slots;
    std::vector<SindiPartitionStatistics> partitions;
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
    uint64_t planning_topk{0};
    bool clear_window_scratch{false};
    // Borrowed until SearchSindiPartitions drains. Null keeps the original
    // SINDI path free of the optional diagnostics allocations and counters.
    SindiParallelStatistics* diagnostics{nullptr};
    uint32_t requested{1};
    // Internal A/B/C test control, never parsed from request JSON.
    uint32_t execution_limit{16};
    uint32_t granted{0};
    uint32_t filter_slots{0};
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
SindiExecutionStatistics(JsonType&& json,
                         const SindiParallelState* parallel,
                         const SindiCpuTime& caller_cpu,
                         bool profile_cpu,
                         uint32_t requested_slots = 1);

void
SearchSindiPartitions(const SindiPartitionSearchRequest& request,
                      SindiParallelState& state,
                      MaxHeap& output);

}  // namespace vsag
