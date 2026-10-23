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

#include <cstdint>
#include <string>
#include <vector>

#include "algorithm/sindi_metadata_filter.h"
#include "datacell/sindi_search_term_datacell.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "query_context.h"
#include "sindi_query_profile.h"

namespace vsag {

class FlattenInterface;

enum class SindiWindowScheduler { COUNT_DEQUE, COUNT_ATOMIC, COST_ATOMIC };
enum class SindiRerankScheduler { COUNT_STATIC, COUNT_ATOMIC, NNZ_ATOMIC };

struct SindiWindowSearchOptions {
    SindiWindowScheduler scheduler{SindiWindowScheduler::COUNT_DEQUE};
    bool profile{false};
    bool collect_statistics{true};
    bool stage_profile{false};
    bool rerank_batch{false};
    uint32_t rerank_parallelism{1};
    SindiRerankScheduler rerank_scheduler{SindiRerankScheduler::COUNT_STATIC};
};

struct alignas(64) SindiWindowWorkerProfile {
    uint64_t windows{0};
    uint64_t stolen_windows{0};
    uint64_t steal_attempts{0};
    uint64_t estimated_cost{0};
    double start_delay_us{0};
    double elapsed_us{0};
};

struct SindiWindowSearchProfile {
    bool ran{false};
    double cost_estimation_us{0};
    double partition_us{0};
    double dispatch_join_us{0};
    double merge_us{0};
    bool bulk_merge_used{false};
    std::vector<uint64_t> boundaries;
    std::vector<uint64_t> initial_costs;
    std::vector<SindiWindowWorkerProfile> workers;
};

struct SindiWindowSearchRequest {
    const std::vector<uint32_t>& windows;
    const SindiSearchTermDataCellPtr& datacell;
    const SparseTermComputerPtr& computer;
    const InnerSearchParam& search;
    const SindiMetadataFilter& metadata_filter;
    const SindiMetadataSearchRoute& route;
    SafeThreadPoolPtr pool;
    Allocator* allocator;
    uint64_t window_size;
    uint64_t element_count;
    uint64_t worker_candidates;
    bool use_term_lists;
    SearchStatistics* statistics;
};

/** Worker heap with read access to its unsorted candidate storage for bulk merging. */
class SindiWindowHeap : public MaxHeap {
public:
    using MaxHeap::MaxHeap;

    const Vector<std::pair<float, InnerIdType>>&
    Candidates() const {
        return this->c;
    }
};

/** Query-local window scheduling, candidate merging and FP32/FP16 sparse reranking. */
class SindiSearcher {
public:
    static SindiWindowSearchOptions
    ParseOptions(const JsonType& json);

    static const char*
    SchedulerName(SindiWindowScheduler scheduler);

    // Half-open ranges into the selected-window array, never raw window IDs.
    static std::vector<uint64_t>
    Partition(const std::vector<uint64_t>& costs, uint32_t workers, bool weighted);

    // Reads posting metadata only and honors query/term pruning and full-scan routes.
    static uint64_t
    EstimateWindowCost(const SindiSearchTermDataCell& datacell,
                       const SparseTermComputer& computer,
                       uint32_t window_id,
                       bool full_term_scan);

    // Keep the same candidate set; fall back when equal scores straddle the selection cutoff.
    // Equal selected scores may be traversed in a different order during reranking.
    static bool
    MergeHeaps(std::vector<SindiWindowHeap>& heaps,
               MaxHeap& heap,
               uint64_t limit,
               Allocator* allocator);

    // Private computers; each claimed candidate writes its original output position exactly once.
    static Vector<float>
    RerankDistances(FlattenInterface& datacell,
                    const SparseVector& query,
                    const Vector<InnerIdType>& ids,
                    uint32_t workers,
                    const SafeThreadPoolPtr& pool,
                    Allocator* allocator,
                    SearchStatistics* statistics,
                    bool stage_profile,
                    SindiRerankScheduler scheduler = SindiRerankScheduler::COUNT_STATIC);

    static void
    SearchWindows(const SindiWindowSearchRequest& request,
                  const SindiWindowSearchOptions& options,
                  SindiWindowSearchProfile& profile,
                  MaxHeap& heap);

    static std::string
    DumpStatistics(const SearchStatistics& statistics,
                   const SindiWindowSearchOptions& options,
                   const SindiWindowSearchProfile& profile);
};

}  // namespace vsag
