// Copyright 2026-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "batched_graph_search.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "common.h"
#include "impl/heap/standard_heap.h"

namespace vsag {
namespace {
template <bool EnableGap>
DistHeapPtr
search_batched_graph(const GraphInterfacePtr& graph,
                     const FlattenInterfacePtr& flatten,
                     const VisitedListPtr& visited,
                     const ComputerInterfacePtr& computer,
                     const InnerSearchParam& p,
                     Allocator* allocator,
                     const MutexArrayPtr& mutexes,
                     QueryContext* context,
                     DistanceRecordVector* lower_bound_candidates,
                     const GraphTailSearchParameters& tail,
                     uint64_t requested_topk) {
    auto results = std::make_shared<StandardHeap<true, false>>(allocator, -1);
    auto frontier = std::make_shared<StandardHeap<true, false>>(allocator, -1);
    if (lower_bound_candidates != nullptr) {
        lower_bound_candidates->clear();
    }
    // Gap tracking uses the final user k, not the enclosing graph's ef-sized heap.
    Vector<float> nearest(allocator);
    if constexpr (EnableGap) {
        nearest.reserve(requested_topk);
    }
    const auto observe = [&](float distance) {
        if constexpr (EnableGap) {
            if (nearest.size() < requested_topk) {
                nearest.push_back(distance);
                std::push_heap(nearest.begin(), nearest.end());
            } else if (distance < nearest.front()) {
                std::pop_heap(nearest.begin(), nearest.end());
                nearest.back() = distance;
                std::push_heap(nearest.begin(), nearest.end());
            }
        }
    };
    const auto entry = p.ep;
    float distance;
    flatten->QueryWithDistanceLowerBound(&distance, nullptr, computer, &entry, 1, context);
    CHECK_ARGUMENT(std::isfinite(distance), "split tail requires finite filter distances");
    results->Push(distance, entry);
    frontier->Push(-distance, entry);
    visited->Set(entry);
    observe(distance);
    uint64_t expanded = 0, comparisons = 1;
    Vector<InnerIdType> selected(allocator), adjacent(allocator), fresh(allocator);
    Vector<float> distances(allocator), lower_bounds(allocator);
    while (not frontier->Empty()) {
        if (results->Size() >= p.ef and -frontier->Top().first > results->Top().first) {
            break;
        }
        if constexpr (EnableGap) {
            if (expanded >= tail.start_after and nearest.size() == requested_topk and
                -frontier->Top().first > nearest.front() * (1.0F + *tail.relative_gap)) {
                break;
            }
        }
        const bool batching = expanded >= tail.start_after;
        const uint64_t limit = batching ? tail.batch_size : 1;
        const auto round_threshold = results->Top().first;
        selected.clear();
        fresh.clear();
        while (not frontier->Empty() and selected.size() < limit and
               expanded + selected.size() + 1 < p.hops_limit) {
            if (results->Size() >= p.ef and -frontier->Top().first > round_threshold) {
                break;
            }
            selected.push_back(frontier->Top().second);
            frontier->Pop();
        }
        if (selected.empty()) {
            break;
        }
        for (const auto id : selected) {
            if (mutexes != nullptr) {
                SharedLock lock(mutexes, id);
                graph->GetNeighbors(id, adjacent);
            } else {
                graph->GetNeighbors(id, adjacent);
            }
            for (uint64_t i = 0; i < adjacent.size(); ++i) {
                if (i + 3 < adjacent.size()) {
                    visited->Prefetch(adjacent[i + 3]);
                }
                if (not visited->TestAndSet(adjacent[i])) {
                    fresh.push_back(adjacent[i]);
                }
            }
        }
        // Match native collection semantics: collect only if the ef heap was full before
        // scoring this expansion round, then compare each lower bound with the current heap.
        const bool collect = results->Size() == p.ef and lower_bound_candidates != nullptr;
        distances.resize(fresh.size());
        lower_bounds.resize(collect ? fresh.size() : 0);
        if (not fresh.empty()) {
            flatten->QueryWithDistanceLowerBound(distances.data(),
                                                 collect ? lower_bounds.data() : nullptr,
                                                 computer,
                                                 fresh.data(),
                                                 fresh.size(),
                                                 context);
        }
        for (uint64_t i = 0; i < fresh.size(); ++i) {
            const auto d = distances[i];
            CHECK_ARGUMENT(std::isfinite(d), "split tail requires finite filter distances");
            observe(d);
            if (collect and lower_bounds[i] < results->Top().first) {
                lower_bound_candidates->emplace_back(lower_bounds[i], fresh[i]);
            }
            if (results->Size() < p.ef or d < results->Top().first) {
                frontier->Push(-d, fresh[i]);
                results->Push(d, fresh[i]);
                if (results->Size() > p.ef) {
                    results->Pop();
                }
            }
        }
        expanded += selected.size();
        comparisons += fresh.size();
    }
    while (results->Size() > static_cast<uint64_t>(p.topk)) {
        results->Pop();
    }
    if (context != nullptr and context->stats != nullptr) {
        context->stats->dist_cmp.fetch_add(comparisons, std::memory_order_relaxed);
        context->stats->hops.fetch_add(expanded, std::memory_order_relaxed);
    }
    return results;
}
}  // namespace

DistHeapPtr
SearchBatchedGraph(const GraphInterfacePtr& graph,
                   const FlattenInterfacePtr& flatten,
                   const VisitedListPtr& visited,
                   const ComputerInterfacePtr& computer,
                   const InnerSearchParam& parameters,
                   Allocator* allocator,
                   const MutexArrayPtr& mutexes,
                   QueryContext* context,
                   DistanceRecordVector* lower_bound_candidates,
                   const GraphTailSearchParameters& tail,
                   uint64_t requested_topk) {
    CHECK_ARGUMENT(
        parameters.search_mode == KNN_SEARCH and parameters.enable_rabitq_one_bit_search and
            parameters.enable_reorder and flatten->SupportSplitCodeStorage() and
            parameters.is_inner_id_allowed == nullptr and parameters.executors.empty() and
            not parameters.consider_duplicate and not parameters.find_duplicate and
            parameters.distance_batch_func == nullptr and parameters.time_cost == nullptr and
            not parameters.distance_threshold.has_value() and requested_topk > 0 and
            tail.batch_size > 0,
        "batched graph search requires unfiltered split KNN");
    if (tail.relative_gap.has_value()) {
        return search_batched_graph<true>(graph,
                                          flatten,
                                          visited,
                                          computer,
                                          parameters,
                                          allocator,
                                          mutexes,
                                          context,
                                          lower_bound_candidates,
                                          tail,
                                          requested_topk);
    }
    return search_batched_graph<false>(graph,
                                       flatten,
                                       visited,
                                       computer,
                                       parameters,
                                       allocator,
                                       mutexes,
                                       context,
                                       lower_bound_candidates,
                                       tail,
                                       requested_topk);
}
}  // namespace vsag
