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

#include "hybrid_mci_searcher.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "container_types.h"
#include "impl/heap/search_candidate_queue.h"
#include "impl/heap/standard_heap.h"
#include "impl/query_computer_pool.h"
#include "impl/searcher/searcher_utils.h"
#include "index_common_param.h"
#include "query_context.h"
#include "vsag/filter.h"

namespace vsag {
namespace {

/**
 * O(1) reset visited table.
 *
 * The traversal resets the table once per search, so instead of clearing an
 * N-entry array per query the entries store a generation tag. A tag bump
 * invalidates every entry at once; only a tag wrap costs a real clear.
 *
 * Lifetime contract: instances are `thread_local` in Search(), so each worker
 * thread keeps one buffer for the largest index it has served. That is one byte
 * per vector (plus one per clique for the clique table) per thread, which the
 * buffer never grows past for a given index. The buffer is released when the
 * thread exits; it holds no lock, no shared state and no borrow of the index, so
 * Search() stays reentrant across threads and safe to call concurrently.
 * MCISearcher uses the same thread-local epoch pattern for its own tables.
 */
// NOLINTNEXTLINE(readability-identifier-naming)
struct EpochVisited {
    static constexpr size_t K_OVERSIZE_FACTOR = 4;
    static constexpr size_t K_OVERSIZE_FLOOR = 1U << 20U;
    std::vector<uint8_t> marks;
    uint8_t tag{1};

    void
    Reset(uint64_t size) {
        if (marks.size() < size) {
            // A thread that served a much larger index would otherwise pin that
            // buffer forever; drop the oversized one instead of reusing it.
            if (marks.capacity() > K_OVERSIZE_FACTOR * size and
                marks.capacity() > K_OVERSIZE_FLOOR) {
                marks.clear();
                marks.shrink_to_fit();
            }
            marks.assign(size, 0);
            tag = 1;
            return;
        }
        ++tag;
        if (tag == 0) {
            std::memset(marks.data(), 0, marks.size());
            tag = 1;
        }
    }

    // Both accessors rely on one invariant: `id` is an inner id produced by the graph or by
    // the clique member lists, so it is non-negative. The conversion is written out rather
    // than left to the usual arithmetic conversions: a negative id then becomes a huge
    // unsigned index, which fails the bound check instead of being dereferenced. Every
    // caller rejects ids >= total before marking and Reset() has sized the table for the
    // whole index, so Set()'s resize is a safety net rather than the normal path.
    [[nodiscard]] bool
    Get(InnerIdType id) const {
        const auto index = static_cast<uint64_t>(id);
        return index < marks.size() and marks[index] == tag;
    }

    void
    Set(InnerIdType id) {
        const auto index = static_cast<uint64_t>(id);
        if (index >= marks.size()) {
            marks.resize(index + 1, 0);
        }
        marks[index] = tag;
    }
};

/**
 * traverse_should_stop(): the virtual-overhead budget of one MCI traversal.
 *
 * The accumulator lives for exactly one neighbour traversal (one expanded
 * v_0), which matches the scope of the early-termination rule. The local
 * selectivity is estimated online as the running hit rate of the predicate, so
 * the same boundary adapts to both a cheap predicate over a dense satisfactory
 * region and an expensive predicate over a sparse one.
 *
 *   accounted = considered * (filter_cost_ratio + satisfied / considered)
 *             = considered * filter_cost_ratio + satisfied
 *
 * The right-hand form is the one to reason about: during a single neighbour
 * traversal every considered member pays the filter cost, and every satisfactory
 * member adds the unit distance cost, so vob is a budget in units of one distance
 * computation.
 */
// NOLINTNEXTLINE(readability-identifier-naming)
class VirtualOverheadBudget {
public:
    VirtualOverheadBudget(double vob, double filter_cost_ratio)
        : vob_(vob), filter_cost_ratio_(filter_cost_ratio) {
    }

    void
    Reset() {
        considered_ = 0;
        satisfied_ = 0;
    }

    void
    Account(bool satisfied) {
        ++considered_;
        if (satisfied) {
            ++satisfied_;
        }
    }

    [[nodiscard]] bool
    ShouldStop() const {
        if (vob_ <= 0.0 or considered_ == 0) {
            return false;
        }
        const auto local_selectivity =
            static_cast<double>(satisfied_) / static_cast<double>(considered_);
        const auto accumulated =
            static_cast<double>(considered_) * (filter_cost_ratio_ + local_selectivity);
        return accumulated >= vob_;
    }

    [[nodiscard]] uint64_t
    Considered() const {
        return considered_;
    }

private:
    double vob_{0.0};
    double filter_cost_ratio_{1.0};
    uint64_t considered_{0};
    uint64_t satisfied_{0};
};

}  // namespace

HybridMCISearcher::HybridMCISearcher(const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()) {
}

DistHeapPtr
HybridMCISearcher::Search(const GraphInterfacePtr& graph,
                          const CliqueDataCellPtr& cliques,
                          const FlattenInterfacePtr& flatten,
                          const void* query,
                          const FilterPtr& filter,
                          const HybridTraversalParam& param,
                          QueryContext* ctx,
                          HybridSearchStats* stats) const {
    if (stats != nullptr) {
        *stats = HybridSearchStats{};
    }
    auto* alloc = select_query_allocator(ctx, allocator_);
    const auto result_capacity = std::max<int64_t>(1, param.topk);
    // heap_r: max-heap, so Top() is heap_r.farthest_dist() and the fixed size
    // evicts the farthest member as soon as a better vector arrives.
    auto heap_r = DistanceHeap::MakeInstanceBySize<true, true>(alloc, result_capacity);
    if (graph == nullptr or flatten == nullptr or query == nullptr) {
        return heap_r;
    }

    const auto total = flatten->TotalCount();
    if (total == 0 or param.entry_id >= total) {
        return heap_r;
    }

    // heap_c: ordered candidate list of capacity e_f, which gives a real
    // pop_min() and keeps the frontier bounded the way the traversal expects.
    const auto candidate_capacity = std::max<uint64_t>(1, param.ef);
    // NOLINTNEXTLINE(readability-identifier-naming)
    SearchCandidateQueue heap_c(alloc);
    heap_c.Reset(candidate_capacity);

    thread_local EpochVisited visited_nodes;
    thread_local EpochVisited visited_cliques;
    visited_nodes.Reset(total);
    const bool mci_available =
        cliques != nullptr and cliques->HasCliqueIndex(static_cast<uint64_t>(total));
    if (mci_available) {
        visited_cliques.Reset(cliques->TotalLogicalCliqueCount());
    }

    auto computer_lease = AcquireQueryComputer(flatten, query, ctx);
    const auto& computer = computer_lease.computer;

    auto is_satisfactory = [&filter](InnerIdType id) {
        return filter == nullptr or filter->CheckValid(static_cast<int64_t>(id));
    };
    uint64_t distance_computations = 0;
    // Candidates are gathered per expansion and scored with one batched Query call. The
    // flatten API already takes an id array, so scoring them one at a time only pays the
    // per-call overhead ~20 times per expansion, plus once per injected seed.
    Vector<InnerIdType> batch_ids(alloc);
    Vector<float> batch_dists(alloc);
    auto compute_batch = [&]() {
        const auto count = batch_ids.size();
        if (count == 0) {
            return;
        }
        batch_dists.resize(count);
        flatten->Query(
            batch_dists.data(), computer, batch_ids.data(), static_cast<InnerIdType>(count), ctx);
        distance_computations += count;
        if (stats != nullptr) {
            stats->dist_computations += count;
        }
    };

    // ---- lines 4-8: seed the heaps ----
    // A single coarse-search entry only works while the predicate is loose. At low
    // selectivity almost the entire neighbourhood of that entry is filtered out, so the
    // caller may inject seeds sampled from the predicate's own valid set instead; each
    // seed enters the traversal exactly like the entry vector does. The whole list is
    // scored in one batch because it can hold thousands of ids.
    const bool has_seed_list =
        param.seed_inner_ids != nullptr and not param.seed_inner_ids->empty();
    {
        const auto available =
            has_seed_list ? static_cast<uint64_t>(param.seed_inner_ids->size()) : 1;
        const auto wanted =
            not has_seed_list
                ? 1
                : (param.seed_count == 0 ? available
                                         : std::min<uint64_t>(param.seed_count, available));
        for (uint64_t i = 0; i < wanted; ++i) {
            // Even strides keep the sample spread over the whole valid set.
            const auto offset = has_seed_list ? i * available / wanted : 0;
            const auto id = has_seed_list ? (*param.seed_inner_ids)[static_cast<size_t>(offset)]
                                          : param.entry_id;
            if (id >= total or visited_nodes.Get(id)) {
                continue;
            }
            visited_nodes.Set(id);
            batch_ids.push_back(id);
        }
        compute_batch();
        for (size_t i = 0; i < batch_ids.size(); ++i) {
            const auto id = batch_ids[i];
            const auto distance = batch_dists[i];
            heap_c.Insert(distance, id);
            if (is_satisfactory(id)) {
                heap_r->Push(distance, id);
            }
            // seeded_entries counts injected seeds, not the fallback entry point.
            if (stats != nullptr and has_seed_list) {
                ++stats->seeded_entries;
            }
        }
        batch_ids.clear();
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    VirtualOverheadBudget mci_budget(param.vob, param.filter_cost_ratio);
    Vector<InnerIdType> neighbors(alloc);
    Vector<InnerIdType> clique_ids(alloc);
    Vector<InnerIdType> members(alloc);

    // ---- line 9: main traversal loop ----
    // When the seed list is exactly the whole valid set, every valid point already has a
    // distance in heap_r, so expansion can only re-derive candidates that are already there.
    // Answering straight from the seeds is exact, not an approximation.
    uint64_t expanded_nodes = 0;
    const bool skip_expansion = param.seeds_are_exhaustive and has_seed_list;
    if (skip_expansion and stats != nullptr) {
        stats->expansion_skipped = true;
    }
    while (not skip_expansion and heap_c.Size() > 0) {
        if (expanded_nodes >= param.hops_limit) {
            break;
        }
        auto* current = heap_c.GetClosestUnexpanded();  // line 10: pop_min()
        if (current == nullptr) {
            break;
        }
        ++expanded_nodes;
        if (stats != nullptr) {
            stats->expanded_nodes = expanded_nodes;
        }
        const auto v_0 = current->inner_id;

        // ---- lines 11-19: Hgraph(v_0) handled with distance-first semantics ----
        neighbors.clear();
        graph->GetNeighbors(v_0, neighbors);
        batch_ids.clear();
        for (const auto v_1 : neighbors) {
            if (v_1 >= total) {
                continue;
            }
            if (visited_nodes.Get(v_1)) {  // lines 12-13
                continue;
            }
            visited_nodes.Set(v_1);  // line 14
            batch_ids.push_back(v_1);
        }
        compute_batch();  // line 15, batched over the whole unvisited frontier
        for (size_t i = 0; i < batch_ids.size(); ++i) {
            const auto v_1 = batch_ids[i];
            const auto distance = batch_dists[i];
            heap_c.Insert(distance, v_1);  // line 16
            if ((heap_r->Size() < static_cast<uint64_t>(result_capacity) or
                 distance < heap_r->Top().first) and
                not is_nan_distance(distance)) {  // line 17
                if (is_satisfactory(v_1)) {       // line 18
                    heap_r->Push(distance, v_1);  // line 19
                }
            }
            if (stats != nullptr) {
                ++stats->sparse_neighbors_visited;
            }
        }

        if (not mci_available) {
            continue;
        }

        // ---- lines 20-29: MCI(v_0) handled with predicate-first semantics ----
        // The budget is evaluated before the distance of a member is known, so the
        // predicate-first pass below decides membership and the distances are batched
        // afterwards. The decision sequence is unchanged.
        mci_budget.Reset();
        clique_ids.clear();
        cliques->CollectNodeCliqueIds(v_0, clique_ids);
        bool stopped = false;
        batch_ids.clear();
        for (const auto clique_id : clique_ids) {
            if (visited_cliques.Get(clique_id)) {
                continue;
            }
            visited_cliques.Set(clique_id);
            if (stats != nullptr) {
                ++stats->mci_cliques_expanded;
            }
            members.clear();
            cliques->GetCliqueMembers(clique_id, members);
            for (const auto v_1 : members) {
                if (mci_budget.ShouldStop()) {  // lines 21-22
                    stopped = true;
                    break;
                }
                if (v_1 >= total or visited_nodes.Get(v_1)) {  // lines 23-24
                    continue;
                }
                visited_nodes.Set(v_1);                       // line 25
                const bool satisfied = is_satisfactory(v_1);  // line 26
                mci_budget.Account(satisfied);
                if (stats != nullptr) {
                    ++stats->mci_members_considered;
                }
                if (not satisfied) {
                    continue;
                }
                if (stats != nullptr) {
                    ++stats->mci_members_satisfied;
                }
                batch_ids.push_back(v_1);
            }
            if (stopped) {
                break;
            }
        }
        compute_batch();  // line 27, batched over every satisfactory member of v_0
        for (size_t i = 0; i < batch_ids.size(); ++i) {
            const auto v_1 = batch_ids[i];
            const auto distance = batch_dists[i];
            heap_c.Insert(distance, v_1);  // line 28
            // Same capacity guard as the sparse path: every member here is already known to
            // be satisfactory, so the only way a Push can be wasted is a full heap whose
            // farthest entry is at least as close. Skipping it avoids the virtual dispatch.
            if ((heap_r->Size() < static_cast<uint64_t>(result_capacity) or
                 distance < heap_r->Top().first) and
                not is_nan_distance(distance)) {
                heap_r->Push(distance, v_1);  // line 29
            }
        }
        batch_ids.clear();
        if (stopped and stats != nullptr) {
            stats->mci_stopped_early = true;
        }
    }
    if (ctx != nullptr and ctx->stats != nullptr) {
        // Publish the traversal counters the way the other searchers do, so callers can
        // compare routes on stats->dist_cmp / stats->hops instead of only on wall time.
        ctx->stats->dist_cmp.fetch_add(static_cast<uint32_t>(distance_computations),
                                       std::memory_order_relaxed);
        ctx->stats->hops.fetch_add(static_cast<uint32_t>(expanded_nodes),
                                   std::memory_order_relaxed);
    }
    return heap_r;
}

}  // namespace vsag
