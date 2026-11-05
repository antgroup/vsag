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
#include <limits>

#include "container_types.h"
#include "datacell/clique_datacell.h"
#include "datacell/flatten_interface.h"
#include "datacell/graph_interface.h"
#include "impl/heap/distance_heap.h"
#include "impl/inner_search_param.h"
#include "index_common_param_fwd.h"
#include "utils/pointer_define.h"

namespace vsag {

DEFINE_POINTER(HybridMCISearcher);

/**
 * Configuration of the dynamic neighbor traversal.
 *
 * The traversal expands one candidate vector v_0 at a time and handles its two
 * neighbour sources with two different neighbour-handling strategies:
 *
 *  - Hgraph(v_0) -- the sparse proximity graph. Always traversed in full
 *    (distance-compute every unvisited neighbour and push it into the candidate
 *    heap) so that the connectivity of the search subgraph does not depend on
 *    the predicate selectivity.
 *  - MCI(v_0) -- the clique companion. Traversed in predicate-first order: only
 *    the neighbours that satisfy the predicate are distance-computed, and the
 *    whole loop is cut short by traverse_should_stop().
 *
 * traverse_should_stop() is driven by the *virtual overhead* accumulated during
 * the current MCI traversal against the boundary `vob`:
 *
 *     ratio      = O_filter / O_dist
 *     stop       <=>  considered * (ratio + local_selectivity) >= vob
 *
 * so a costly predicate (large ratio) or a dense satisfactory region (large
 * local selectivity) both shrink the number of MCI neighbours that get
 * expanded, without the caller having to pick an explicit expanding-zone size.
 * vob is expressed in units of one distance computation.
 */
struct HybridTraversalParam {
    /** e_f: capacity of the candidate heap (heap_c). */
    uint64_t ef{30};
    /** k: capacity of the result heap (heap_r). */
    int64_t topk{10};
    /**
     * v_n: entry vector of the traversal. Callers normally pass the vector that
     * a coarse search already located; the default keeps a standalone call usable.
     * Ignored when seed_inner_ids is provided and non-empty.
     */
    InnerIdType entry_id{0};
    /**
     * Optional seed set. A single coarse-search entry cannot reach a highly
     * selective valid region, because the predicate removes almost the whole
     * neighbourhood around it. Supplying seeds sampled from the predicate's own
     * valid set (the same strategy the MCI route uses) is what makes the
     * traversal usable at low selectivity; each seed is inserted exactly like
     * the entry vector would be.
     */
    const Vector<InnerIdType>* seed_inner_ids{nullptr};
    /**
     * Upper bound on the number of seeds actually used. 0 means "all of them".
     * When the list is longer, seeds are taken at even strides so the sampling
     * stays spread over the whole valid set.
     */
    uint64_t seed_count{0};
    /**
     * Set when the caller knows the seed list is exactly the whole valid set. Every valid
     * point then already has a distance, so expansion cannot change the top-k and the
     * traversal answers straight from the seeds. This is the saturated case of the
     * coverage seed budget, and skipping is lossless rather than a heuristic.
     */
    bool seeds_are_exhaustive{false};
    /** Hard cap on the number of expanded v_0 vectors. */
    uint32_t hops_limit{std::numeric_limits<uint32_t>::max()};

    // -------- traverse_should_stop() budget --------
    /**
     * VOB, the virtual overhead boundary. A non-positive value disables the
     * early stop, so the MCI part degenerates into a full clique traversal --
     * useful as a correctness baseline.
     */
    double vob{0.0};
    /**
     * O_filter / O_dist: the cost of one predicate evaluation relative to one distance
     * computation. Since local_selectivity is satisfied / considered, the budget
     * accumulator expands to
     *
     *     considered * filter_cost_ratio + satisfied
     *
     * so every considered member pays the filter cost and every satisfied member adds the
     * unit distance cost. With O_dist normalised to 1 the ratio doubles as the additive
     * per-member filter overhead, which is why vob is expressed in distance-computation
     * units.
     */
    double filter_cost_ratio{1.0};
};

/** Per-search counters. Handy for tests and for tuning the budget. */
struct HybridSearchStats {
    /** Seeds pushed into the heaps before the traversal started. */
    uint64_t seeded_entries{0};
    /** v_0 vectors popped from the candidate heap. */
    uint64_t expanded_nodes{0};
    /** Unvisited Hgraph neighbours that were distance-computed. */
    uint64_t sparse_neighbors_visited{0};
    /** MCI members that passed the visited check and were accounted. */
    uint64_t mci_members_considered{0};
    /** Of those, the ones that satisfied the predicate. */
    uint64_t mci_members_satisfied{0};
    /** Distinct cliques of v_0 that were expanded. */
    uint64_t mci_cliques_expanded{0};
    /** Total distance computations issued. */
    uint64_t dist_computations{0};
    /** True when traverse_should_stop() cut an MCI traversal short. */
    bool mci_stopped_early{false};
    /** True when the traversal was skipped because the seeds already covered the valid set. */
    bool expansion_skipped{false};
};

/**
 * @brief Hybrid sparse-graph + clique-companion traversal for filtered KNN.
 *
 * Implements the dynamic neighbor traversal: one candidate heap and one result
 * heap are shared by both neighbour sources, and the predicate is the caller's
 * Filter (nullptr means "everything is satisfactory"). The entry point is
 * supplied through HybridTraversalParam::entry_id, so the class stays
 * independent of the coarse-search index layout.
 */
class HybridMCISearcher {
public:
    explicit HybridMCISearcher(const IndexCommonParam& common_param);

    HybridMCISearcher(const HybridMCISearcher&) = delete;
    HybridMCISearcher&
    operator=(const HybridMCISearcher&) = delete;

    /**
     * @param graph    sparse proximity graph, expanded as Hgraph(v_0)
     * @param cliques  MCI companion, expanded as MCI(v_0); may be nullptr
     * @param flatten  vector codes used for distance computation
     * @param query    query vector in the data cell's input format
     * @param filter   predicate p_q; nullptr accepts every vector
     * @param param    traversal configuration
     * @param ctx      optional query context (allocator, stats, computer pool)
     * @param stats    optional out-parameter with per-search counters
     * @return max-heap holding up to topk nearest satisfactory vectors
     */
    DistHeapPtr
    Search(const GraphInterfacePtr& graph,
           const CliqueDataCellPtr& cliques,
           const FlattenInterfacePtr& flatten,
           const void* query,
           const FilterPtr& filter,
           const HybridTraversalParam& param,
           QueryContext* ctx = nullptr,
           HybridSearchStats* stats = nullptr) const;

private:
    Allocator* allocator_{nullptr};
};

}  // namespace vsag
