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

#include "impl/searcher/mci_searcher.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include "container_types.h"
#include "impl/heap/standard_heap.h"
#include "impl/query_computer_pool.h"
#include "impl/searcher/mci_clique_counts.h"
#include "impl/searcher/mci_depth_stats.h"
#include "impl/searcher/mci_epoch_marks.h"
#include "impl/searcher/mci_filtered_cliques.h"
#include "impl/searcher/mci_member_prefetch.h"
#include "impl/searcher/mci_ordered_clique_scan.h"
#include "impl/searcher/mci_search_candidate_queue.h"
#include "impl/searcher/searcher_utils.h"
#include "index_common_param.h"
#include "mci_live_searcher.h"
#include "simd/fp32_simd.h"
#include "utils/search_lock.h"
#include "vsag/filter.h"

namespace vsag {
namespace {

thread_local MCIDepthSearchStatistics depth_stats;
thread_local MCIMemberPrefetch member_prefetch;
thread_local MCIFilteredCliques<> filtered_clique_pool;

MCIFilteredCliques<>*
// NOLINTNEXTLINE(readability-identifier-naming)
PrepareFilteredCliques(const CliqueDataCellBaseView& view,
                       uint64_t total,
                       const MCISearcherParam& params,
                       bool can_expand) {
    if (not params.filtered_cliques or not can_expand or not SearchLockScope::IsNoLock() or
        params.valid_inner_ids == nullptr or params.valid_inner_count == 0) {
        return nullptr;
    }
    const auto begin = std::chrono::steady_clock::now();
    const bool built = filtered_clique_pool.Build(view.p_node_to_cid,
                                                  view.node_to_cids,
                                                  total,
                                                  view.total_clique_count,
                                                  params.valid_inner_ids,
                                                  params.valid_inner_count);
    depth_stats.filtered_clique_build_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    if (not built) {
        return nullptr;
    }
    depth_stats.filtered_cliques_used = true;
    depth_stats.filtered_valid_count = params.valid_inner_count;
    depth_stats.filtered_incidences = filtered_clique_pool.Incidences();
    depth_stats.filtered_nonempty_cliques = filtered_clique_pool.NonemptyCliques();
    depth_stats.filtered_kept_cliques = filtered_clique_pool.KeptCliques();
    depth_stats.filtered_members = filtered_clique_pool.MemberCount();
    depth_stats.filtered_capacity_bytes = filtered_clique_pool.CapacityBytes();
    return &filtered_clique_pool;
}

// MCI expansion early stop, driven by the search parameters (HGraphSearchParameters::
// mci_expansion_idle_window / mci_expansion_lower_bound).  `VSAG_MCI_EARLY_STOP` and
// `VSAG_MCI_IDLE_LIMIT` remain as debug overrides for A/B experiments.
struct MCIEarlyStopKnobs {
    bool idle_break{false};
    bool lower_break{false};
    uint64_t idle_limit{32};
};

MCIEarlyStopKnobs
mci_early_stop_knobs(const MCISearcherParam& mci_param) {
    MCIEarlyStopKnobs result;
    result.idle_break = mci_param.expansion_idle_window > 0;
    result.idle_limit = mci_param.expansion_idle_window;
    result.lower_break = mci_param.expansion_lower_bound;
    const char* mode = std::getenv("VSAG_MCI_EARLY_STOP");  // debug override
    if (mode != nullptr) {
        const std::string text(mode);
        result.idle_break = text.find("idle") != std::string::npos;
        result.lower_break = text.find("lower") != std::string::npos;
    }
    const char* limit = std::getenv("VSAG_MCI_IDLE_LIMIT");
    if (limit != nullptr) {
        result.idle_limit = std::strtoull(limit, nullptr, 10);
    }
    return result;
}

// Reuse the caller's immutable filter bitmap directly; visits use query epoch markers.
const uint8_t*
// NOLINTNEXTLINE(readability-identifier-naming)
GetMCIValidBitmap(const MCISearcherParam& mci_param, uint64_t total) {
    if (mci_param.valid_bitmap == nullptr or mci_param.valid_bitmap_size < total or
        mci_param.enumerated_valid_count != 0) {
        return nullptr;
    }
    return mci_param.valid_bitmap;
}

bool
mci_check_overtime(const InnerSearchParam& inner_search_param, QueryContext* ctx) {
    if (inner_search_param.time_cost == nullptr or
        not inner_search_param.time_cost->CheckOvertime()) {
        return false;
    }
    if (ctx != nullptr and ctx->stats != nullptr) {
        ctx->stats->is_timeout.store(true, std::memory_order_relaxed);
    }
    return true;
}

float
calc_mci_cosine_query_inv_norm(const float* query, uint64_t dim) {
    const auto norm = FP32ComputeIP(query, query, dim);
    if (norm <= 0.0F) {
        return 0.0F;
    }
    return 1.0F / std::sqrt(norm);
}

float
mci_precise_float_distance(const float* query,
                           const float* vector,
                           uint64_t dim,
                           MetricType metric,
                           float cosine_query_inv_norm,
                           bool cosine_hold_mold) {
    if (metric == MetricType::METRIC_TYPE_L2SQR) {
        return FP32ComputeL2Sqr(query, vector, dim);
    }

    auto similarity = FP32ComputeIP(query, vector, dim);
    if (metric == MetricType::METRIC_TYPE_COSINE) {
        similarity *= cosine_query_inv_norm;
        if (cosine_hold_mold) {
            if (vector[dim] <= 0.0F) {
                return 1.0F;
            }
            similarity /= vector[dim];
        }
    }
    return 1.0F - similarity;
}

// Score every point at most once. The candidate heap changes immediately after each distance,
// and clique counts track exactly that heap. A new zero-to-one clique transition requires another
// pass over still-unscored IDs: it may cover points preceding the current scan position.
template <bool ordered = false, bool hub_navigation = false>
DistHeapPtr
search_dynamic_clique_scan(const CliqueDataCellBaseView& view,
                           const FlattenInterfacePtr& flatten,
                           const void* query,
                           uint64_t total,
                           const InnerSearchParam& search,
                           const MCISearcherParam& params,
                           QueryContext* ctx,
                           Allocator* allocator) {
    const auto ef =
        static_cast<uint64_t>(std::max<int64_t>(search.topk, static_cast<int64_t>(search.ef)));
    auto result = std::make_shared<StandardHeap<true, true>>(allocator, search.topk);
    Vector<DistanceHeap::DistanceRecord> candidates(allocator);
    candidates.reserve(ef);
    thread_local MCIEpochMarks scored;
    using Counts = std::conditional_t<ordered, MCIOrderedCliqueCounts<>, MCICliqueCounts<>>;
    thread_local Counts counts;
    thread_local MCIEpochMarks navigation_expanded;
    Vector<InnerIdType> pending_hubs(allocator);
    Vector<InnerIdType> neighbors(allocator);
    Vector<InnerIdType> bridge_neighbors(allocator);
    uint64_t pending_head = 0;
    if constexpr (hub_navigation) {
        navigation_expanded.Reset(total);
        neighbors.reserve(params.hub_graph->MaximumDegree());
        bridge_neighbors.reserve(params.hub_graph->MaximumDegree());
    }
    auto is_hub = [&](InnerIdType id) {
        return view.p_node_to_cid[id + 1] - view.p_node_to_cid[id] >= params.hub_clique_threshold;
    };
    uint32_t changed_min = std::numeric_limits<uint32_t>::max();
    uint32_t changed_max = 0;
    bool record_activations = false;
    bool second_pass = false;
    uint64_t scan_position = 0;
    uint64_t last_activation_position = 0;
    uint64_t scan_limit = 0;
    uint64_t ordered_size = 0;
    scored.Reset(total);
    depth_stats.dynamic_clique_scan_used = true;
    // A dynamic union can propagate beyond two hops. This path has no depth limit; report zero
    // rather than presenting scan passes or stale parent witnesses as graph-hop measurements.
    depth_stats.hops_limit = 0;
    const auto* bitmap = GetMCIValidBitmap(params, total);
    if (params.used_bitmap_fast_path != nullptr) {
        *params.used_bitmap_fast_path = bitmap != nullptr;
    }
    const bool raw = params.precise_vectors != nullptr and params.dim > 0 and
                     params.precise_vector_stride >= params.dim;
    if (params.used_precise_float_csr != nullptr) {
        *params.used_precise_float_csr = raw;
    }
    ComputerLease lease;
    QueryContext distance_storage;
    QueryContext* distance_ctx = nullptr;
    if (not raw) {
        lease = AcquireQueryComputer(flatten, query, ctx);
        if (ctx != nullptr) {
            distance_storage = *ctx;
            distance_storage.track_distance_evaluations = false;
            distance_ctx = &distance_storage;
        }
    }
    const auto* float_query = static_cast<const float*>(query);
    const auto inv_norm = raw and params.metric == MetricType::METRIC_TYPE_COSINE
                              ? calc_mci_cosine_query_inv_norm(float_query, params.dim)
                              : 1.0F;
    const bool hold_mold = params.metric == MetricType::METRIC_TYPE_COSINE and
                           params.precise_vector_stride > params.dim;
    uint32_t distance_count = 0;
    auto add_node = [&](InnerIdType id) {
        if constexpr (hub_navigation) {
            if (is_hub(id)) {
                ++depth_stats.hub_candidates;
                depth_stats.hub_skipped_memberships +=
                    view.p_node_to_cid[id + 1] - view.p_node_to_cid[id];
                pending_hubs.push_back(id);
                return;
            }
        }
        for (auto off = view.p_node_to_cid[id]; off < view.p_node_to_cid[id + 1]; ++off) {
            ++depth_stats.dynamic_count_updates;
            if (counts.Add(view.node_to_cids[off])) {
                ++depth_stats.dynamic_activations;
                ++depth_stats.dynamic_active_cliques;
                if constexpr (ordered) {
                    if (record_activations) {
                        const auto cid = view.node_to_cids[off];
                        changed_min = std::min(changed_min, cid);
                        changed_max = std::max(changed_max, cid);
                        last_activation_position = scan_position;
                        if (second_pass) {
                            scan_limit = ordered_size;
                        }
                    }
                }
            }
        }
    };
    auto remove_node = [&](InnerIdType id) {
        if constexpr (hub_navigation) {
            if (is_hub(id)) {
                return;
            }
        }
        for (auto off = view.p_node_to_cid[id]; off < view.p_node_to_cid[id + 1]; ++off) {
            const auto cid = view.node_to_cids[off];
            ++depth_stats.dynamic_count_updates;
            if (counts.Remove(cid) and counts.Get(cid) == 0) {
                --depth_stats.dynamic_active_cliques;
            }
        }
    };
    auto score = [&](InnerIdType id, bool update_counts) {
        float distance;
        if (raw) {
            distance = mci_precise_float_distance(
                float_query,
                params.precise_vectors + uint64_t(id) * params.precise_vector_stride,
                params.dim,
                params.metric,
                inv_norm,
                hold_mold);
        } else {
            flatten->Query(&distance, lease.computer, &id, 1, distance_ctx);
        }
        ++distance_count;
        ++depth_stats.dynamic_scored_points;
        if (is_result_distance_eligible<KNN_SEARCH>(distance, search)) {
            result->Push(distance, id);
        }
        const DistanceHeap::DistanceRecord record{distance, id};
        if (candidates.size() == ef and not(record < candidates.front())) {
            return;
        }
        const bool replacing = candidates.size() == ef;
        const auto removed = replacing ? candidates.front().second : InnerIdType{0};
        if (replacing) {
            if (update_counts) {
                ++depth_stats.dynamic_replacements;
            }
            std::pop_heap(candidates.begin(), candidates.end());
            candidates.pop_back();
        }
        candidates.push_back(record);
        std::push_heap(candidates.begin(), candidates.end());
        if (update_counts) {
            // Commit one replacement before scanning another point. Adding first avoids a false
            // zero-to-one activation for a clique shared by the incoming and outgoing points.
            add_node(id);
            if (replacing) {
                remove_node(removed);
            }
        }
    };
    const bool check_time = search.time_cost != nullptr;
    bool timed_out = false;
    auto overtime = [&]() {
        timed_out = check_time and mci_check_overtime(search, ctx);
        return timed_out;
    };
    // Drain outside score(): newly accepted hubs enqueue work instead of recursively expanding.
    // Invalid neighbors may bridge one extra edge, but are never scored or put in either heap.
    auto drain_hubs = [&]() {
        if constexpr (hub_navigation) {
            auto valid = [&](InnerIdType id) {
                if (id >= total) {
                    return false;
                }
                if (bitmap != nullptr) {
                    return bitmap[id] != 0;
                }
                if (search.is_inner_id_allowed != nullptr) {
                    return search.is_inner_id_allowed->CheckValid(id);
                }
                return std::binary_search(
                    params.valid_inner_ids, params.valid_inner_ids + params.valid_inner_count, id);
            };
            auto visit = [&](InnerIdType id) {
                if (scored.TryVisit(id)) {
                    ++depth_stats.hub_distance_evaluations;
                    score(id, true);
                }
            };
            while (pending_head < pending_hubs.size() and not overtime()) {
                const auto hub = pending_hubs[pending_head++];
                const bool retained = std::any_of(candidates.begin(),
                                                  candidates.end(),
                                                  [&](const auto& p) { return p.second == hub; });
                if (not retained or not navigation_expanded.TryVisit(hub)) {
                    continue;
                }
                ++depth_stats.hub_expanded_nodes;
                params.hub_graph->GetNeighbors(hub, neighbors);
                for (const auto id : neighbors) {
                    if (overtime()) {
                        break;
                    }
                    ++depth_stats.hub_neighbor_checks;
                    if (id >= total) {
                        continue;
                    }
                    if (valid(id)) {
                        visit(id);
                    } else if (params.hub_hgraph_hops == 2 and navigation_expanded.TryVisit(id)) {
                        ++depth_stats.hub_bridge_expansions;
                        params.hub_graph->GetNeighbors(id, bridge_neighbors);
                        for (const auto next : bridge_neighbors) {
                            if (overtime()) {
                                break;
                            }
                            ++depth_stats.hub_neighbor_checks;
                            if (valid(next)) {
                                visit(next);
                            }
                        }
                    }
                }
            }
        }
    };
    const bool trusted = params.enumerated_valid_count != 0;
    auto visit_seed = [&](InnerIdType id) {
        if (id >= total or (not trusted and bitmap != nullptr and bitmap[id] == 0) or
            not scored.TryVisit(id)) {
            return false;
        }
        if (not trusted and bitmap == nullptr and search.is_inner_id_allowed != nullptr and
            not search.is_inner_id_allowed->CheckValid(id)) {
            return false;
        }
        score(id, false);
        return true;
    };
    const auto target = std::min(params.seed_count, total);
    uint64_t seeds = 0;
    if (params.seed_inner_ids != nullptr) {
        const auto available = params.seed_inner_ids->size();
        const auto sampled = std::min<uint64_t>(target, available);
        for (uint64_t i = 0; i < sampled and not overtime(); ++i) {
            seeds += visit_seed((*params.seed_inner_ids)[i * available / sampled]);
        }
    } else {
        for (uint64_t id = 0; id < total and seeds < target and not overtime(); ++id) {
            seeds += visit_seed(static_cast<InnerIdType>(id));
        }
    }
    const bool saturated =
        params.enumerated_valid_count != 0 and seeds >= params.enumerated_valid_count;
    if (params.skipped_expansion != nullptr) {
        *params.skipped_expansion = saturated;
    }
    depth_stats.dynamic_selected_seeds = candidates.size();
    if (not timed_out and not saturated and not candidates.empty()) {
        const auto begin = std::chrono::steady_clock::now();
        counts.Reset(view.total_clique_count);
        for (const auto& candidate : candidates) {
            add_node(candidate.second);
        }
        drain_hubs();
        depth_stats.dynamic_initial_union_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                .count();
        const auto scan_begin = std::chrono::steady_clock::now();
        if constexpr (ordered) {
            thread_local MCIOrderedValidRows sequences;
            const auto prepare_begin = std::chrono::steady_clock::now();
            if (counts.Minimum() != Counts::kNone) {
                sequences.Build(view.p_node_to_cid,
                                view.node_to_cids,
                                total,
                                params.valid_inner_ids,
                                params.valid_inner_count);
            } else {
                sequences.rows.clear();
                sequences.by_max.clear();
            }
            depth_stats.bounded_order_prepare_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                          prepare_begin)
                    .count();
            depth_stats.bounded_order_rows = sequences.rows.size();
            depth_stats.bounded_initial_min = counts.Minimum();
            depth_stats.bounded_initial_max = counts.Maximum();
            ordered_size = sequences.rows.size();
            scan_limit = ordered_size;
            record_activations = true;
            for (uint32_t pass = 0; pass < params.clique_scan_passes and not timed_out; ++pass) {
                if (pass != 0) {
                    if (changed_min == Counts::kNone) {
                        break;
                    }
                    second_pass = true;
                    // After the last activation in pass 1, coverage could only shrink. Its
                    // suffix is complete unless pass 2 activates another clique.
                    scan_limit = last_activation_position;
                    depth_stats.bounded_second_initial_end = scan_limit;
                }
                const auto lower =
                    second_pass ? std::max(counts.Minimum(), changed_min) : counts.Minimum();
                auto i = sequences.Start(lower);
                depth_stats.bounded_prefix_skips += i;
                if (second_pass) {
                    depth_stats.bounded_second_start = i;
                }
                if (i >= scan_limit) {
                    break;
                }
                ++depth_stats.dynamic_scan_passes;
                const auto activations_before = depth_stats.dynamic_activations;
                while (i < scan_limit) {
                    if ((i & 63U) == 0 and overtime()) {
                        break;
                    }
                    const auto low =
                        second_pass ? std::max(counts.Minimum(), changed_min) : counts.Minimum();
                    const auto high =
                        second_pass ? std::min(counts.Maximum(), changed_max) : counts.Maximum();
                    const auto& row = sequences.rows[i];
                    if (low > high or row.minimum > high) {
                        depth_stats.bounded_tail_skips += scan_limit - i;
                        break;
                    }
                    if (i % MCIOrderedValidRows::kBlock == 0 and
                        sequences.block_max[i / MCIOrderedValidRows::kBlock] < low) {
                        const auto skip = std::min(MCIOrderedValidRows::kBlock, scan_limit - i);
                        depth_stats.bounded_block_skips += skip;
                        i += skip;
                        continue;
                    }
#if defined(__GNUC__) || defined(__clang__)
                    if (i + 8 < scan_limit) {
                        const auto& next = sequences.rows[i + 8];
                        __builtin_prefetch(view.node_to_cids + next.begin, 0, 1);
                        if (next.end - next.begin > 16) {
                            __builtin_prefetch(view.node_to_cids + next.begin + 16, 0, 1);
                        }
                    }
#endif
                    ++i;
                    if (row.maximum < low) {
                        ++depth_stats.bounded_range_rejects;
                        continue;
                    }
                    if (scored.Get(row.id)) {
                        continue;
                    }
                    ++depth_stats.dynamic_point_checks;
                    const auto last = std::min(row.maximum, high);
                    auto target = counts.Next(std::max(row.minimum, low));
                    if (target > last) {
                        ++depth_stats.bounded_gap_rejects;
                        continue;
                    }
                    auto lower_bound =
                        [&](const uint32_t* begin, const uint32_t* end, uint32_t value) {
                            return std::lower_bound(begin, end, value, [](auto cid, auto bound) {
                                ++depth_stats.bounded_binary_comparisons;
                                return cid < bound;
                            });
                        };
                    const auto* end = view.node_to_cids + row.end;
                    const auto* member = view.node_to_cids + row.begin;
                    if (*member < target) {
                        ++member;
                        if (member != end and *member < target) {
                            member = lower_bound(member + 1, end, target);
                        }
                    }
                    while (member != end and *member <= last) {
                        ++depth_stats.dynamic_clique_checks;
                        if (counts.ContainsKnownValid(*member)) {
                            scored.Set(row.id);
                            scan_position = i - 1;
                            score(row.id, true);
                            drain_hubs();
                            break;
                        }
                        ++member;
                        if (params.clique_scan_gallop and member != end) {
                            target = counts.Next(*member);
                            if (target > last) {
                                break;
                            }
                            if (*member < target) {
                                ++member;
                                if (member != end and *member < target) {
                                    member = lower_bound(member + 1, end, target);
                                }
                            }
                        }
                    }
                }
                if (depth_stats.dynamic_activations == activations_before) {
                    break;
                }
            }
            depth_stats.bounded_final_min = counts.Minimum();
            depth_stats.bounded_final_max = counts.Maximum();
        } else {
            // NOLINTNEXTLINE(bugprone-infinite-loop)
            while (not timed_out) {
                const auto activations_before = depth_stats.dynamic_activations;
                ++depth_stats.dynamic_scan_passes;
                for (uint64_t i = 0; i < params.valid_inner_count; ++i) {
                    if ((i & 63U) == 0 and overtime()) {
                        break;
                    }
                    const auto id = params.valid_inner_ids[i];
                    if (id >= total or scored.Get(id)) {
                        continue;
                    }
                    ++depth_stats.dynamic_point_checks;
                    for (auto off = view.p_node_to_cid[id]; off < view.p_node_to_cid[id + 1];
                         ++off) {
                        ++depth_stats.dynamic_clique_checks;
                        if (counts.Get(view.node_to_cids[off]) != 0) {
                            scored.Set(id);
                            score(id, true);
                            drain_hubs();
                            break;
                        }
                    }
                }
                // With no new active clique, final coverage is a subset of the coverage seen earlier
                // in this pass, so no remaining unscored point can become eligible on another pass.
                if (depth_stats.dynamic_activations == activations_before) {
                    break;
                }
            }
        }
        depth_stats.dynamic_scan_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - scan_begin)
                .count();
    }
    if (ctx != nullptr and ctx->stats != nullptr) {
        ctx->stats->dist_cmp.fetch_add(distance_count, std::memory_order_relaxed);
        ctx->stats->AddDistance(SearchStatistics::DistancePhase::APPROXIMATE,
                                raw ? DistanceEvaluationBackend::FP32 : flatten->backend_,
                                distance_count);
    }
    return result;
}

DistHeapPtr
search_precise_float_csr(const CliqueDataCellBaseView& view,
                         const float* precise_vectors,
                         const float* query,
                         uint64_t dim,
                         uint64_t precise_vector_stride,
                         MetricType metric,
                         InnerIdType total,
                         const InnerSearchParam& inner_search_param,
                         const MCISearcherParam& mci_param,
                         QueryContext* ctx,
                         Allocator* allocator) {
    const auto candidate_limit =
        std::max<int64_t>(inner_search_param.topk, static_cast<int64_t>(inner_search_param.ef));
    // Keep `candidate_limit` results: the conjugate-graph enhancement consumes up to LOOK_AT_K of
    // them, and the caller truncates to the user's top-k anyway.
    auto result_heap = DistanceHeap::MakeInstanceBySize<true, true>(allocator, candidate_limit);
    thread_local MCIEpochMarks visited_nodes;
    thread_local MCIEpochMarks visited_cliques;
    MCISearchCandidateQueue candidates(allocator);
    visited_nodes.Reset(total);
    visited_cliques.Reset(view.total_clique_count);
    const auto* valid_bitmap = GetMCIValidBitmap(mci_param, total);
    if (mci_param.used_bitmap_fast_path != nullptr) {
        *mci_param.used_bitmap_fast_path = (valid_bitmap != nullptr);
    }
    candidates.Reset(static_cast<uint64_t>(candidate_limit));
    uint32_t dist_cmp = 0;

    const auto cosine_query_inv_norm = metric == MetricType::METRIC_TYPE_COSINE
                                           ? calc_mci_cosine_query_inv_norm(query, dim)
                                           : 1.0F;
    const auto cosine_hold_mold =
        metric == MetricType::METRIC_TYPE_COSINE and precise_vector_stride > dim;
    const auto early_stop = mci_early_stop_knobs(mci_param);
    DistHeapPtr kth_heap;
    if (early_stop.lower_break) {
        kth_heap = DistanceHeap::MakeInstanceBySize<true, true>(allocator, inner_search_param.topk);
    }
    uint64_t accepted = 0;
    uint64_t idle_candidates = 0;
    auto insert_candidate = [&](float distance, InnerIdType inner_id, uint32_t hops) {
        if (candidates.CanUpdate(distance)) {
            ++accepted;
            depth_stats.max_hops = std::max(depth_stats.max_hops, hops);
        }
        candidates.Insert(distance, inner_id, hops);
    };
    auto get_closest_unexpanded = [&]() -> MCISearchCandidate* {
        return candidates.GetClosestUnexpanded();
    };
    // An exhaustive seed list enumerates every valid point, so the seed phase can score it directly
    // (no filter check, no visited marks); the expansion cannot run in that case.
    const bool trusted_seed_list = mci_param.enumerated_valid_count != 0;
    auto try_visit = [&](InnerIdType inner_id,
                         uint32_t hops,
                         bool known_valid = false,
                         const float* known_distance = nullptr) -> bool {
        if (inner_id >= total) {
            return false;
        }
        if (not trusted_seed_list) {
            if (valid_bitmap != nullptr) {
                // Reject invalid members before touching the larger query-marker array.
                if ((not known_valid and valid_bitmap[inner_id] == 0) or
                    not visited_nodes.TryVisit(inner_id)) {
                    return false;
                }
            } else {
                if (not visited_nodes.TryVisit(inner_id)) {
                    return false;
                }
                if (not known_valid and inner_search_param.is_inner_id_allowed != nullptr and
                    not inner_search_param.is_inner_id_allowed->CheckValid(inner_id)) {
                    return false;
                }
            }
        }
        float dist;
        if (known_distance != nullptr) {
            dist = *known_distance;
        } else {
            const auto* vector =
                precise_vectors + static_cast<uint64_t>(inner_id) * precise_vector_stride;
            dist = mci_precise_float_distance(
                query, vector, dim, metric, cosine_query_inv_norm, cosine_hold_mold);
            ++dist_cmp;
        }
        insert_candidate(dist, inner_id, hops);
        if (is_result_distance_eligible<KNN_SEARCH>(dist, inner_search_param)) {
            result_heap->Push(dist, inner_id);
            if (kth_heap != nullptr) {
                kth_heap->Push(dist, inner_id);
            }
        }
        return true;
    };

    const auto seed_target = std::min<uint64_t>(mci_param.seed_count, total);
    const bool check_overtime = inner_search_param.time_cost != nullptr;
    uint64_t seeds = 0;
    bool timed_out = false;
    bool seed_list_provided = false;
    if (mci_param.seed_inner_ids != nullptr) {
        seed_list_provided = true;
        const auto seed_count = mci_param.seed_inner_ids->size();
        const auto sampled_seed_count = std::min<uint64_t>(seed_target, seed_count);
        for (uint64_t i = 0; i < sampled_seed_count; ++i) {
            if (check_overtime and mci_check_overtime(inner_search_param, ctx)) {
                timed_out = true;
                break;
            }
            const auto offset = i * seed_count / sampled_seed_count;
            const float* cached = mci_param.seed_distances != nullptr
                                      ? &(*mci_param.seed_distances)[offset]
                                      : nullptr;
            if (try_visit((*mci_param.seed_inner_ids)[offset], 1, false, cached)) {
                ++seeds;
            }
        }
    }
    if (not seed_list_provided) {
        for (InnerIdType seed = 0; seed < total and seeds < seed_target; ++seed) {
            if (check_overtime and mci_check_overtime(inner_search_param, ctx)) {
                timed_out = true;
                break;
            }
            if (try_visit(seed, 1)) {
                ++seeds;
            }
        }
    }

    const bool seeds_cover_all_valid_points =
        mci_param.enumerated_valid_count != 0 and seeds >= mci_param.enumerated_valid_count;
    if (mci_param.skipped_expansion != nullptr) {
        *mci_param.skipped_expansion = seeds_cover_all_valid_points;
    }
    auto* filtered = PrepareFilteredCliques(
        view,
        total,
        mci_param,
        not timed_out and not seeds_cover_all_valid_points and mci_param.hops_limit > 1);
    while (not timed_out and not seeds_cover_all_valid_points) {
        if (check_overtime and mci_check_overtime(inner_search_param, ctx)) {
            break;
        }
        auto* candidate = get_closest_unexpanded();
        if (candidate == nullptr) {
            break;
        }
        if (kth_heap != nullptr and
            kth_heap->Size() >= static_cast<uint64_t>(inner_search_param.topk) and
            candidate->distance > kth_heap->Top().first) {
            break;
        }
        // The queue is ordered by distance, so a boundary-depth point must not stop
        // shallower candidates that are still queued.
        const auto parent_hops = candidate->hops;
        if (parent_hops >= mci_param.hops_limit) {
            ++depth_stats.depth_limit_skips;
            continue;
        }
        const auto child_hops = parent_hops + 1;
        ++depth_stats.expanded_nodes;
        depth_stats.max_expanded_hops = std::max(depth_stats.max_expanded_hops, parent_hops);
        const auto accepted_before = accepted;
        const auto inner_id = candidate->inner_id;
        for (auto offset = view.p_node_to_cid[inner_id]; offset < view.p_node_to_cid[inner_id + 1];
             ++offset) {
            const auto clique_id = view.node_to_cids[offset];
            if (clique_id >= view.total_clique_count or visited_cliques.Get(clique_id)) {
                continue;
            }
            visited_cliques.Set(clique_id);
            ++depth_stats.expanded_cliques;
            const auto begin = view.p_maxc[clique_id];
            const auto count = view.p_maxc[clique_id + 1] - begin;
            if (filtered != nullptr) {
                depth_stats.filtered_source_member_checks += count;
                const auto members = filtered->Get(clique_id);
                depth_stats.member_checks += members.count;
                for (uint64_t i = 0; i < members.count; ++i) {
                    try_visit(members.data[i], child_hops, true);
                }
                continue;
            }
            depth_stats.member_checks += count;
            VisitMCIMembers(
                view.maxcs + begin,
                count,
                valid_bitmap,
                total,
                member_prefetch,
                [&](auto id) { try_visit(id, child_hops); },
                [&](auto id) { try_visit(id, child_hops, true); });
        }
        if (early_stop.idle_break) {
            if (accepted == accepted_before) {
                if (++idle_candidates >= early_stop.idle_limit) {
                    break;
                }
            } else {
                idle_candidates = 0;
            }
        }
    }

    if (ctx != nullptr and ctx->stats != nullptr) {
        ctx->stats->dist_cmp.fetch_add(dist_cmp, std::memory_order_relaxed);
        ctx->stats->hops.fetch_add(depth_stats.max_hops, std::memory_order_relaxed);
        ctx->stats->AddDistance(SearchStatistics::DistancePhase::APPROXIMATE,
                                DistanceEvaluationBackend::FP32,
                                dist_cmp);
    }
    return result_heap;
}

}  // namespace

const MCIDepthSearchStatistics&
GetMCIDepthSearchStatistics() {
    return depth_stats;
}

MCISearcher::MCISearcher(const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()) {
}

DistHeapPtr
MCISearcher::Search(const CliqueDataCellPtr& cliques,
                    const FlattenInterfacePtr& flatten,
                    const void* query,
                    const InnerSearchParam& inner_search_param,
                    const MCISearcherParam& mci_param,
                    QueryContext* ctx) const {
    depth_stats = {};
    depth_stats.hops_limit = mci_param.hops_limit;
    depth_stats.filtered_cliques_requested = mci_param.filtered_cliques;
    depth_stats.filtered_id_prepare_ms =
        mci_param.filtered_cliques ? mci_param.filtered_id_prepare_ms : 0.0;
    depth_stats.dynamic_clique_scan_requested = mci_param.dynamic_clique_scan;
    depth_stats.bounded_scan_pass_limit = mci_param.clique_scan_passes;
    depth_stats.bounded_scan_gallop = mci_param.clique_scan_gallop;
    depth_stats.hub_clique_threshold = mci_param.hub_clique_threshold;
    depth_stats.hub_hgraph_hops = mci_param.hub_hgraph_hops;
    depth_stats.dynamic_id_prepare_ms =
        mci_param.dynamic_clique_scan ? mci_param.filtered_id_prepare_ms : 0.0;
    member_prefetch = GetMCIMemberPrefetch();
    depth_stats.member_prefetch_mode = member_prefetch.mode;
    depth_stats.member_prefetch_distance = member_prefetch.distance;
    depth_stats.member_prefetch_block = member_prefetch.block;
    depth_stats.member_prefetch_locality = member_prefetch.locality;
    if (mci_param.preserve_live_traversal or not SearchLockScope::IsNoLock()) {
        return MCILiveSearcher(allocator_)
            .Search(cliques, flatten, query, inner_search_param, mci_param, ctx);
    }
    auto* alloc = select_query_allocator(ctx, allocator_);
    const auto candidate_limit =
        std::max<int64_t>(inner_search_param.topk, static_cast<int64_t>(inner_search_param.ef));
    auto heap = DistanceHeap::MakeInstanceBySize<true, true>(alloc, candidate_limit);
    const auto early_stop = mci_early_stop_knobs(mci_param);
    DistHeapPtr kth_heap;
    if (early_stop.lower_break) {
        kth_heap = DistanceHeap::MakeInstanceBySize<true, true>(alloc, inner_search_param.topk);
    }
    if (cliques == nullptr or flatten == nullptr or query == nullptr) {
        return heap;
    }

    const auto total = static_cast<InnerIdType>(flatten->TotalCount());
    if (total == 0) {
        return heap;
    }

    CliqueDataCellBaseView base_view;
    const bool no_lock_csr = SearchLockScope::IsNoLock();
    if (no_lock_csr) {
        if (not cliques->TryGetBaseViewNoLock(total, base_view)) {
            return heap;
        }
    } else if (not cliques->HasCliqueIndex(total)) {
        return heap;
    }
    if (mci_param.dynamic_clique_scan) {
        // NOLINTNEXTLINE(readability-simplify-boolean-expr)
        CHECK_ARGUMENT(no_lock_csr and not mci_param.filtered_cliques,
                       "dynamic clique scan requires an immutable no-lock CSR and no filtered CSR");
        // NOLINTNEXTLINE(readability-simplify-boolean-expr)
        CHECK_ARGUMENT(
            mci_param.enumerated_valid_count != 0 or
                (mci_param.valid_inner_ids != nullptr and mci_param.valid_inner_count > 0),
            "dynamic clique scan requires complete valid inner IDs");
        if (mci_param.hub_clique_threshold != 0) {
            // NOLINTNEXTLINE(readability-simplify-boolean-expr)
            CHECK_ARGUMENT(mci_param.hub_graph != nullptr and
                               (mci_param.hub_hgraph_hops == 1 or mci_param.hub_hgraph_hops == 2),
                           "hub navigation requires a base HGraph and one or two hops");
            if (mci_param.clique_scan_passes != 0) {
                return search_dynamic_clique_scan<true, true>(
                    base_view, flatten, query, total, inner_search_param, mci_param, ctx, alloc);
            }
            return search_dynamic_clique_scan<false, true>(
                base_view, flatten, query, total, inner_search_param, mci_param, ctx, alloc);
        }
        if (mci_param.clique_scan_passes != 0) {
            return search_dynamic_clique_scan<true>(
                base_view, flatten, query, total, inner_search_param, mci_param, ctx, alloc);
        }
        return search_dynamic_clique_scan<false>(
            base_view, flatten, query, total, inner_search_param, mci_param, ctx, alloc);
    }
    if (mci_param.precise_vectors != nullptr and mci_param.dim > 0 and
        mci_param.precise_vector_stride >= mci_param.dim and
        (no_lock_csr or cliques->TryGetBaseView(total, base_view))) {
        if (mci_param.used_precise_float_csr != nullptr) {
            *mci_param.used_precise_float_csr = true;
        }
        return search_precise_float_csr(base_view,
                                        mci_param.precise_vectors,
                                        static_cast<const float*>(query),
                                        mci_param.dim,
                                        mci_param.precise_vector_stride,
                                        mci_param.metric,
                                        total,
                                        inner_search_param,
                                        mci_param,
                                        ctx,
                                        alloc);
    }
    if (mci_param.used_precise_float_csr != nullptr) {
        *mci_param.used_precise_float_csr = false;
    }

    auto computer_lease = AcquireQueryComputer(flatten, query, ctx);
    const auto& computer = computer_lease.computer;
    // Distances are counted once at the end of the search, so the datacell must not aggregate
    // them per call -- same contract as HGraph's brute force.
    QueryContext distance_ctx_storage;
    QueryContext* distance_ctx = nullptr;
    if (ctx != nullptr) {
        distance_ctx_storage = *ctx;
        distance_ctx_storage.track_distance_evaluations = false;
        distance_ctx = &distance_ctx_storage;
    }
    thread_local MCIEpochMarks visited_nodes;
    thread_local MCIEpochMarks visited_cliques;
    visited_nodes.Reset(total);
    visited_cliques.Reset(no_lock_csr ? base_view.total_clique_count
                                      : cliques->TotalLogicalCliqueCount());
    const auto* valid_bitmap = GetMCIValidBitmap(mci_param, total);
    if (mci_param.used_bitmap_fast_path != nullptr) {
        *mci_param.used_bitmap_fast_path = (valid_bitmap != nullptr);
    }
    Vector<MCISearchCandidate> candidates(alloc);
    candidates.reserve(static_cast<uint64_t>(candidate_limit));

    auto can_update = [&](float distance) {
        return static_cast<int64_t>(candidates.size()) < candidate_limit or
               distance < candidates.back().distance;
    };
    uint64_t accepted = 0;
    uint64_t idle_candidates = 0;
    auto insert_candidate = [&](float distance, InnerIdType inner_id, uint32_t hops) {
        if (not can_update(distance)) {
            return;
        }
        MCISearchCandidate candidate{distance, inner_id, false, hops};
        auto iter = std::lower_bound(
            candidates.begin(), candidates.end(), candidate, MCISearchCandidateLess);
        candidates.insert(iter, candidate);
        if (static_cast<int64_t>(candidates.size()) > candidate_limit) {
            candidates.pop_back();
        }
        ++accepted;
        depth_stats.max_hops = std::max(depth_stats.max_hops, hops);
    };
    auto get_closest_unexpanded = [&]() -> MCISearchCandidate* {
        for (auto& candidate : candidates) {
            if (not candidate.expanded) {
                candidate.expanded = true;
                return &candidate;
            }
        }
        return nullptr;
    };
    uint32_t dist_cmp = 0;
    // An exhaustive seed list enumerates every valid point, so the seed phase can score it directly
    // (no filter check, no visited marks); the expansion cannot run in that case.
    const bool trusted_seed_list = mci_param.enumerated_valid_count != 0;
    auto try_mark = [&](InnerIdType inner_id, bool known_valid = false) -> bool {
        if (inner_id >= total) {
            return false;
        }
        if (valid_bitmap != nullptr) {
            if ((not known_valid and valid_bitmap[inner_id] == 0) or
                not visited_nodes.TryVisit(inner_id)) {
                return false;
            }
        } else {
            if (not visited_nodes.TryVisit(inner_id)) {
                return false;
            }
            if (not known_valid and inner_search_param.is_inner_id_allowed != nullptr and
                not inner_search_param.is_inner_id_allowed->CheckValid(inner_id)) {
                return false;
            }
        }
        return true;
    };
    auto score_marked = [&](float dist, InnerIdType inner_id, uint32_t hops, bool computed = true) {
        dist_cmp += computed ? 1 : 0;
        insert_candidate(dist, inner_id, hops);
        if (is_result_distance_eligible<KNN_SEARCH>(dist, inner_search_param)) {
            heap->Push(dist, inner_id);
            if (kth_heap != nullptr) {
                kth_heap->Push(dist, inner_id);
            }
        }
    };
    auto try_visit = [&](InnerIdType inner_id, uint32_t hops, bool known_valid = false) -> bool {
        if (not try_mark(inner_id, known_valid)) {
            return false;
        }
        float dist = 0.0F;
        flatten->Query(&dist, computer, &inner_id, 1, distance_ctx);
        score_marked(dist, inner_id, hops);
        return true;
    };

    const auto seed_target = std::min<uint64_t>(mci_param.seed_count, total);
    const bool check_overtime = inner_search_param.time_cost != nullptr;
    uint64_t seeds = 0;
    bool timed_out = false;
    bool seed_list_provided = false;
    if (mci_param.seed_inner_ids != nullptr) {
        seed_list_provided = true;
        const auto seed_count = mci_param.seed_inner_ids->size();
        const auto sampled_seed_count = std::min<uint64_t>(seed_target, seed_count);
        // Batch the seed distances: HGraph's brute force scores 64 ids per Query call, and a
        // single-id call pays the per-call overhead (dispatch, layout acquire, statistics) once
        // per point instead of once per 64.
        // NOLINTNEXTLINE(readability-identifier-naming)
        constexpr uint64_t kSeedBatch = 64;
        std::vector<InnerIdType> seed_batch_ids;
        std::vector<float> seed_batch_dists(kSeedBatch, 0.0F);
        seed_batch_ids.reserve(kSeedBatch);
        auto flush_seed_batch = [&]() {
            if (seed_batch_ids.empty()) {
                return;
            }
            flatten->Query(seed_batch_dists.data(),
                           computer,
                           seed_batch_ids.data(),
                           static_cast<uint64_t>(seed_batch_ids.size()),
                           distance_ctx);
            for (size_t k = 0; k < seed_batch_ids.size(); ++k) {
                score_marked(seed_batch_dists[k], seed_batch_ids[k], 1);
            }
            seed_batch_ids.clear();
        };
        for (uint64_t i = 0; i < sampled_seed_count; ++i) {
            if (check_overtime and mci_check_overtime(inner_search_param, ctx)) {
                timed_out = true;
                break;
            }
            const auto offset = i * seed_count / sampled_seed_count;
            const auto seed_inner_id = (*mci_param.seed_inner_ids)[offset];
            // An exhaustive list needs no marks and no filter check (see the float path).
            if (trusted_seed_list or try_mark(seed_inner_id)) {
                ++seeds;
                if (mci_param.seed_distances != nullptr) {
                    score_marked((*mci_param.seed_distances)[offset], seed_inner_id, 1, false);
                } else {
                    seed_batch_ids.push_back(seed_inner_id);
                    if (seed_batch_ids.size() == kSeedBatch) {
                        flush_seed_batch();
                    }
                }
            }
        }
        flush_seed_batch();
    }
    if (not seed_list_provided) {
        for (InnerIdType seed = 0; seed < total and seeds < seed_target; ++seed) {
            if (check_overtime and mci_check_overtime(inner_search_param, ctx)) {
                timed_out = true;
                break;
            }
            if (try_visit(seed, 1)) {
                ++seeds;
            }
        }
    }

    const bool seeds_cover_all_valid_points =
        mci_param.enumerated_valid_count != 0 and seeds >= mci_param.enumerated_valid_count;
    if (mci_param.skipped_expansion != nullptr) {
        *mci_param.skipped_expansion = seeds_cover_all_valid_points;
    }
    Vector<InnerIdType> clique_ids(alloc);
    Vector<InnerIdType> members(alloc);
    auto* filtered =
        no_lock_csr ? PrepareFilteredCliques(base_view,
                                             total,
                                             mci_param,
                                             not timed_out and not seeds_cover_all_valid_points and
                                                 mci_param.hops_limit > 1)
                    : nullptr;
    while (not timed_out and not seeds_cover_all_valid_points) {
        if (check_overtime and mci_check_overtime(inner_search_param, ctx)) {
            break;
        }
        auto* candidate = get_closest_unexpanded();
        if (candidate == nullptr) {
            break;
        }
        if (kth_heap != nullptr and
            kth_heap->Size() >= static_cast<uint64_t>(inner_search_param.topk) and
            candidate->distance > kth_heap->Top().first) {
            break;
        }
        // The queue is ordered by distance, so a boundary-depth point must not stop
        // shallower candidates that are still queued.
        const auto parent_hops = candidate->hops;
        if (parent_hops >= mci_param.hops_limit) {
            ++depth_stats.depth_limit_skips;
            continue;
        }
        const auto child_hops = parent_hops + 1;
        ++depth_stats.expanded_nodes;
        depth_stats.max_expanded_hops = std::max(depth_stats.max_expanded_hops, parent_hops);
        const auto accepted_before = accepted;
        if (no_lock_csr) {
            const auto inner_id = candidate->inner_id;
            for (auto offset = base_view.p_node_to_cid[inner_id];
                 offset < base_view.p_node_to_cid[inner_id + 1];
                 ++offset) {
                const auto clique_id = base_view.node_to_cids[offset];
                if (clique_id >= base_view.total_clique_count or visited_cliques.Get(clique_id)) {
                    continue;
                }
                visited_cliques.Set(clique_id);
                ++depth_stats.expanded_cliques;
                const auto begin = base_view.p_maxc[clique_id];
                const auto count = base_view.p_maxc[clique_id + 1] - begin;
                if (filtered != nullptr) {
                    depth_stats.filtered_source_member_checks += count;
                    const auto valid_members = filtered->Get(clique_id);
                    depth_stats.member_checks += valid_members.count;
                    for (uint64_t i = 0; i < valid_members.count; ++i) {
                        try_visit(valid_members.data[i], child_hops, true);
                    }
                    continue;
                }
                depth_stats.member_checks += count;
                VisitMCIMembers(
                    base_view.maxcs + begin,
                    count,
                    valid_bitmap,
                    total,
                    member_prefetch,
                    [&](auto id) { try_visit(id, child_hops); },
                    [&](auto id) { try_visit(id, child_hops, true); });
            }
        } else {
            clique_ids.clear();
            cliques->CollectNodeCliqueIds(candidate->inner_id, clique_ids);
            for (auto clique_id : clique_ids) {
                if (visited_cliques.Get(clique_id)) {
                    continue;
                }
                visited_cliques.Set(clique_id);
                ++depth_stats.expanded_cliques;
                members.clear();
                cliques->GetCliqueMembers(clique_id, members);
                depth_stats.member_checks += members.size();
                for (auto member : members) {
                    try_visit(member, child_hops);
                }
            }
        }
        if (early_stop.idle_break) {
            if (accepted == accepted_before) {
                if (++idle_candidates >= early_stop.idle_limit) {
                    break;
                }
            } else {
                idle_candidates = 0;
            }
        }
    }

    if (ctx != nullptr and ctx->stats != nullptr) {
        ctx->stats->dist_cmp.fetch_add(dist_cmp, std::memory_order_relaxed);
        ctx->stats->hops.fetch_add(depth_stats.max_hops, std::memory_order_relaxed);
        ctx->stats->AddDistance(
            SearchStatistics::DistancePhase::APPROXIMATE, flatten->backend_, dist_cmp);
    }
    return heap;
}

}  // namespace vsag
