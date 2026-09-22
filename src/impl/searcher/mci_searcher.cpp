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

#include "mci_searcher.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "container_types.h"
#include "impl/heap/search_candidate_queue.h"
#include "impl/heap/standard_heap.h"
#include "impl/query_computer_pool.h"
#include "impl/searcher/searcher_utils.h"
#include "index_common_param.h"
#include "simd/fp32_simd.h"
#include "vsag/filter.h"

namespace vsag {
namespace {

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

// NOLINTNEXTLINE(readability-identifier-naming)
struct MCIEpochMarks {
    std::vector<uint16_t> marks;
    uint16_t tag{1};

    void
    Reset(uint64_t size) {
        if (marks.size() < size) {
            marks.assign(size, 0);
            tag = 1;
            return;
        }
        ++tag;
        if (tag == 0) {
            std::memset(marks.data(), 0, marks.size() * sizeof(uint16_t));
            tag = 1;
        }
    }

    [[nodiscard]] bool
    Get(InnerIdType id) const {
        return id < marks.size() and marks[id] == tag;
    }

    void
    Set(InnerIdType id) {
        if (id >= marks.size()) {
            marks.resize(static_cast<uint64_t>(id) + 1, 0);
        }
        marks[id] = tag;
    }
};

// Merge "filtered out" and "visited" into one byte array, so a probe answers both questions with a
// single memory access and without a virtual Filter::CheckValid() call. The state is rebuilt per
// search from the filter bitmap: the probe loop walks the index in random order, so faulting in
// one dense array is cheaper than faulting in two sparse ones per probe.
// NOLINTNEXTLINE(readability-identifier-naming)
struct MCIMergedVisitMarks {
    // Scratch for one search: the marks must not outlive it, because they are allocated from the
    // allocator the search was given, which the caller owns and can destroy together with the index.
    Vector<uint8_t> state;

    explicit MCIMergedVisitMarks(Allocator* allocator) : state(allocator) {
    }

    void
    Reset(const uint8_t* valid_bitmap, uint64_t total) {
        if (state.size() < total) {
            state.resize(total);
        }
        // The destination is reached through a local pointer: indexing the vector member directly
        // makes the compiler reload its data pointer on every iteration and keeps the loop scalar.
        auto* destination = state.data();
        for (uint64_t id = 0; id < total; ++id) {
            // Normalize: any non-zero bitmap value means valid, 0 stays filtered out.
            destination[id] = valid_bitmap[id] == 0 ? 0 : 1;
        }
    }

    // Returns true only for a valid id that was not visited yet; 2 records the visit.
    [[nodiscard]] bool
    TryVisit(InnerIdType id) {
        if (id >= state.size()) {
            return false;
        }
        uint8_t& value = state[id];
        if (value != 1) {
            return false;
        }
        value = 2;
        return true;
    }
};

// Returns the filter bitmap when the searcher may index it with inner ids, and prepares the merged
// marks in that case. Callers must already have validated the id space of the provider.
const uint8_t*
// NOLINTNEXTLINE(readability-identifier-naming)
PrepareMergedVisitMarks(const MCISearcherParam& mci_param,
                        Allocator* allocator,
                        uint64_t total,
                        MCIMergedVisitMarks& merged_marks) {
    if (mci_param.valid_bitmap == nullptr or allocator == nullptr or
        mci_param.valid_bitmap_size < total) {
        return nullptr;
    }
    // An exhaustive seed list already is the snapshot of the valid set, so the merged marks would
    // never be read: the saturation guard stops the search right after the seed phase.  Skip the
    // O(total) materialisation in that case.
    if (mci_param.enumerated_valid_count != 0) {
        return nullptr;
    }
    merged_marks.Reset(mci_param.valid_bitmap, total);
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
    SearchCandidateQueue candidates(allocator);
    visited_nodes.Reset(total);
    visited_cliques.Reset(view.total_clique_count);
    MCIMergedVisitMarks merged_marks(allocator);
    const auto* valid_bitmap = PrepareMergedVisitMarks(mci_param, allocator, total, merged_marks);
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
    auto insert_candidate = [&](float distance, InnerIdType inner_id) {
        if (candidates.CanUpdate(distance)) {
            ++accepted;
        }
        candidates.Insert(distance, inner_id);
    };
    auto get_closest_unexpanded = [&]() -> SearchCandidate* {
        return candidates.GetClosestUnexpanded();
    };
    // An exhaustive seed list enumerates every valid point, so the seed phase can score it directly
    // (no filter check, no visited marks); the expansion cannot run in that case.
    const bool trusted_seed_list = mci_param.enumerated_valid_count != 0;
    auto try_visit = [&](InnerIdType inner_id) -> bool {
        if (inner_id >= total) {
            return false;
        }
        if (not trusted_seed_list) {
            if (valid_bitmap != nullptr) {
                // The merged marks answer "already visited" and "filtered out" in one access.
                if (not merged_marks.TryVisit(inner_id)) {
                    return false;
                }
            } else {
                if (visited_nodes.Get(inner_id)) {
                    return false;
                }
                visited_nodes.Set(inner_id);
                if (inner_search_param.is_inner_id_allowed != nullptr and
                    not inner_search_param.is_inner_id_allowed->CheckValid(inner_id)) {
                    return false;
                }
            }
        }
        const auto* vector =
            precise_vectors + static_cast<uint64_t>(inner_id) * precise_vector_stride;
        auto dist = mci_precise_float_distance(
            query, vector, dim, metric, cosine_query_inv_norm, cosine_hold_mold);
        ++dist_cmp;
        insert_candidate(dist, inner_id);
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
            if (try_visit((*mci_param.seed_inner_ids)[offset])) {
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
            if (try_visit(seed)) {
                ++seeds;
            }
        }
    }

    const bool seeds_cover_all_valid_points =
        mci_param.enumerated_valid_count != 0 and seeds >= mci_param.enumerated_valid_count;
    if (mci_param.skipped_expansion != nullptr) {
        *mci_param.skipped_expansion = seeds_cover_all_valid_points;
    }
    uint32_t hops = 0;
    while (not timed_out and not seeds_cover_all_valid_points and hops < mci_param.hops_limit) {
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
        const auto accepted_before = accepted;
        const auto inner_id = candidate->inner_id;
        for (auto offset = view.p_node_to_cid[inner_id]; offset < view.p_node_to_cid[inner_id + 1];
             ++offset) {
            const auto clique_id = view.node_to_cids[offset];
            if (clique_id >= view.total_clique_count or visited_cliques.Get(clique_id)) {
                continue;
            }
            visited_cliques.Set(clique_id);
            ++hops;
            for (auto member_offset = view.p_maxc[clique_id];
                 member_offset < view.p_maxc[clique_id + 1];
                 ++member_offset) {
                try_visit(view.maxcs[member_offset]);
            }
            if (hops >= mci_param.hops_limit) {
                break;
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
        ctx->stats->hops.fetch_add(hops, std::memory_order_relaxed);
        ctx->stats->AddDistance(SearchStatistics::DistancePhase::APPROXIMATE,
                                DistanceEvaluationBackend::FP32,
                                dist_cmp);
    }
    return result_heap;
}

}  // namespace

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
    if (total == 0 or not cliques->HasCliqueIndex(total)) {
        return heap;
    }

    CliqueDataCellBaseView base_view;
    if (mci_param.precise_vectors != nullptr and mci_param.dim > 0 and
        mci_param.precise_vector_stride >= mci_param.dim and
        cliques->TryGetBaseView(total, base_view)) {
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
    visited_cliques.Reset(cliques->TotalLogicalCliqueCount());
    MCIMergedVisitMarks merged_marks(alloc);
    const auto* valid_bitmap = PrepareMergedVisitMarks(mci_param, alloc, total, merged_marks);
    if (mci_param.used_bitmap_fast_path != nullptr) {
        *mci_param.used_bitmap_fast_path = (valid_bitmap != nullptr);
    }
    Vector<SearchCandidate> candidates(alloc);
    candidates.reserve(static_cast<uint64_t>(candidate_limit));

    auto can_update = [&](float distance) {
        return static_cast<int64_t>(candidates.size()) < candidate_limit or
               distance < candidates.back().distance;
    };
    uint64_t accepted = 0;
    uint64_t idle_candidates = 0;
    auto insert_candidate = [&](float distance, InnerIdType inner_id) {
        if (not can_update(distance)) {
            return;
        }
        SearchCandidate candidate{distance, inner_id, false};
        auto iter =
            std::lower_bound(candidates.begin(), candidates.end(), candidate, SearchCandidateLess);
        candidates.insert(iter, candidate);
        if (static_cast<int64_t>(candidates.size()) > candidate_limit) {
            candidates.pop_back();
        }
        ++accepted;
    };
    auto get_closest_unexpanded = [&]() -> SearchCandidate* {
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
    auto try_mark = [&](InnerIdType inner_id) -> bool {
        if (inner_id >= total) {
            return false;
        }
        if (valid_bitmap != nullptr) {
            // The merged marks answer "already visited" and "filtered out" in one access.
            if (not merged_marks.TryVisit(inner_id)) {
                return false;
            }
        } else {
            if (visited_nodes.Get(inner_id)) {
                return false;
            }
            visited_nodes.Set(inner_id);
            if (inner_search_param.is_inner_id_allowed != nullptr and
                not inner_search_param.is_inner_id_allowed->CheckValid(inner_id)) {
                return false;
            }
        }
        return true;
    };
    auto score_marked = [&](float dist, InnerIdType inner_id) {
        ++dist_cmp;
        insert_candidate(dist, inner_id);
        if (is_result_distance_eligible<KNN_SEARCH>(dist, inner_search_param)) {
            heap->Push(dist, inner_id);
            if (kth_heap != nullptr) {
                kth_heap->Push(dist, inner_id);
            }
        }
    };
    auto try_visit = [&](InnerIdType inner_id) -> bool {
        if (not try_mark(inner_id)) {
            return false;
        }
        float dist = 0.0F;
        flatten->Query(&dist, computer, &inner_id, 1, distance_ctx);
        score_marked(dist, inner_id);
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
                score_marked(seed_batch_dists[k], seed_batch_ids[k]);
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
                seed_batch_ids.push_back(seed_inner_id);
                if (seed_batch_ids.size() == kSeedBatch) {
                    flush_seed_batch();
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
            if (try_visit(seed)) {
                ++seeds;
            }
        }
    }

    const bool seeds_cover_all_valid_points =
        mci_param.enumerated_valid_count != 0 and seeds >= mci_param.enumerated_valid_count;
    if (mci_param.skipped_expansion != nullptr) {
        *mci_param.skipped_expansion = seeds_cover_all_valid_points;
    }
    uint32_t hops = 0;
    Vector<InnerIdType> clique_ids(alloc);
    Vector<InnerIdType> members(alloc);
    while (not timed_out and not seeds_cover_all_valid_points and hops < mci_param.hops_limit) {
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
        const auto accepted_before = accepted;
        clique_ids.clear();
        cliques->CollectNodeCliqueIds(candidate->inner_id, clique_ids);
        for (auto clique_id : clique_ids) {
            if (visited_cliques.Get(clique_id)) {
                continue;
            }
            visited_cliques.Set(clique_id);
            ++hops;
            members.clear();
            cliques->GetCliqueMembers(clique_id, members);
            for (auto member : members) {
                try_visit(member);
            }
            if (hops >= mci_param.hops_limit) {
                break;
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
        ctx->stats->hops.fetch_add(hops, std::memory_order_relaxed);
        ctx->stats->AddDistance(
            SearchStatistics::DistancePhase::APPROXIMATE, flatten->backend_, dist_cmp);
    }
    return heap;
}

}  // namespace vsag
