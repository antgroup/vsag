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

#include "impl/heap/search_candidate_queue.h"
#include "impl/heap/standard_heap.h"
#include "index_common_param.h"
#include "simd/fp32_simd.h"
#include "vsag/filter.h"

namespace vsag {
namespace {

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
    auto heap = DistanceHeap::MakeInstanceBySize<true, true>(allocator, candidate_limit);
    thread_local MCIEpochMarks visited_nodes;
    thread_local MCIEpochMarks visited_cliques;
    SearchCandidateQueue candidates(allocator);
    visited_nodes.Reset(total);
    visited_cliques.Reset(view.total_clique_count);
    candidates.Reset(static_cast<uint64_t>(candidate_limit));
    uint32_t dist_cmp = 0;

    const auto cosine_query_inv_norm = metric == MetricType::METRIC_TYPE_COSINE
                                           ? calc_mci_cosine_query_inv_norm(query, dim)
                                           : 1.0F;
    const auto cosine_hold_mold =
        metric == MetricType::METRIC_TYPE_COSINE and precise_vector_stride > dim;
    auto insert_candidate = [&](float distance, InnerIdType inner_id) {
        candidates.Insert(distance, inner_id);
    };
    auto get_closest_unexpanded = [&]() -> SearchCandidate* {
        return candidates.GetClosestUnexpanded();
    };
    auto try_visit = [&](InnerIdType inner_id) -> bool {
        if (inner_id >= total or visited_nodes.Get(inner_id)) {
            return false;
        }
        visited_nodes.Set(inner_id);
        if (inner_search_param.is_inner_id_allowed != nullptr and
            not inner_search_param.is_inner_id_allowed->CheckValid(inner_id)) {
            return false;
        }
        const auto* vector =
            precise_vectors + static_cast<uint64_t>(inner_id) * precise_vector_stride;
        auto dist = mci_precise_float_distance(
            query, vector, dim, metric, cosine_query_inv_norm, cosine_hold_mold);
        ++dist_cmp;
        insert_candidate(dist, inner_id);
        return true;
    };

    const auto seed_target = std::min<uint64_t>(mci_param.seed_count, total);
    uint64_t seeds = 0;
    bool timed_out = false;
    bool seed_list_provided = false;
    if (mci_param.seed_inner_ids != nullptr) {
        seed_list_provided = true;
        const auto seed_count = mci_param.seed_inner_ids->size();
        const auto sampled_seed_count = std::min<uint64_t>(seed_target, seed_count);
        for (uint64_t i = 0; i < sampled_seed_count; ++i) {
            if (mci_check_overtime(inner_search_param, ctx)) {
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
            if (mci_check_overtime(inner_search_param, ctx)) {
                timed_out = true;
                break;
            }
            if (try_visit(seed)) {
                ++seeds;
            }
        }
    }

    uint32_t hops = 0;
    while (not timed_out and hops < mci_param.hops_limit) {
        if (mci_check_overtime(inner_search_param, ctx)) {
            break;
        }
        auto* candidate = get_closest_unexpanded();
        if (candidate == nullptr) {
            break;
        }
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
    }

    for (uint64_t i = 0; i < candidates.Size(); ++i) {
        const auto& candidate = candidates.Data()[i];
        heap->Push(candidate.distance, candidate.inner_id);
    }
    if (ctx != nullptr and ctx->stats != nullptr) {
        ctx->stats->dist_cmp.fetch_add(dist_cmp, std::memory_order_relaxed);
        ctx->stats->hops.fetch_add(hops, std::memory_order_relaxed);
    }
    return heap;
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

    auto computer = flatten->FactoryComputer(query);
    thread_local MCIEpochMarks visited_nodes;
    thread_local MCIEpochMarks visited_cliques;
    visited_nodes.Reset(total);
    visited_cliques.Reset(cliques->TotalLogicalCliqueCount());
    Vector<SearchCandidate> candidates(alloc);
    candidates.reserve(static_cast<uint64_t>(candidate_limit));

    auto can_update = [&](float distance) {
        return static_cast<int64_t>(candidates.size()) < candidate_limit or
               distance < candidates.back().distance;
    };
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
    auto try_visit = [&](InnerIdType inner_id) -> bool {
        if (inner_id >= total or visited_nodes.Get(inner_id)) {
            return false;
        }
        visited_nodes.Set(inner_id);
        if (inner_search_param.is_inner_id_allowed != nullptr and
            not inner_search_param.is_inner_id_allowed->CheckValid(inner_id)) {
            return false;
        }
        float dist = 0.0F;
        flatten->Query(&dist, computer, &inner_id, 1, ctx);
        ++dist_cmp;
        insert_candidate(dist, inner_id);
        return true;
    };

    const auto seed_target = std::min<uint64_t>(mci_param.seed_count, total);
    uint64_t seeds = 0;
    bool timed_out = false;
    bool seed_list_provided = false;
    if (mci_param.seed_inner_ids != nullptr) {
        seed_list_provided = true;
        const auto seed_count = mci_param.seed_inner_ids->size();
        const auto sampled_seed_count = std::min<uint64_t>(seed_target, seed_count);
        for (uint64_t i = 0; i < sampled_seed_count; ++i) {
            if (mci_check_overtime(inner_search_param, ctx)) {
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
            if (mci_check_overtime(inner_search_param, ctx)) {
                timed_out = true;
                break;
            }
            if (try_visit(seed)) {
                ++seeds;
            }
        }
    }

    uint32_t hops = 0;
    Vector<InnerIdType> clique_ids(alloc);
    Vector<InnerIdType> members(alloc);
    while (not timed_out and hops < mci_param.hops_limit) {
        if (mci_check_overtime(inner_search_param, ctx)) {
            break;
        }
        auto* candidate = get_closest_unexpanded();
        if (candidate == nullptr) {
            break;
        }
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
    }

    for (const auto& candidate : candidates) {
        heap->Push(candidate.distance, candidate.inner_id);
    }
    if (ctx != nullptr and ctx->stats != nullptr) {
        ctx->stats->dist_cmp.fetch_add(dist_cmp, std::memory_order_relaxed);
        ctx->stats->hops.fetch_add(hops, std::memory_order_relaxed);
    }
    return heap;
}

}  // namespace vsag
