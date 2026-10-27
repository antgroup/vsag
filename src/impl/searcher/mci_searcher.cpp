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
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "container_types.h"
#include "impl/heap/standard_heap.h"
#include "impl/query_computer_pool.h"
#include "impl/searcher/searcher_utils.h"
#include "index_common_param.h"
#include "mci_packed_filter.h"
#include "mci_search_candidate_queue.h"
#include "simd/fp32_simd.h"
#include "vsag/filter.h"

namespace vsag {
namespace {

// NOLINTNEXTLINE(readability-identifier-naming)
struct MCIEpochMarks {
    std::vector<uint16_t> marks;
    uint16_t tag{1};
    uint64_t domain{0};

    void
    Reset(uint64_t size) {
        domain = size;
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
        return id < domain and marks[id] == tag;
    }

    void
    Set(InnerIdType id) {
        // Callers validate the domain too; keep writes safe in every build configuration.
        if (id < domain) {
            marks[id] = tag;
        }
    }
};

// Keep one scratch per active call, not just per worker: user filters can issue a
// nested search while the outer call still owns epoch marks and packed member views.
class MCISearchScratchLease {
public:
    MCISearchScratchLease() {
        if (depth == cache.size()) {
            cache.emplace_back(std::make_unique<Scratch>());
        }
        scratch_ = cache[depth++].get();
    }

    ~MCISearchScratchLease() {
        --depth;
        // Reuse the ordinary and one nested slot, but do not pin an arbitrarily deep
        // callback's full-size marks forever. Reclaim only after all views are gone.
        if (depth == 0 and cache.size() > 2) {
            cache.resize(2);
        }
    }

    MCISearchScratchLease(const MCISearchScratchLease&) = delete;
    MCISearchScratchLease&
    operator=(const MCISearchScratchLease&) = delete;

    struct Scratch {
        MCIEpochMarks visited_nodes;
        MCIEpochMarks visited_cliques;
        MCIPackedFilter packed;
    };

    [[nodiscard]] Scratch&
    Get() const {
        return *scratch_;
    }

private:
    Scratch* scratch_;
    inline static thread_local uint64_t depth{0};
    inline static thread_local std::vector<std::unique_ptr<Scratch>> cache;
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

template <bool use_raw_float>
DistHeapPtr
search_clique_view(const CliqueDataCellSearchView& view,
                   const FlattenInterfacePtr& flatten,
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
    auto result_heap = DistanceHeap::MakeInstanceBySize<true, true>(allocator, candidate_limit);
    const MCISearchScratchLease scratch;
    auto& visited_nodes = scratch.Get().visited_nodes;
    auto& visited_cliques = scratch.Get().visited_cliques;
    auto& packed = scratch.Get().packed;
    visited_nodes.Reset(total);
    visited_cliques.Reset(view.total_clique_count);
    MCISearchCandidateQueue candidates(allocator);
    candidates.Reset(static_cast<uint64_t>(candidate_limit));
    const auto* valid_bitmap =
        mci_param.valid_bitmap_size >= total ? mci_param.valid_bitmap : nullptr;
    if (mci_param.used_bitmap_fast_path != nullptr) {
        *mci_param.used_bitmap_fast_path = valid_bitmap != nullptr;
    }
    ComputerLease computer_lease;
    if constexpr (not use_raw_float) {
        computer_lease = AcquireQueryComputer(flatten, query, ctx);
    }
    const auto& computer = computer_lease.computer;
    const auto cosine_query_inv_norm = use_raw_float and metric == MetricType::METRIC_TYPE_COSINE
                                           ? calc_mci_cosine_query_inv_norm(query, dim)
                                           : 1.0F;
    const auto cosine_hold_mold =
        metric == MetricType::METRIC_TYPE_COSINE and precise_vector_stride > dim;
    std::array<InnerIdType, 64> batch_ids{};
    std::array<float, 64> batch_distances{};
    uint64_t batch_count = 0;
    uint64_t accepted = 0;
    uint32_t batch_depth = 1;
    uint32_t dist_cmp = 0;
    auto flush = [&]() {
        if (batch_count == 0) {
            return;
        }
        if constexpr (use_raw_float) {
            for (uint64_t i = 0; i < batch_count; ++i) {
                const auto* vector =
                    precise_vectors + static_cast<uint64_t>(batch_ids[i]) * precise_vector_stride;
                batch_distances[i] = mci_precise_float_distance(
                    query, vector, dim, metric, cosine_query_inv_norm, cosine_hold_mold);
            }
        } else {
            flatten->Query(batch_distances.data(), computer, batch_ids.data(), batch_count, ctx);
        }
        for (uint64_t i = 0; i < batch_count; ++i) {
            const auto distance = batch_distances[i];
            ++dist_cmp;
            if (candidates.CanUpdate(distance)) {
                candidates.Insert(distance, batch_ids[i], batch_depth);
                ++accepted;
            }
            if (is_result_distance_eligible<KNN_SEARCH>(distance, inner_search_param)) {
                result_heap->Push(distance, batch_ids[i]);
            }
        }
        batch_count = 0;
    };
    auto try_visit = [&](InnerIdType id, uint32_t depth, bool known_valid = false) {
        if (id >= total) {
            return false;
        }
        // Reject sparse byte predicates before touching the larger epoch array. A compact
        // CSR without tombstones proves liveness, so its hot path needs no second bitmap.
        if (not known_valid and valid_bitmap != nullptr and valid_bitmap[id] == 0) {
            return false;
        }
        if (visited_nodes.Get(id) or (not view.static_csr and not view.IsLiveNode(id))) {
            return false;
        }
        // Remember rejected fallback IDs too: overlapping cliques must not repeat user
        // callbacks or deletion lookups. Byte/packed rejection above remains a cheap probe.
        visited_nodes.Set(id);
        if (not known_valid and valid_bitmap == nullptr and
            inner_search_param.is_inner_id_allowed != nullptr and
            not inner_search_param.is_inner_id_allowed->CheckValid(id)) {
            return false;
        }
        if (mci_param.deleted_ids_filter != nullptr and
            not mci_param.deleted_ids_filter->CheckValid(id)) {
            return false;
        }
        batch_depth = depth;
        batch_ids[batch_count++] = id;
        if (not mci_param.batch_distance or batch_count == batch_ids.size()) {
            flush();
        }
        return true;
    };
    const auto seed_target = std::min<uint64_t>(mci_param.seed_count, total);
    const bool check_overtime = inner_search_param.time_cost != nullptr;
    uint64_t seeds = 0;
    bool timed_out = false;
    if (mci_param.seed_inner_ids != nullptr) {
        const auto count = mci_param.seed_inner_ids->size();
        const auto sampled = std::min<uint64_t>(seed_target, count);
        for (uint64_t i = 0; i < sampled; ++i) {
            if (check_overtime and mci_check_overtime(inner_search_param, ctx)) {
                timed_out = true;
                break;
            }
            seeds += try_visit((*mci_param.seed_inner_ids)[i * count / sampled], 1) ? 1 : 0;
        }
    } else {
        for (InnerIdType id = 0; id < total and seeds < seed_target; ++id) {
            if (check_overtime and mci_check_overtime(inner_search_param, ctx)) {
                timed_out = true;
                break;
            }
            seeds += try_visit(id, 1) ? 1 : 0;
        }
    }
    flush();
    const bool exhaustive =
        mci_param.enumerated_valid_count != 0 and seeds >= mci_param.enumerated_valid_count;
    const bool use_packed = mci_param.packed_filter and not timed_out and not exhaustive and
                            mci_param.depth_limit > 1 and
                            (mci_param.valid_inner_ids != nullptr or valid_bitmap != nullptr);
    if (use_packed) {
        packed.Build(total, valid_bitmap, mci_param.valid_inner_ids, mci_param.valid_inner_count);
    }
    uint32_t hops = 0;
    uint64_t idle = 0;
    while (not timed_out and not exhaustive and hops < mci_param.hops_limit) {
        if (check_overtime and mci_check_overtime(inner_search_param, ctx)) {
            break;
        }
        // Seed scoring and every preceding parent have already flushed their batches.
        auto* candidate = candidates.GetClosestUnexpanded();
        if (candidate == nullptr) {
            break;
        }
        // Copy before inserts can relocate/replace candidate storage.
        const auto id = candidate->inner_id;
        const auto depth = candidate->depth;
        if (depth >= mci_param.depth_limit) {
            continue;
        }
        const auto accepted_before = accepted;
        uint64_t considered = 0;
        view.ForEachNodeClique(id, [&](InnerIdType cid) {
            if (mci_param.activation_cap != 0 and considered++ >= mci_param.activation_cap) {
                return false;
            }
            if (cid >= view.total_clique_count or visited_cliques.Get(cid)) {
                return true;
            }
            visited_cliques.Set(cid);
            ++hops;
            if (use_packed and view.static_csr) {
                const auto begin = view.p_maxc[cid];
                const auto count = view.p_maxc[cid + 1] - begin;
                const auto& members = packed.Filter(view.maxcs + begin, count, total);
                for (auto member : members) {
                    try_visit(member, depth + 1, true);
                }
            } else {
                view.ForEachMember(cid, [&](InnerIdType member) {
                    if (not use_packed or packed.IsValid(member)) {
                        try_visit(member, depth + 1, use_packed);
                    }
                    return true;
                });
            }
            return hops < mci_param.hops_limit;
        });
        // All distances of this parent must be visible before choosing the next parent.
        flush();
        idle = accepted == accepted_before ? idle + 1 : 0;
        if (mci_param.expansion_idle_window != 0 and idle >= mci_param.expansion_idle_window) {
            break;
        }
    }

    if (ctx != nullptr and ctx->stats != nullptr) {
        ctx->stats->dist_cmp.fetch_add(dist_cmp, std::memory_order_relaxed);
        ctx->stats->hops.fetch_add(hops, std::memory_order_relaxed);
        if constexpr (use_raw_float) {
            ctx->stats->AddDistance(SearchStatistics::DistancePhase::APPROXIMATE,
                                    DistanceEvaluationBackend::FP32,
                                    dist_cmp);
        }
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
    if (cliques == nullptr or flatten == nullptr or query == nullptr) {
        return heap;
    }

    const auto total = static_cast<InnerIdType>(flatten->TotalCount());
    CliqueDataCellSearchView view;
    if (total == 0 or not cliques->TryGetSearchView(total, view)) {
        return heap;
    }
    const bool use_raw_float = mci_param.precise_vectors != nullptr and mci_param.dim > 0 and
                               mci_param.precise_vector_stride >= mci_param.dim;
    if (mci_param.used_precise_float_csr != nullptr) {
        *mci_param.used_precise_float_csr = use_raw_float;
    }
    if (use_raw_float) {
        return search_clique_view<true>(view,
                                        flatten,
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
    return search_clique_view<false>(view,
                                     flatten,
                                     nullptr,
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

}  // namespace vsag
