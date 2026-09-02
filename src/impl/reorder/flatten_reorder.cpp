
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

#include "flatten_reorder.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <numeric>

#include "datacell/flatten_interface.h"
#include "datacell/rabitq_split_datacell.h"
#include "impl/filter/iterator_filter.h"
#include "impl/heap/standard_heap.h"
#include "impl/query_computer_pool.h"
#include "impl/reasoning/search_reasoning.h"
#include "query_context.h"

namespace vsag {

void
FlattenReorder::QueryFusedLowerBound(float* distances,
                                     float* lower_bounds,
                                     float* filter_inner_products,
                                     RaBitQSplitDataCellInterface* split_codes,
                                     const ComputerInterfacePtr& computer,
                                     const InnerIdType* ids,
                                     uint64_t count,
                                     QueryContext* ctx) const {
    uint32_t fallback_count = 0;
    QueryContext rate_context;
    QueryContext* rate_context_ptr = nullptr;
    if (ctx != nullptr) {
        rate_context.rabitq_error_rate = ctx->rabitq_error_rate;
        rate_context_ptr = &rate_context;
    }
    for (uint64_t i = 0; i < count; ++i) {
        const auto code = fused_graph_->GetCodeView(ids[i]);
        if (not split_codes->ComputeFusedOneBitWithFilterIP(computer,
                                                            code.cluster_id,
                                                            code.one_bit_code,
                                                            code.supplement_code,
                                                            distances + i,
                                                            lower_bounds + i,
                                                            filter_inner_products + i,
                                                            rate_context_ptr)) {
            distances[i] = std::numeric_limits<float>::max();
            lower_bounds[i] = std::numeric_limits<float>::max();
            filter_inner_products[i] = std::numeric_limits<float>::quiet_NaN();
            ++fallback_count;
        }
    }
    if (ctx != nullptr and ctx->stats != nullptr) {
        if (ctx->track_distance_evaluations and count > 0) {
            ctx->stats->AddDistance(ctx->distance_phase, DistanceEvaluationBackend::RABITQ, count);
        }
        ctx->stats->rabitq_filter_count.fetch_add(static_cast<uint32_t>(count),
                                                  std::memory_order_relaxed);
        ctx->stats->rabitq_filter_fallback_full_count.fetch_add(fallback_count,
                                                                std::memory_order_relaxed);
    }
}

void
FlattenReorder::QueryFusedFullWithHint(float* distances,
                                       const float* filter_inner_products,
                                       RaBitQSplitDataCellInterface* split_codes,
                                       const ComputerInterfacePtr& computer,
                                       const InnerIdType* ids,
                                       uint64_t count,
                                       QueryContext* ctx) const {
    uint32_t hint_full_count = 0;
    uint32_t fallback_full_count = 0;
    const bool exact_filter_ip_hint =
        split_codes->FusedFilterBits() >= 2 and not split_codes->UsesLegacyHnswFusedCodec();
    for (uint64_t i = 0; i < count; ++i) {
        const auto code = fused_graph_->GetCodeView(ids[i]);
        bool used_hint = false;
        if (exact_filter_ip_hint and IsFiniteRaBitQValue(filter_inner_products[i])) {
            used_hint = split_codes->ComputeFusedFullWithFilterIP(computer,
                                                                  code.cluster_id,
                                                                  code.one_bit_code,
                                                                  code.supplement_code,
                                                                  filter_inner_products[i],
                                                                  distances + i,
                                                                  nullptr);
        }
        if (used_hint) {
            ++hint_full_count;
        } else {
            ++fallback_full_count;
            if (not split_codes->ComputeFusedFull(computer,
                                                  code.cluster_id,
                                                  code.one_bit_code,
                                                  code.supplement_code,
                                                  distances + i,
                                                  nullptr)) {
                distances[i] = std::numeric_limits<float>::max();
            }
        }
    }
    if (ctx != nullptr and ctx->stats != nullptr) {
        if (ctx->track_distance_evaluations and count > 0) {
            ctx->stats->AddDistance(ctx->distance_phase, DistanceEvaluationBackend::RABITQ, count);
        }
        ctx->stats->rabitq_full_count.fetch_add(static_cast<uint32_t>(count),
                                                std::memory_order_relaxed);
        ctx->stats->rabitq_reorder_hint_full_count.fetch_add(hint_full_count,
                                                             std::memory_order_relaxed);
        ctx->stats->rabitq_reorder_fallback_full_count.fetch_add(fallback_full_count,
                                                                 std::memory_order_relaxed);
    }
}

namespace {

void
add_reorder_distance_count(QueryContext& ctx, uint64_t count) {
    if (ctx.stats != nullptr) {
        ctx.stats->reorder_distance_count.fetch_add(static_cast<uint32_t>(count),
                                                    std::memory_order_relaxed);
    }
}

void
add_reorder_lower_bound_probe_count(QueryContext& ctx, uint64_t count) {
    if (ctx.stats != nullptr) {
        ctx.stats->reorder_lower_bound_probe_count.fetch_add(static_cast<uint32_t>(count),
                                                             std::memory_order_relaxed);
    }
}

}  // namespace

DistHeapPtr
FlattenReorder::Reorder(const vsag::DistHeapPtr& input,
                        const void* query,
                        int64_t topk,
                        QueryContext& ctx,
                        IteratorFilterContext* iter_ctx,
                        const DistanceRecordVector* rabitq_lower_bound_candidates,
                        const std::optional<float>& distance_threshold) {
    // set query allocator
    Allocator* query_allocator = select_query_allocator(ctx.alloc, allocator_);
    auto is_distance_eligible = [&distance_threshold](float distance) {
        return not distance_threshold.has_value() or
               (IsFiniteRaBitQValue(distance) and distance <= distance_threshold.value());
    };
    auto consume_if_ineligible = [&](float distance, InnerIdType id) {
        if (is_distance_eligible(distance)) {
            return false;
        }
        if (iter_ctx != nullptr) {
            iter_ctx->SetPoint(id);
        }
        return true;
    };
    const uint64_t heap_candidate_size = input == nullptr ? 0 : input->Size();
    if (rabitq_lower_bound_candidates == nullptr) {
        topk = std::min(topk, static_cast<int64_t>(heap_candidate_size));
        auto reorder_heap = std::make_shared<StandardHeap<true, false>>(query_allocator, topk);
        auto computer_lease = AcquireQueryComputer(flatten_, query, &ctx);
        const auto& computer = computer_lease.computer;
        Vector<InnerIdType> ids(heap_candidate_size, query_allocator);
        Vector<float> dists(heap_candidate_size, query_allocator);
        const auto* candidate_result = input == nullptr ? nullptr : input->GetData();
        for (uint64_t i = 0; i < heap_candidate_size; ++i) {
            ids[i] = candidate_result[i].second;
        }
        add_reorder_distance_count(ctx, heap_candidate_size);
        {
            ScopedDistancePhase scoped(ctx, DistanceEvaluationPhase::RERANK);
            flatten_->Query(dists.data(), computer, ids.data(), heap_candidate_size, &ctx);
        }
        for (uint64_t i = 0; i < heap_candidate_size; ++i) {
            if (ctx.reasoning_ctx != nullptr) {
                ctx.reasoning_ctx->RecordReorder(
                    candidate_result[i].second, candidate_result[i].first, dists[i]);
            }
            if (consume_if_ineligible(dists[i], candidate_result[i].second)) {
                continue;
            }
            if (reorder_heap->Size() < topk || dists[i] < reorder_heap->Top().first) {
                reorder_heap->Push(dists[i], candidate_result[i].second);
                if (reorder_heap->Size() > topk) {
                    if (iter_ctx != nullptr) {
                        auto curr = reorder_heap->Top();
                        iter_ctx->AddDiscardNode(curr.first, curr.second);
                    }
                    if (ctx.reasoning_ctx != nullptr) {
                        ctx.reasoning_ctx->RecordReorderEviction(reorder_heap->Top().second, 0);
                    }
                    reorder_heap->Pop();
                }
            }
        }
        return reorder_heap;
    }

    const uint64_t rabitq_candidate_size =
        rabitq_lower_bound_candidates == nullptr ? 0 : rabitq_lower_bound_candidates->size();
    const uint64_t max_candidate_size = heap_candidate_size + rabitq_candidate_size;
    if (topk <= 0) {
        topk = static_cast<int64_t>(max_candidate_size);
    }
    auto computer_lease = AcquireQueryComputer(flatten_, query, &ctx);
    const auto& computer = computer_lease.computer;
    if (topk == 0 || max_candidate_size == 0) {
        return std::make_shared<StandardHeap<true, false>>(query_allocator, 0);
    }

    Vector<InnerIdType> all_ids(max_candidate_size, query_allocator);
    Vector<float> lower_bound_probe_dists(max_candidate_size, query_allocator);
    Vector<float> lower_bounds(max_candidate_size, query_allocator);
    UnorderedSet<InnerIdType> merged_ids(query_allocator);
    merged_ids.reserve(max_candidate_size);

    uint64_t candidate_size = 0;
    const auto* candidate_result = input == nullptr ? nullptr : input->GetData();
    for (uint64_t i = 0; i < heap_candidate_size; ++i) {
        const auto id = candidate_result[i].second;
        if (merged_ids.insert(id).second) {
            all_ids[candidate_size++] = id;
        }
    }
    const uint64_t heap_unique_size = candidate_size;
    if (heap_unique_size > 0) {
        add_reorder_lower_bound_probe_count(ctx, heap_unique_size);
        ScopedDistancePhase scoped_phase(ctx, DistanceEvaluationPhase::RERANK);
        flatten_->QueryWithDistanceLowerBound(lower_bound_probe_dists.data(),
                                              lower_bounds.data(),
                                              computer,
                                              all_ids.data(),
                                              heap_unique_size,
                                              &ctx);
    }

    if (rabitq_lower_bound_candidates != nullptr) {
        for (const auto& item : *rabitq_lower_bound_candidates) {
            if (merged_ids.insert(item.second).second) {
                all_ids[candidate_size] = item.second;
                lower_bound_probe_dists[candidate_size] = item.first;
                lower_bounds[candidate_size] = item.first;
                ++candidate_size;
            }
        }
    }

    topk = std::min(topk, static_cast<int64_t>(candidate_size));
    auto reorder_heap = std::make_shared<StandardHeap<true, false>>(query_allocator, topk);
    if (topk == 0 || candidate_size == 0) {
        return reorder_heap;
    }

    auto has_valid_lower_bound = [](float lower_bound) {
        return IsFiniteRaBitQValue(lower_bound) and lower_bound < std::numeric_limits<float>::max();
    };
    bool lower_bounds_available = true;
    for (uint64_t i = 0; i < candidate_size; ++i) {
        if (not has_valid_lower_bound(lower_bounds[i])) {
            lower_bounds_available = false;
            break;
        }
    }

    if (not lower_bounds_available) {
        add_reorder_distance_count(ctx, candidate_size);
        {
            ScopedDistancePhase scoped(ctx, DistanceEvaluationPhase::RERANK);
            flatten_->Query(
                lower_bound_probe_dists.data(), computer, all_ids.data(), candidate_size, &ctx);
        }
        for (uint64_t i = 0; i < candidate_size; ++i) {
            if (ctx.reasoning_ctx != nullptr) {
                ctx.reasoning_ctx->RecordReorder(
                    all_ids[i], lower_bounds[i], lower_bound_probe_dists[i]);
            }
            if (consume_if_ineligible(lower_bound_probe_dists[i], all_ids[i])) {
                continue;
            }
            if (reorder_heap->Size() < topk or
                lower_bound_probe_dists[i] < reorder_heap->Top().first) {
                reorder_heap->Push(lower_bound_probe_dists[i], all_ids[i]);
                if (reorder_heap->Size() > topk) {
                    if (iter_ctx != nullptr) {
                        auto curr = reorder_heap->Top();
                        iter_ctx->AddDiscardNode(curr.first, curr.second);
                    }
                    if (ctx.reasoning_ctx != nullptr) {
                        ctx.reasoning_ctx->RecordReorderEviction(reorder_heap->Top().second, 0);
                    }
                    reorder_heap->Pop();
                }
            }
        }
        return reorder_heap;
    }

    Vector<uint64_t> order(candidate_size, query_allocator);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&lower_bounds](uint64_t lhs, uint64_t rhs) {
        return lower_bounds[lhs] < lower_bounds[rhs];
    });

    const uint64_t bootstrap_size = std::min<uint64_t>(static_cast<uint64_t>(topk), candidate_size);
    constexpr uint64_t batch_size = 256;
    const auto buffer_size = std::max<uint64_t>(bootstrap_size, batch_size);
    Vector<InnerIdType> ids(buffer_size, query_allocator);
    Vector<float> dists(buffer_size, query_allocator);
    Vector<float> hint_dists(buffer_size, query_allocator);
    Vector<uint64_t> batch_indices(buffer_size, query_allocator);

    for (uint64_t i = 0; i < bootstrap_size; ++i) {
        const auto idx = order[i];
        ids[i] = all_ids[idx];
        hint_dists[i] = idx < heap_unique_size ? lower_bound_probe_dists[idx]
                                               : std::numeric_limits<float>::max();
    }
    add_reorder_distance_count(ctx, bootstrap_size);
    {
        ScopedDistancePhase scoped(ctx, DistanceEvaluationPhase::RERANK);
        flatten_->QueryWithDistanceHint(
            dists.data(), hint_dists.data(), computer, ids.data(), bootstrap_size, &ctx);
    }
    for (uint64_t i = 0; i < bootstrap_size; ++i) {
        if (ctx.reasoning_ctx != nullptr) {
            const auto idx = order[i];
            ctx.reasoning_ctx->RecordReorder(ids[i], lower_bound_probe_dists[idx], dists[i]);
        }
        if (not consume_if_ineligible(dists[i], ids[i])) {
            reorder_heap->Push(dists[i], ids[i]);
        }
    }

    uint64_t cursor = bootstrap_size;
    while (cursor < candidate_size) {
        if (reorder_heap->Size() == topk &&
            lower_bounds[order[cursor]] >= reorder_heap->Top().first) {
            break;
        }

        const auto pruning_threshold = reorder_heap->Size() == topk
                                           ? reorder_heap->Top().first
                                           : std::numeric_limits<float>::max();
        uint64_t batch_count = 0;
        while (cursor < candidate_size && batch_count < batch_size) {
            const auto idx = order[cursor];
            if (lower_bounds[idx] >= pruning_threshold) {
                break;
            }
            ids[batch_count] = all_ids[idx];
            hint_dists[batch_count] = idx < heap_unique_size ? lower_bound_probe_dists[idx]
                                                             : std::numeric_limits<float>::max();
            batch_indices[batch_count] = idx;
            ++batch_count;
            ++cursor;
        }

        if (batch_count == 0) {
            break;
        }

        add_reorder_distance_count(ctx, batch_count);
        {
            ScopedDistancePhase scoped(ctx, DistanceEvaluationPhase::RERANK);
            flatten_->QueryWithDistanceHint(
                dists.data(), hint_dists.data(), computer, ids.data(), batch_count, &ctx);
        }
        for (uint64_t i = 0; i < batch_count; ++i) {
            if (ctx.reasoning_ctx != nullptr) {
                ctx.reasoning_ctx->RecordReorder(
                    ids[i], lower_bound_probe_dists[batch_indices[i]], dists[i]);
            }
            if (consume_if_ineligible(dists[i], ids[i])) {
                continue;
            }
            if (reorder_heap->Size() < topk or dists[i] < reorder_heap->Top().first) {
                reorder_heap->Push(dists[i], ids[i]);
                if (reorder_heap->Size() > topk) {
                    if (iter_ctx != nullptr) {
                        auto curr = reorder_heap->Top();
                        iter_ctx->AddDiscardNode(curr.first, curr.second);
                    }
                    if (ctx.reasoning_ctx != nullptr) {
                        ctx.reasoning_ctx->RecordReorderEviction(reorder_heap->Top().second, 0);
                    }
                    reorder_heap->Pop();
                }
            }
        }
    }
    return reorder_heap;
}

DistHeapPtr
FlattenReorder::ReorderFused(const vsag::DistHeapPtr& input,
                             const void* query,
                             int64_t topk,
                             QueryContext& ctx,
                             IteratorFilterContext* iter_ctx,
                             const RaBitQCandidateVector* rabitq_lower_bound_candidates,
                             const std::optional<float>& distance_threshold) {
    auto* split_codes = dynamic_cast<RaBitQSplitDataCellInterface*>(flatten_.get());
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        fused_graph_ != nullptr and split_codes != nullptr,
        "fused reorder requires fused graph and RaBitQ split codes");
    // set query allocator
    Allocator* query_allocator = select_query_allocator(ctx.alloc, allocator_);
    auto is_distance_eligible = [&distance_threshold](float distance) {
        return not distance_threshold.has_value() or
               (IsFiniteRaBitQValue(distance) and distance <= distance_threshold.value());
    };
    auto consume_if_ineligible = [&](float distance, InnerIdType id) {
        if (is_distance_eligible(distance)) {
            return false;
        }
        if (iter_ctx != nullptr) {
            iter_ctx->SetPoint(id);
        }
        return true;
    };
    const auto has_valid_distance = [](float distance) {
        return IsFiniteRaBitQValue(distance) and distance < std::numeric_limits<float>::max();
    };
    const uint64_t heap_candidate_size = input == nullptr ? 0 : input->Size();
    const auto add_iterator_discard = [iter_ctx](float distance, InnerIdType id) {
        if (iter_ctx != nullptr) {
            iter_ctx->AddDiscardNode(distance, id);
        }
    };
    if (rabitq_lower_bound_candidates == nullptr) {
        topk = std::min(topk, static_cast<int64_t>(heap_candidate_size));
        auto reorder_heap = std::make_shared<StandardHeap<true, false>>(query_allocator, topk);
        auto computer = split_codes->FactoryFusedComputer(query);
        Vector<InnerIdType> ids(heap_candidate_size, query_allocator);
        Vector<float> dists(heap_candidate_size, query_allocator);
        const auto* candidate_result = input == nullptr ? nullptr : input->GetData();
        for (uint64_t i = 0; i < heap_candidate_size; ++i) {
            ids[i] = candidate_result[i].second;
        }
        add_reorder_distance_count(ctx, heap_candidate_size);
        {
            ScopedDistancePhase scoped(ctx, DistanceEvaluationPhase::RERANK);
            for (uint64_t i = 0; i < heap_candidate_size; ++i) {
                const auto code = fused_graph_->GetCodeView(ids[i]);
                if (not split_codes->ComputeFusedFull(computer,
                                                      code.cluster_id,
                                                      code.one_bit_code,
                                                      code.supplement_code,
                                                      dists.data() + i,
                                                      &ctx)) {
                    dists[i] = std::numeric_limits<float>::max();
                }
            }
        }
        for (uint64_t i = 0; i < heap_candidate_size; ++i) {
            if (not has_valid_distance(dists[i])) {
                continue;
            }
            if (ctx.reasoning_ctx != nullptr) {
                ctx.reasoning_ctx->RecordReorder(
                    candidate_result[i].second, candidate_result[i].first, dists[i]);
            }
            if (consume_if_ineligible(dists[i], candidate_result[i].second)) {
                continue;
            }
            if (reorder_heap->Size() < topk || dists[i] < reorder_heap->Top().first) {
                reorder_heap->Push(dists[i], candidate_result[i].second);
                if (reorder_heap->Size() > topk) {
                    const auto curr = reorder_heap->Top();
                    add_iterator_discard(curr.first, curr.second);
                    if (ctx.reasoning_ctx != nullptr) {
                        ctx.reasoning_ctx->RecordReorderEviction(reorder_heap->Top().second, 0);
                    }
                    reorder_heap->Pop();
                }
            } else {
                add_iterator_discard(dists[i], candidate_result[i].second);
            }
        }
        return reorder_heap;
    }

    const uint64_t rabitq_candidate_size = rabitq_lower_bound_candidates->size();
    const uint64_t max_candidate_size = heap_candidate_size + rabitq_candidate_size;
    if (topk <= 0) {
        topk = static_cast<int64_t>(max_candidate_size);
    }
    if (topk == 0 or max_candidate_size == 0) {
        return std::make_shared<StandardHeap<true, false>>(query_allocator, 0);
    }

    Vector<InnerIdType> all_ids(max_candidate_size, query_allocator);
    Vector<float> lower_bound_probe_dists(max_candidate_size, query_allocator);
    Vector<float> lower_bounds(max_candidate_size, query_allocator);
    Vector<float> filter_inner_products(
        max_candidate_size, std::numeric_limits<float>::quiet_NaN(), query_allocator);
    Vector<float> full_distances(
        max_candidate_size, std::numeric_limits<float>::quiet_NaN(), query_allocator);
    UnorderedMap<InnerIdType, uint64_t> merged_ids(query_allocator);
    merged_ids.reserve(max_candidate_size);

    uint64_t candidate_size = 0;
    for (const auto& item : *rabitq_lower_bound_candidates) {
        const auto [iter, inserted] = merged_ids.emplace(item.id, candidate_size);
        if (inserted) {
            all_ids[candidate_size] = item.id;
            lower_bound_probe_dists[candidate_size] = item.lower_bound;
            lower_bounds[candidate_size] = item.lower_bound;
            filter_inner_products[candidate_size] = item.filter_inner_product;
            full_distances[candidate_size] = item.full_distance;
            ++candidate_size;
            continue;
        }

        const auto idx = iter->second;
        if (has_valid_distance(item.lower_bound) and
            (not has_valid_distance(lower_bounds[idx]) or item.lower_bound < lower_bounds[idx])) {
            lower_bound_probe_dists[idx] = item.lower_bound;
            lower_bounds[idx] = item.lower_bound;
        }
        if (not IsFiniteRaBitQValue(filter_inner_products[idx]) and
            IsFiniteRaBitQValue(item.filter_inner_product)) {
            filter_inner_products[idx] = item.filter_inner_product;
        }
        if (not has_valid_distance(full_distances[idx]) and
            has_valid_distance(item.full_distance)) {
            full_distances[idx] = item.full_distance;
        }
    }
    const uint64_t hinted_candidate_size = candidate_size;

    const auto* candidate_result = input == nullptr ? nullptr : input->GetData();
    for (uint64_t i = 0; i < heap_candidate_size; ++i) {
        const auto id = candidate_result[i].second;
        if (merged_ids.emplace(id, candidate_size).second) {
            all_ids[candidate_size++] = id;
        }
    }

    ComputerInterfacePtr computer{nullptr};
    const auto ensure_computer = [&]() -> const ComputerInterfacePtr& {
        if (computer == nullptr) {
            computer = split_codes->FactoryFusedComputer(query);
        }
        return computer;
    };

    const uint64_t unhinted_candidate_size = candidate_size - hinted_candidate_size;
    if (unhinted_candidate_size > 0) {
        add_reorder_lower_bound_probe_count(ctx, unhinted_candidate_size);
        ScopedDistancePhase scoped_phase(ctx, DistanceEvaluationPhase::RERANK);
        QueryFusedLowerBound(lower_bound_probe_dists.data() + hinted_candidate_size,
                             lower_bounds.data() + hinted_candidate_size,
                             filter_inner_products.data() + hinted_candidate_size,
                             split_codes,
                             ensure_computer(),
                             all_ids.data() + hinted_candidate_size,
                             unhinted_candidate_size,
                             &ctx);
    }

    topk = std::min(topk, static_cast<int64_t>(candidate_size));
    auto reorder_heap = std::make_shared<StandardHeap<true, false>>(query_allocator, topk);
    if (topk == 0 or candidate_size == 0) {
        return reorder_heap;
    }

    const auto push_full_distance = [&](uint64_t idx, float distance) {
        if (not has_valid_distance(distance)) {
            return;
        }
        if (ctx.reasoning_ctx != nullptr) {
            ctx.reasoning_ctx->RecordReorder(all_ids[idx], lower_bound_probe_dists[idx], distance);
        }
        if (consume_if_ineligible(distance, all_ids[idx])) {
            return;
        }
        if (reorder_heap->Size() < topk or distance < reorder_heap->Top().first) {
            reorder_heap->Push(distance, all_ids[idx]);
            if (reorder_heap->Size() > topk) {
                const auto curr = reorder_heap->Top();
                add_iterator_discard(curr.first, curr.second);
                if (ctx.reasoning_ctx != nullptr) {
                    ctx.reasoning_ctx->RecordReorderEviction(curr.second, 0);
                }
                reorder_heap->Pop();
            }
        } else {
            add_iterator_discard(distance, all_ids[idx]);
        }
    };

    bool all_full_distances_available = true;
    for (uint64_t i = 0; i < candidate_size; ++i) {
        if (not has_valid_distance(full_distances[i])) {
            all_full_distances_available = false;
            break;
        }
    }
    if (all_full_distances_available) {
        for (uint64_t i = 0; i < candidate_size; ++i) {
            push_full_distance(i, full_distances[i]);
        }
        return reorder_heap;
    }

    bool lower_bounds_available = true;
    for (uint64_t i = 0; i < candidate_size; ++i) {
        if (not has_valid_distance(lower_bounds[i])) {
            lower_bounds_available = false;
            break;
        }
    }

    constexpr uint64_t batch_size = 256;
    const uint64_t missing_buffer_size = lower_bounds_available ? batch_size : candidate_size;
    Vector<InnerIdType> missing_ids(missing_buffer_size, query_allocator);
    Vector<float> missing_hints(missing_buffer_size, query_allocator);
    Vector<float> missing_dists(missing_buffer_size, query_allocator);
    Vector<uint64_t> missing_positions(missing_buffer_size, query_allocator);

    const auto compute_missing_full_distances =
        [&](const uint64_t* indices, uint64_t count, float* output) {
            uint64_t missing_count = 0;
            for (uint64_t i = 0; i < count; ++i) {
                const auto idx = indices == nullptr ? i : indices[i];
                if (has_valid_distance(full_distances[idx])) {
                    output[i] = full_distances[idx];
                    continue;
                }
                missing_ids[missing_count] = all_ids[idx];
                missing_hints[missing_count] = filter_inner_products[idx];
                missing_positions[missing_count] = i;
                ++missing_count;
            }
            if (missing_count == 0) {
                return;
            }
            add_reorder_distance_count(ctx, missing_count);
            {
                ScopedDistancePhase scoped(ctx, DistanceEvaluationPhase::RERANK);
                QueryFusedFullWithHint(missing_dists.data(),
                                       missing_hints.data(),
                                       split_codes,
                                       ensure_computer(),
                                       missing_ids.data(),
                                       missing_count,
                                       &ctx);
            }
            for (uint64_t i = 0; i < missing_count; ++i) {
                const auto position = missing_positions[i];
                const auto idx = indices == nullptr ? position : indices[position];
                output[position] = missing_dists[i];
                full_distances[idx] = missing_dists[i];
            }
        };

    if (not lower_bounds_available) {
        compute_missing_full_distances(nullptr, candidate_size, full_distances.data());
        for (uint64_t i = 0; i < candidate_size; ++i) {
            push_full_distance(i, full_distances[i]);
        }
        return reorder_heap;
    }

    Vector<uint64_t> order(candidate_size, query_allocator);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&lower_bounds](uint64_t lhs, uint64_t rhs) {
        return lower_bounds[lhs] < lower_bounds[rhs];
    });

    Vector<float> batch_dists(batch_size, query_allocator);
    Vector<uint64_t> batch_indices(batch_size, query_allocator);

    // Seed the exact threshold with every full distance produced during traversal.  Missing
    // candidates then load their supplement only while their lower bound can still enter top-k.
    for (uint64_t i = 0; i < candidate_size; ++i) {
        if (has_valid_distance(full_distances[i])) {
            push_full_distance(i, full_distances[i]);
        }
    }

    uint64_t cursor = 0;
    while (cursor < candidate_size) {
        while (cursor < candidate_size and has_valid_distance(full_distances[order[cursor]])) {
            ++cursor;
        }
        if (cursor == candidate_size) {
            break;
        }
        const bool heap_full = reorder_heap->Size() == static_cast<uint64_t>(topk);
        const float threshold =
            heap_full ? reorder_heap->Top().first : std::numeric_limits<float>::max();
        if (heap_full and lower_bounds[order[cursor]] >= threshold) {
            break;
        }
        const uint64_t batch_limit =
            heap_full ? batch_size
                      : std::min<uint64_t>(batch_size,
                                           static_cast<uint64_t>(topk) - reorder_heap->Size());
        uint64_t batch_count = 0;
        while (cursor < candidate_size and batch_count < batch_limit) {
            const auto idx = order[cursor];
            if (has_valid_distance(full_distances[idx])) {
                ++cursor;
                continue;
            }
            if (heap_full and lower_bounds[idx] >= threshold) {
                break;
            }
            batch_indices[batch_count++] = idx;
            ++cursor;
        }

        if (batch_count == 0) {
            break;
        }

        compute_missing_full_distances(batch_indices.data(), batch_count, batch_dists.data());
        for (uint64_t i = 0; i < batch_count; ++i) {
            push_full_distance(batch_indices[i], batch_dists[i]);
        }
    }
    if (iter_ctx != nullptr) {
        // Iterator pages can later expose discard values directly (including on the final drain),
        // so every buffered candidate must carry a full distance rather than a lower bound.
        while (cursor < candidate_size) {
            uint64_t batch_count = 0;
            while (cursor < candidate_size and batch_count < batch_size) {
                const auto idx = order[cursor++];
                if (not has_valid_distance(full_distances[idx])) {
                    batch_indices[batch_count++] = idx;
                }
            }
            if (batch_count == 0) {
                continue;
            }
            compute_missing_full_distances(batch_indices.data(), batch_count, batch_dists.data());
            for (uint64_t i = 0; i < batch_count; ++i) {
                const auto idx = batch_indices[i];
                if (ctx.reasoning_ctx != nullptr) {
                    ctx.reasoning_ctx->RecordReorder(
                        all_ids[idx], lower_bound_probe_dists[idx], batch_dists[i]);
                }
                add_iterator_discard(batch_dists[i], all_ids[idx]);
            }
        }
    }
    return reorder_heap;
}
}  // namespace vsag
