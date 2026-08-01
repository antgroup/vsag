
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
#include "impl/reasoning/search_reasoning.h"
#include "query_context.h"

namespace vsag {

void
FlattenReorder::QueryLowerBound(float* distances,
                                float* lower_bounds,
                                float* filter_inner_products,
                                const ComputerInterfacePtr& computer,
                                const InnerIdType* ids,
                                uint64_t count,
                                QueryContext* ctx) const {
    auto* split_codes = dynamic_cast<RaBitQSplitDataCellInterface*>(flatten_.get());
    if (fused_graph_ == nullptr or split_codes == nullptr) {
        flatten_->QueryWithDistanceLowerBoundAndFilterIP(
            distances, lower_bounds, filter_inner_products, computer, ids, count, ctx);
        return;
    }
    uint32_t fallback_count = 0;
    QueryContext rate_context;
    QueryContext* rate_context_ptr = nullptr;
    if (ctx != nullptr) {
        rate_context.rabitq_error_rate = ctx->rabitq_error_rate;
        rate_context_ptr = &rate_context;
    }
    for (uint64_t i = 0; i < count; ++i) {
        const auto node = fused_graph_->GetNodeView(ids[i]);
        if (not split_codes->ComputeFusedOneBitWithFilterIP(computer,
                                                            node.cluster_id,
                                                            node.one_bit_code,
                                                            node.supplement_code,
                                                            distances + i,
                                                            lower_bounds + i,
                                                            filter_inner_products + i,
                                                            rate_context_ptr)) {
            ++fallback_count;
        }
    }
    if (ctx != nullptr and ctx->stats != nullptr) {
        ctx->stats->rabitq_filter_count.fetch_add(static_cast<uint32_t>(count),
                                                  std::memory_order_relaxed);
        ctx->stats->rabitq_filter_fallback_full_count.fetch_add(fallback_count,
                                                                std::memory_order_relaxed);
    }
}

void
FlattenReorder::QueryFullWithHint(float* distances,
                                  const float* filter_inner_products,
                                  const ComputerInterfacePtr& computer,
                                  const InnerIdType* ids,
                                  uint64_t count,
                                  QueryContext* ctx) const {
    auto* split_codes = dynamic_cast<RaBitQSplitDataCellInterface*>(flatten_.get());
    if (fused_graph_ == nullptr or split_codes == nullptr) {
        flatten_->QueryWithFilterIPHint(
            distances, filter_inner_products, computer, ids, count, ctx);
        return;
    }
    uint32_t hint_full_count = 0;
    uint32_t fallback_full_count = 0;
    uint32_t full_count = 0;
    const bool exact_filter_ip_hint =
        split_codes->FusedFilterBits() >= 2 and not split_codes->UsesLegacyHnswFusedCodec();
    for (uint64_t i = 0; i < count; ++i) {
        const auto* record = fused_graph_->GetNodeRecord(ids[i]);
        const auto cluster_id = fused_graph_->GetClusterId(record);
        bool used_hint = false;
        if (exact_filter_ip_hint and IsFiniteRaBitQValue(filter_inner_products[i])) {
            ++full_count;
            used_hint =
                split_codes->ComputeFusedFullWithFilterIP(computer,
                                                          cluster_id,
                                                          fused_graph_->GetOneBitCode(record),
                                                          fused_graph_->GetSupplementCode(record),
                                                          filter_inner_products[i],
                                                          distances + i,
                                                          nullptr);
        }
        if (used_hint) {
            ++hint_full_count;
        } else {
            ++fallback_full_count;
            ++full_count;
            split_codes->ComputeFusedFull(computer,
                                          cluster_id,
                                          fused_graph_->GetOneBitCode(record),
                                          fused_graph_->GetSupplementCode(record),
                                          distances + i,
                                          nullptr);
        }
    }
    if (ctx != nullptr and ctx->stats != nullptr) {
        ctx->stats->rabitq_full_count.fetch_add(full_count, std::memory_order_relaxed);
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
                        const RaBitQCandidateVector* rabitq_lower_bound_candidates) {
    // set query allocator
    Allocator* query_allocator = select_query_allocator(ctx.alloc, allocator_);
    const uint64_t heap_candidate_size = input == nullptr ? 0 : input->Size();
    const auto add_iterator_discard = [iter_ctx](float distance, InnerIdType id) {
        if (iter_ctx != nullptr) {
            iter_ctx->AddDiscardNode(distance, id);
        }
    };
    if (rabitq_lower_bound_candidates == nullptr) {
        topk = std::min(topk, static_cast<int64_t>(heap_candidate_size));
        auto reorder_heap = std::make_shared<StandardHeap<true, false>>(query_allocator, topk);
        auto* split_codes = dynamic_cast<RaBitQSplitDataCellInterface*>(flatten_.get());
        auto computer = fused_graph_ != nullptr and split_codes != nullptr
                            ? split_codes->FactoryFusedComputer(query)
                            : flatten_->FactoryComputer(query);
        Vector<InnerIdType> ids(heap_candidate_size, query_allocator);
        Vector<float> dists(heap_candidate_size, query_allocator);
        const auto* candidate_result = input == nullptr ? nullptr : input->GetData();
        for (uint64_t i = 0; i < heap_candidate_size; ++i) {
            ids[i] = candidate_result[i].second;
        }
        add_reorder_distance_count(ctx, heap_candidate_size);
        if (fused_graph_ != nullptr and split_codes != nullptr) {
            for (uint64_t i = 0; i < heap_candidate_size; ++i) {
                const auto* record = fused_graph_->GetNodeRecord(ids[i]);
                split_codes->ComputeFusedFull(computer,
                                              fused_graph_->GetClusterId(record),
                                              fused_graph_->GetOneBitCode(record),
                                              fused_graph_->GetSupplementCode(record),
                                              dists.data() + i,
                                              &ctx);
            }
        } else {
            flatten_->Query(dists.data(), computer, ids.data(), heap_candidate_size, &ctx);
        }
        for (uint64_t i = 0; i < heap_candidate_size; ++i) {
            if (ctx.reasoning_ctx != nullptr) {
                ctx.reasoning_ctx->RecordReorder(
                    candidate_result[i].second, candidate_result[i].first, dists[i]);
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

    const uint64_t rabitq_candidate_size =
        rabitq_lower_bound_candidates == nullptr ? 0 : rabitq_lower_bound_candidates->size();
    const uint64_t max_candidate_size = heap_candidate_size + rabitq_candidate_size;
    if (topk <= 0) {
        topk = static_cast<int64_t>(max_candidate_size);
    }
    auto* split_codes = dynamic_cast<RaBitQSplitDataCellInterface*>(flatten_.get());
    auto computer = fused_graph_ != nullptr and split_codes != nullptr
                        ? split_codes->FactoryFusedComputer(query)
                        : flatten_->FactoryComputer(query);
    if (topk == 0 || max_candidate_size == 0) {
        return std::make_shared<StandardHeap<true, false>>(query_allocator, 0);
    }

    Vector<InnerIdType> all_ids(max_candidate_size, query_allocator);
    Vector<float> lower_bound_probe_dists(max_candidate_size, query_allocator);
    Vector<float> lower_bounds(max_candidate_size, query_allocator);
    Vector<float> filter_inner_products(
        max_candidate_size, std::numeric_limits<float>::quiet_NaN(), query_allocator);
    UnorderedSet<InnerIdType> merged_ids(query_allocator);
    merged_ids.reserve(max_candidate_size);

    uint64_t candidate_size = 0;
    if (rabitq_lower_bound_candidates != nullptr) {
        for (const auto& item : *rabitq_lower_bound_candidates) {
            if (merged_ids.insert(item.id).second) {
                all_ids[candidate_size] = item.id;
                lower_bound_probe_dists[candidate_size] = item.lower_bound;
                lower_bounds[candidate_size] = item.lower_bound;
                filter_inner_products[candidate_size] = item.filter_inner_product;
                ++candidate_size;
            }
        }
    }
    const uint64_t hinted_candidate_size = candidate_size;

    const auto* candidate_result = input == nullptr ? nullptr : input->GetData();
    for (uint64_t i = 0; i < heap_candidate_size; ++i) {
        const auto id = candidate_result[i].second;
        if (merged_ids.insert(id).second) {
            all_ids[candidate_size++] = id;
        }
    }
    const uint64_t unhinted_candidate_size = candidate_size - hinted_candidate_size;
    if (unhinted_candidate_size > 0) {
        add_reorder_lower_bound_probe_count(ctx, unhinted_candidate_size);
        QueryLowerBound(lower_bound_probe_dists.data() + hinted_candidate_size,
                        lower_bounds.data() + hinted_candidate_size,
                        filter_inner_products.data() + hinted_candidate_size,
                        computer,
                        all_ids.data() + hinted_candidate_size,
                        unhinted_candidate_size,
                        &ctx);
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
        flatten_->Query(
            lower_bound_probe_dists.data(), computer, all_ids.data(), candidate_size, &ctx);
        for (uint64_t i = 0; i < candidate_size; ++i) {
            if (ctx.reasoning_ctx != nullptr) {
                ctx.reasoning_ctx->RecordReorder(
                    all_ids[i], lower_bounds[i], lower_bound_probe_dists[i]);
            }
            if (reorder_heap->Size() < topk or
                lower_bound_probe_dists[i] < reorder_heap->Top().first) {
                reorder_heap->Push(lower_bound_probe_dists[i], all_ids[i]);
                if (reorder_heap->Size() > topk) {
                    const auto curr = reorder_heap->Top();
                    add_iterator_discard(curr.first, curr.second);
                    if (ctx.reasoning_ctx != nullptr) {
                        ctx.reasoning_ctx->RecordReorderEviction(reorder_heap->Top().second, 0);
                    }
                    reorder_heap->Pop();
                }
            } else {
                add_iterator_discard(lower_bound_probe_dists[i], all_ids[i]);
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
    Vector<float> filter_ip_hints(buffer_size, query_allocator);
    Vector<uint64_t> batch_indices(buffer_size, query_allocator);

    for (uint64_t i = 0; i < bootstrap_size; ++i) {
        const auto idx = order[i];
        ids[i] = all_ids[idx];
        filter_ip_hints[i] = filter_inner_products[idx];
    }
    add_reorder_distance_count(ctx, bootstrap_size);
    QueryFullWithHint(
        dists.data(), filter_ip_hints.data(), computer, ids.data(), bootstrap_size, &ctx);
    for (uint64_t i = 0; i < bootstrap_size; ++i) {
        if (ctx.reasoning_ctx != nullptr) {
            const auto idx = order[i];
            ctx.reasoning_ctx->RecordReorder(ids[i], lower_bound_probe_dists[idx], dists[i]);
        }
        reorder_heap->Push(dists[i], ids[i]);
    }

    uint64_t cursor = bootstrap_size;
    while (cursor < candidate_size) {
        if (reorder_heap->Size() == topk &&
            lower_bounds[order[cursor]] >= reorder_heap->Top().first) {
            break;
        }

        const auto threshold = reorder_heap->Top().first;
        uint64_t batch_count = 0;
        while (cursor < candidate_size && batch_count < batch_size) {
            const auto idx = order[cursor];
            if (lower_bounds[idx] >= threshold) {
                break;
            }
            ids[batch_count] = all_ids[idx];
            filter_ip_hints[batch_count] = filter_inner_products[idx];
            batch_indices[batch_count] = idx;
            ++batch_count;
            ++cursor;
        }

        if (batch_count == 0) {
            break;
        }

        add_reorder_distance_count(ctx, batch_count);
        QueryFullWithHint(
            dists.data(), filter_ip_hints.data(), computer, ids.data(), batch_count, &ctx);
        for (uint64_t i = 0; i < batch_count; ++i) {
            if (ctx.reasoning_ctx != nullptr) {
                ctx.reasoning_ctx->RecordReorder(
                    ids[i], lower_bound_probe_dists[batch_indices[i]], dists[i]);
            }
            if (dists[i] < reorder_heap->Top().first) {
                reorder_heap->Push(dists[i], ids[i]);
                if (reorder_heap->Size() > topk) {
                    const auto curr = reorder_heap->Top();
                    add_iterator_discard(curr.first, curr.second);
                    if (ctx.reasoning_ctx != nullptr) {
                        ctx.reasoning_ctx->RecordReorderEviction(reorder_heap->Top().second, 0);
                    }
                    reorder_heap->Pop();
                }
            } else {
                add_iterator_discard(dists[i], ids[i]);
            }
        }
    }
    if (iter_ctx != nullptr) {
        // Iterator pages can later expose discard values directly (including on the final drain),
        // so every buffered candidate must carry a full distance rather than a lower bound.
        while (cursor < candidate_size) {
            uint64_t batch_count = 0;
            while (cursor < candidate_size and batch_count < batch_size) {
                const auto idx = order[cursor++];
                ids[batch_count] = all_ids[idx];
                filter_ip_hints[batch_count] = filter_inner_products[idx];
                batch_indices[batch_count] = idx;
                ++batch_count;
            }
            add_reorder_distance_count(ctx, batch_count);
            QueryFullWithHint(
                dists.data(), filter_ip_hints.data(), computer, ids.data(), batch_count, &ctx);
            for (uint64_t i = 0; i < batch_count; ++i) {
                if (ctx.reasoning_ctx != nullptr) {
                    ctx.reasoning_ctx->RecordReorder(
                        ids[i], lower_bound_probe_dists[batch_indices[i]], dists[i]);
                }
                add_iterator_discard(dists[i], ids[i]);
            }
        }
    }
    return reorder_heap;
}
}  // namespace vsag
