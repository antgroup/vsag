
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
#include "impl/heap/standard_heap.h"
#include "impl/reasoning/search_reasoning.h"
#include "query_context.h"

namespace vsag {

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
                        const DistanceRecordVector* rabitq_lower_bound_candidates) {
    // set query allocator
    Allocator* query_allocator = select_query_allocator(ctx.alloc, allocator_);
    const uint64_t heap_candidate_size = input == nullptr ? 0 : input->Size();
    if (rabitq_lower_bound_candidates == nullptr) {
        topk = std::min(topk, static_cast<int64_t>(heap_candidate_size));
        auto reorder_heap = std::make_shared<StandardHeap<true, false>>(query_allocator, topk);
        auto computer = flatten_->FactoryComputer(query);
        Vector<InnerIdType> ids(heap_candidate_size, query_allocator);
        Vector<float> dists(heap_candidate_size, query_allocator);
        const auto* candidate_result = input == nullptr ? nullptr : input->GetData();
        for (uint64_t i = 0; i < heap_candidate_size; ++i) {
            ids[i] = candidate_result[i].second;
        }
        add_reorder_distance_count(ctx, heap_candidate_size);
        flatten_->Query(dists.data(), computer, ids.data(), heap_candidate_size, &ctx);
        for (uint64_t i = 0; i < heap_candidate_size; ++i) {
            if (ctx.reasoning_ctx != nullptr) {
                ctx.reasoning_ctx->RecordReorder(
                    candidate_result[i].second, candidate_result[i].first, dists[i]);
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
    auto computer = flatten_->FactoryComputer(query);
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
        return std::isfinite(lower_bound) and lower_bound < std::numeric_limits<float>::max();
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
    flatten_->QueryWithDistanceHint(
        dists.data(), hint_dists.data(), computer, ids.data(), bootstrap_size, &ctx);
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
        flatten_->QueryWithDistanceHint(
            dists.data(), hint_dists.data(), computer, ids.data(), batch_count, &ctx);
        for (uint64_t i = 0; i < batch_count; ++i) {
            if (ctx.reasoning_ctx != nullptr) {
                ctx.reasoning_ctx->RecordReorder(
                    ids[i], lower_bound_probe_dists[batch_indices[i]], dists[i]);
            }
            if (dists[i] < reorder_heap->Top().first) {
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
}  // namespace vsag
