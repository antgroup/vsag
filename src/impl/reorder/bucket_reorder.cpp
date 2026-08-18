// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "bucket_reorder.h"

#include <algorithm>
#include <atomic>
#include <cmath>

#include "impl/heap/standard_heap.h"
#include "impl/reasoning/search_reasoning.h"
#include "query_context.h"
#include "vsag_exception.h"

namespace vsag {

DistHeapPtr
BucketReorder::Reorder(const DistHeapPtr& input,
                       const void* query,
                       int64_t topk,
                       QueryContext& ctx,
                       IteratorFilterContext* iter_ctx,
                       const DistanceRecordVector* rabitq_lower_bound_candidates,
                       const std::optional<float>& distance_threshold) {
    if (iter_ctx != nullptr) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "BucketReorder does not support iterator filtering");
    }
    if (rabitq_lower_bound_candidates != nullptr) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "BucketReorder does not support RaBitQ lower-bound candidates");
    }

    Allocator* query_allocator = select_query_allocator(ctx.alloc, allocator_);
    const uint64_t candidate_count = input == nullptr ? 0 : input->Size();
    topk = std::min(topk, static_cast<int64_t>(candidate_count));
    auto reorder_heap = std::make_shared<StandardHeap<true, false>>(query_allocator, topk);
    if (candidate_count == 0 or topk == 0) {
        return reorder_heap;
    }

    if (ctx.stats != nullptr) {
        ctx.stats->reorder_distance_count.fetch_add(static_cast<uint32_t>(candidate_count),
                                                    std::memory_order_relaxed);
    }

    auto computer = bucket_->FactoryComputer(query);
    const auto* candidates = input->GetData();
    Vector<BucketIdType> bucket_ids(candidate_count, query_allocator);
    Vector<InnerIdType> offset_ids(candidate_count, query_allocator);
    Vector<float> precise_distances(candidate_count, query_allocator);
    for (uint64_t i = 0; i < candidate_count; ++i) {
        const auto [bucket_id, offset_id] = location_resolver_(candidates[i].second);
        bucket_ids[i] = bucket_id;
        offset_ids[i] = offset_id;
    }
    bucket_->Query(precise_distances.data(),
                   computer,
                   bucket_ids.data(),
                   offset_ids.data(),
                   static_cast<InnerIdType>(candidate_count),
                   &ctx);
    for (uint64_t i = 0; i < candidate_count; ++i) {
        const auto [coarse_distance, inner_id] = candidates[i];
        const auto precise_distance = precise_distances[i];
        if (ctx.reasoning_ctx != nullptr) {
            ctx.reasoning_ctx->RecordReorder(inner_id, coarse_distance, precise_distance);
        }
        if (distance_threshold.has_value() and (not std::isfinite(precise_distance) or
                                                precise_distance > distance_threshold.value())) {
            continue;
        }
        if (reorder_heap->Size() < topk or precise_distance < reorder_heap->Top().first) {
            reorder_heap->Push(precise_distance, inner_id);
            if (reorder_heap->Size() > topk) {
                if (ctx.reasoning_ctx != nullptr) {
                    ctx.reasoning_ctx->RecordReorderEviction(reorder_heap->Top().second, 0);
                }
                reorder_heap->Pop();
            }
        }
    }
    return reorder_heap;
}

}  // namespace vsag
