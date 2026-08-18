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

#include "flat_bucket_searcher.h"

#include <cmath>
#include <limits>

#include "attr/executor/executor.h"
#include "impl/reasoning/search_reasoning.h"
#include "impl/searcher/basic_searcher.h"
#include "query_context.h"

namespace vsag {

void
FlatBucketSearcher::Search(BucketIdType bucket_id,
                           const BucketInterfacePtr& bucket,
                           const ComputerInterfacePtr& computer,
                           const InnerSearchParam& param,
                           int64_t thread_id,
                           int64_t topk,
                           BucketIdType buckets_per_data,
                           DistHeapPtr& heap,
                           Vector<float>& dist,
                           ReasoningContext* reasoning_ctx) const {
    auto bucket_size = bucket->GetBucketSize(bucket_id);
    const auto* ids = bucket->GetInnerIds(bucket_id);
    const uint64_t scratch_multiplier = param.use_rabitq_heap_search ? 3 : 1;
    const uint64_t scratch_size = static_cast<uint64_t>(bucket_size) * scratch_multiplier;
    if (scratch_size > dist.size()) {
        dist.resize(scratch_size);
    }

    float* lower_bounds = nullptr;
    float* filter_inner_products = nullptr;
    if (param.use_rabitq_heap_search) {
        lower_bounds = dist.data() + bucket_size;
        filter_inner_products = lower_bounds + bucket_size;
        bucket->ScanBucketWithDistanceLowerBound(dist.data(),
                                                 lower_bounds,
                                                 filter_inner_products,
                                                 computer,
                                                 bucket_id,
                                                 param.query_context);
    } else {
        bucket->ScanBucketById(dist.data(), computer, bucket_id, param.query_context);
        if (param.query_context != nullptr and param.query_context->stats != nullptr and
            bucket_size > 0 and not bucket->SupportSplitCodeStorage()) {
            param.query_context->stats->AddDistance(SearchStatistics::DistancePhase::APPROXIMATE,
                                                    bucket->backend_,
                                                    static_cast<uint64_t>(bucket_size));
        }
    }

    Filter* attr_ft = nullptr;
    size_t tid = 0;
    if (thread_id >= 0 and param.executors.size() > static_cast<uint64_t>(thread_id)) {
        tid = static_cast<size_t>(thread_id);
        if (param.executors[tid] != nullptr) {
            param.executors[tid]->Clear();
            attr_ft = param.executors[tid]->Run(bucket_id);
        }
    }

    const auto& ft = param.is_inner_id_allowed;
    const auto topk_u = static_cast<uint64_t>(topk);
    auto cur_heap_top = std::numeric_limits<float>::max();
    if (not heap->Empty() and heap->Size() == topk_u) {
        cur_heap_top = heap->Top().first;
    }

    if (param.use_rabitq_heap_search) {
        CHECK_ARGUMENT(param.search_mode == KNN_SEARCH,
                       "RaBitQ heap search only supports KNN search");
        for (int64_t j = 0; j < bucket_size; ++j) {
            if (ids[j] == std::numeric_limits<InnerIdType>::max()) {
                continue;
            }
            const auto origin_id = ids[j] / buckets_per_data;
            if (reasoning_ctx != nullptr) {
                reasoning_ctx->RecordVisit(origin_id, dist[j], 0);
            }
            if (attr_ft != nullptr and not attr_ft->CheckValid(j)) {
                if (reasoning_ctx != nullptr) {
                    reasoning_ctx->RecordFilterReject(origin_id);
                }
                continue;
            }
            if (ft != nullptr and not ft->CheckValid(origin_id)) {
                if (reasoning_ctx != nullptr) {
                    reasoning_ctx->RecordFilterReject(origin_id);
                }
                continue;
            }

            float distance_limit = cur_heap_top;
            if (param.distance_threshold.has_value()) {
                distance_limit = std::min(distance_limit, param.distance_threshold.value());
            }
            const float lower_bound = lower_bounds[j];
            const bool has_usable_lower_bound =
                std::isfinite(lower_bound) and lower_bound < std::numeric_limits<float>::max();
            if (has_usable_lower_bound and not(lower_bound < distance_limit)) {
                continue;
            }

            float exact_distance = std::numeric_limits<float>::max();
            if (param.query_context != nullptr and param.query_context->stats != nullptr) {
                param.query_context->stats->reorder_distance_count.fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (param.query_context != nullptr) {
                QueryContext exact_context = *param.query_context;
                ScopedDistancePhase scoped(exact_context, DistanceEvaluationPhase::RERANK);
                bucket->QueryWithFilterInnerProductByInnerId(&exact_distance,
                                                             filter_inner_products + j,
                                                             computer,
                                                             ids + j,
                                                             1,
                                                             &exact_context);
            } else {
                bucket->QueryWithFilterInnerProductByInnerId(
                    &exact_distance, filter_inner_products + j, computer, ids + j, 1);
            }
            if (reasoning_ctx != nullptr) {
                reasoning_ctx->RecordReorder(origin_id, dist[j], exact_distance);
            }
            if (not std::isfinite(exact_distance) or
                (param.distance_threshold.has_value() and
                 exact_distance > param.distance_threshold.value())) {
                continue;
            }
            if (heap->Size() < topk_u or exact_distance < cur_heap_top) {
                heap->Push(exact_distance, ids[j]);
            }
            while (heap->Size() > topk_u) {
                if (reasoning_ctx != nullptr) {
                    reasoning_ctx->RecordEviction(heap->Top().second / buckets_per_data, 0);
                }
                heap->Pop();
            }
            if (not heap->Empty() and heap->Size() == topk_u) {
                cur_heap_top = heap->Top().first;
            }
        }
        return;
    }

    if (param.search_mode == KNN_SEARCH) {
        for (int64_t j = 0; j < bucket_size; ++j) {
            if (ids[j] == std::numeric_limits<InnerIdType>::max()) {
                continue;
            }
            auto origin_id = ids[j] / buckets_per_data;
            if (reasoning_ctx != nullptr) {
                reasoning_ctx->RecordVisit(origin_id, dist[j], 0);
            }
            if (param.distance_threshold.has_value() and
                (not std::isfinite(dist[j]) or
                 (not param.enable_reorder and dist[j] > param.distance_threshold.value()))) {
                continue;
            }
            if (attr_ft != nullptr and not attr_ft->CheckValid(j)) {
                if (reasoning_ctx != nullptr) {
                    reasoning_ctx->RecordFilterReject(origin_id);
                }
                continue;
            }
            if (ft == nullptr or ft->CheckValid(origin_id)) {
                if (heap->Size() < topk_u or dist[j] < cur_heap_top) {
                    heap->Push(dist[j], ids[j]);
                }
                while (heap->Size() > topk_u) {
                    if (reasoning_ctx != nullptr) {
                        reasoning_ctx->RecordEviction(heap->Top().second / buckets_per_data, 0);
                    }
                    heap->Pop();
                }
                if (not heap->Empty() and heap->Size() == topk_u) {
                    cur_heap_top = heap->Top().first;
                }
            } else if (reasoning_ctx != nullptr) {
                reasoning_ctx->RecordFilterReject(origin_id);
            }
        }
    } else {  // RANGE_SEARCH
        for (int64_t j = 0; j < bucket_size; ++j) {
            if (ids[j] == std::numeric_limits<InnerIdType>::max()) {
                continue;
            }
            auto origin_id = ids[j] / buckets_per_data;
            if (reasoning_ctx != nullptr) {
                reasoning_ctx->RecordVisit(origin_id, dist[j], 0);
            }
            if (attr_ft != nullptr and not attr_ft->CheckValid(j)) {
                if (reasoning_ctx != nullptr) {
                    reasoning_ctx->RecordFilterReject(origin_id);
                }
                continue;
            }
            if (ft == nullptr or ft->CheckValid(origin_id)) {
                if (dist[j] <= param.radius + THRESHOLD_ERROR and dist[j] < cur_heap_top) {
                    heap->Push(dist[j], ids[j]);
                }
                while (heap->Size() > topk_u) {
                    if (reasoning_ctx != nullptr) {
                        reasoning_ctx->RecordEviction(heap->Top().second / buckets_per_data, 0);
                    }
                    heap->Pop();
                }
                if (not heap->Empty() and heap->Size() == topk_u) {
                    cur_heap_top = heap->Top().first;
                }
            } else if (reasoning_ctx != nullptr) {
                reasoning_ctx->RecordFilterReject(origin_id);
            }
        }
    }
}

}  // namespace vsag
