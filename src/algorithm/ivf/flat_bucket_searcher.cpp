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
                           Vector<InnerIdType>& scanned_inner_ids,
                           ReasoningContext* reasoning_ctx) const {
    auto bucket_size = bucket->GetBucketScanCapacity(bucket_id);
    if (static_cast<uint64_t>(bucket_size) > dist.size()) {
        dist.resize(bucket_size);
    }

    const InnerIdType scan_capacity = bucket_size;
    scanned_inner_ids.resize(scan_capacity);
    InnerIdType scanned_size = 0;
    bucket->ScanBucketById(dist.data(),
                           computer,
                           bucket_id,
                           param.query_context,
                           scanned_inner_ids.data(),
                           scan_capacity,
                           &scanned_size);
    bucket_size = scanned_size;
    const auto* ids = scanned_inner_ids.data();
    if (param.query_context != nullptr and param.query_context->stats != nullptr &&
        bucket_size > 0) {
        param.query_context->stats->AddDistance(SearchStatistics::DistancePhase::APPROXIMATE,
                                                bucket->backend_,
                                                static_cast<uint64_t>(bucket_size));
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
