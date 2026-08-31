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

#include "rabitq_split_bucket_searcher.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "attr/executor/executor.h"
#include "impl/reasoning/search_reasoning.h"
#include "impl/searcher/basic_searcher.h"
#include "query_context.h"
#include "simd/pqfs_simd.h"

namespace vsag {

namespace {

bool
IsFiniteFloatBits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7F800000U) != 0x7F800000U;
}

}  // namespace

void
RaBitQSplitBucketSearcher::Search(BucketIdType bucket_id,
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
    const InnerIdType* ids = nullptr;
    const bool collect_candidate_filter_inner_products =
        param.search_mode == KNN_SEARCH and param.enable_reorder and
        bucket->SupportSplitCodeStorage() and heap->StoresAuxiliary();

    const uint64_t scratch_multiplier = collect_candidate_filter_inner_products ? 2 : 1;
    const uint64_t scratch_size = static_cast<uint64_t>(bucket_size) * scratch_multiplier;
    if (scratch_size > dist.size()) {
        dist.resize(scratch_size);
    }
    if (collect_candidate_filter_inner_products and dist.empty()) {
        // Keep valid output pointers when a bucket was empty at the capacity snapshot. A concurrent
        // first append remains outside the bounded scan instead of writing past the scratch space.
        dist.resize(1);
    }

    float* filter_inner_products = nullptr;
    uint64_t candidate_source_version = std::numeric_limits<uint64_t>::max();
    if (collect_candidate_filter_inner_products) {
        const InnerIdType scan_capacity = bucket_size;
        filter_inner_products = dist.data() + scan_capacity;
        scanned_inner_ids.resize(scan_capacity);
        InnerIdType scanned_size = 0;
        candidate_source_version =
            bucket->ScanBucketWithFilterInnerProduct(dist.data(),
                                                     filter_inner_products,
                                                     computer,
                                                     bucket_id,
                                                     param.query_context,
                                                     scanned_inner_ids.data(),
                                                     scan_capacity,
                                                     &scanned_size);
        bucket_size = scanned_size;
        ids = scanned_inner_ids.data();
    } else {
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
        ids = scanned_inner_ids.data();
    }
    if (param.query_context != nullptr and param.query_context->stats != nullptr and
        bucket_size > 0 and not bucket->SupportSplitCodeStorage()) {
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
        auto push_to_heap = [&](float distance, InnerIdType id, int64_t offset) {
            if (collect_candidate_filter_inner_products) {
                heap->PushWithAuxiliary(distance,
                                        id,
                                        filter_inner_products[offset],
                                        bucket_id,
                                        static_cast<InnerIdType>(offset),
                                        candidate_source_version);
            } else {
                heap->Push(distance, id);
            }
        };
        if (attr_ft == nullptr and ft == nullptr and reasoning_ctx == nullptr) {
            auto push_candidate = [&](int64_t offset) {
                if (ids[offset] == std::numeric_limits<InnerIdType>::max()) {
                    return;
                }
                if (param.distance_threshold.has_value() and
                    (not IsFiniteFloatBits(dist[offset]) or
                     (not param.enable_reorder and
                      dist[offset] > param.distance_threshold.value()))) {
                    return;
                }
                if (heap->Size() < topk_u or dist[offset] < cur_heap_top) {
                    push_to_heap(dist[offset], ids[offset], offset);
                }
                while (heap->Size() > topk_u) {
                    heap->Pop();
                }
                if (not heap->Empty() and heap->Size() == topk_u) {
                    cur_heap_top = heap->Top().first;
                }
            };

            int64_t offset = 0;
            for (; offset + 32 <= bucket_size; offset += 32) {
                uint32_t mask = std::numeric_limits<uint32_t>::max();
                if (heap->Size() >= topk_u) {
                    mask = FP32LessThan32Mask(dist.data() + offset, cur_heap_top);
                }
                while (mask != 0U) {
                    const auto lane = static_cast<int64_t>(__builtin_ctz(mask));
                    const int64_t candidate_offset = offset + lane;
                    if (heap->Size() < topk_u or dist[candidate_offset] < cur_heap_top) {
                        push_candidate(candidate_offset);
                    }
                    mask &= mask - 1U;
                }
            }
            for (; offset < bucket_size; ++offset) {
                push_candidate(offset);
            }
            return;
        }

        for (int64_t j = 0; j < bucket_size; ++j) {
            if (ids[j] == std::numeric_limits<InnerIdType>::max()) {
                continue;
            }
            auto origin_id = ids[j] / buckets_per_data;
            if (reasoning_ctx != nullptr) {
                reasoning_ctx->RecordVisit(origin_id, dist[j], 0);
            }
            if (param.distance_threshold.has_value() and
                (not IsFiniteFloatBits(dist[j]) or
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
                    push_to_heap(dist[j], ids[j], j);
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
