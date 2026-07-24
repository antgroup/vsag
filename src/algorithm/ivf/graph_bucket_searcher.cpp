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

#include "graph_bucket_searcher.h"

#include <limits>

#include "attr/executor/executor.h"
#include "impl/reasoning/search_reasoning.h"
#include "impl/searcher/basic_searcher.h"

namespace vsag {

GraphBucketSearcher::GraphBucketSearcher(int64_t graph_build_threshold,
                                         const Vector<GraphInterfacePtr>& bucket_graphs,
                                         Allocator* allocator)
    : graph_build_threshold_(graph_build_threshold),
      bucket_graphs_(bucket_graphs),
      allocator_(allocator),
      flat_searcher_(std::make_shared<FlatBucketSearcher>()) {
}

void
GraphBucketSearcher::Search(BucketIdType bucket_id,
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
    bool has_graph = (bucket_id < static_cast<BucketIdType>(bucket_graphs_.size()) &&
                      bucket_graphs_[bucket_id] != nullptr);

    bool graph_fresh = has_graph && bucket_graphs_[bucket_id]->TotalCount() ==
                                        static_cast<InnerIdType>(bucket_size);
    if (graph_fresh && bucket_size >= graph_build_threshold_) {
        search_graph(bucket_id,
                     bucket,
                     computer,
                     param,
                     thread_id,
                     topk,
                     buckets_per_data,
                     heap,
                     reasoning_ctx);
    } else {
        flat_searcher_->Search(bucket_id,
                               bucket,
                               computer,
                               param,
                               thread_id,
                               topk,
                               buckets_per_data,
                               heap,
                               dist,
                               reasoning_ctx);
    }
}

void
GraphBucketSearcher::search_graph(BucketIdType bucket_id,
                                  const BucketInterfacePtr& bucket,
                                  const ComputerInterfacePtr& computer,
                                  const InnerSearchParam& param,
                                  int64_t thread_id,
                                  int64_t topk,
                                  BucketIdType buckets_per_data,
                                  DistHeapPtr& heap,
                                  ReasoningContext* reasoning_ctx) const {
    auto bucket_size = bucket->GetBucketSize(bucket_id);
    if (bucket_size == 0) {
        return;
    }
    const auto& graph = bucket_graphs_[bucket_id];
    const auto* ids = bucket->GetInnerIds(bucket_id);

    uint64_t ef = param.ef;
    if (param.search_mode == InnerSearchMode::KNN_SEARCH && ef < static_cast<uint64_t>(topk)) {
        ef = static_cast<uint64_t>(topk);
    }
    // Cap ef at bucket_size to avoid full-graph traversal for unlimited RANGE_SEARCH
    // where the caller sets topk to total index size.
    if (ef > static_cast<uint64_t>(bucket_size)) {
        ef = static_cast<uint64_t>(bucket_size);
    }

    VisitedList vl(bucket_size, allocator_);

    StandardHeap<true, false> top_candidates(allocator_, -1);
    StandardHeap<true, false> candidate_set(allocator_, -1);

    const auto& ft = param.is_inner_id_allowed;
    const auto topk_u = static_cast<uint64_t>(topk);

    Filter* attr_ft = nullptr;
    if (not param.executors.empty() and
        static_cast<uint64_t>(thread_id) < param.executors.size() and
        param.executors[thread_id] != nullptr) {
        param.executors[thread_id]->Clear();
        attr_ft = param.executors[thread_id]->Run(bucket_id);
    }

    auto check_func = [&ft, &attr_ft, &ids](InnerIdType local_id, InnerIdType origin_id) {
        return (ft == nullptr or ft->CheckValid(origin_id)) and
               (attr_ft == nullptr or attr_ft->CheckValid(local_id));
    };

    // Find a valid (non-hole) entry point
    InnerIdType entry = 0;
    while (entry < static_cast<InnerIdType>(bucket_size) &&
           ids[entry] == std::numeric_limits<InnerIdType>::max()) {
        ++entry;
    }
    if (entry >= static_cast<InnerIdType>(bucket_size)) {
        return;  // all entries are holes, fall back
    }
    float entry_dist = bucket->QueryOneById(computer, bucket_id, entry);
    auto origin_entry = ids[entry] / buckets_per_data;
    if (reasoning_ctx != nullptr) {
        reasoning_ctx->RecordVisit(origin_entry, entry_dist, 0);
    }
    if (check_func(entry, origin_entry)) {
        top_candidates.Push(entry_dist, entry);
    } else if (reasoning_ctx != nullptr) {
        reasoning_ctx->RecordFilterReject(origin_entry);
    }
    candidate_set.Push(-entry_dist, entry);
    vl.Set(entry);

    auto lower_bound = std::numeric_limits<float>::max();
    if (not top_candidates.Empty()) {
        lower_bound = top_candidates.Top().first;
    }

    const auto max_degree = graph->MaximumDegree();
    Vector<InnerIdType> neighbors(allocator_);

    while (not candidate_set.Empty()) {
        auto current_node_pair = candidate_set.Top();

        if ((-current_node_pair.first) > lower_bound and top_candidates.Size() >= ef) {
            break;
        }
        candidate_set.Pop();

        const auto neighbor_count = graph->GetNeighborSize(current_node_pair.second);
        if (neighbor_count > max_degree) {
            continue;
        }
        graph->GetNeighbors(current_node_pair.second, neighbors);

        for (uint32_t i = 0; i < neighbors.size(); ++i) {
            auto neighbor_id = neighbors[i];
            if (vl.Get(neighbor_id)) {
                continue;
            }
            vl.Set(neighbor_id);

            float d = bucket->QueryOneById(computer, bucket_id, neighbor_id);
            auto origin_id = ids[neighbor_id] / buckets_per_data;
            if (reasoning_ctx != nullptr) {
                reasoning_ctx->RecordVisit(origin_id, d, 1);
            }

            if (top_candidates.Size() < ef or lower_bound > d or
                (param.search_mode == RANGE_SEARCH and d <= param.radius + THRESHOLD_ERROR)) {
                candidate_set.Push(-d, neighbor_id);
                if (check_func(neighbor_id, origin_id)) {
                    top_candidates.Push(d, neighbor_id);
                } else if (reasoning_ctx != nullptr) {
                    reasoning_ctx->RecordFilterReject(origin_id);
                }
                if (top_candidates.Size() > ef) {
                    if (reasoning_ctx != nullptr) {
                        reasoning_ctx->RecordEviction(
                            ids[top_candidates.Top().second] / buckets_per_data, 1);
                    }
                    top_candidates.Pop();
                }
                if (not top_candidates.Empty()) {
                    lower_bound = top_candidates.Top().first;
                }
            }
        }
    }

    if (param.search_mode == KNN_SEARCH) {
        while (top_candidates.Size() > topk_u) {
            top_candidates.Pop();
        }
    } else {
        while (not top_candidates.Empty() and
               top_candidates.Top().first > param.radius + THRESHOLD_ERROR) {
            top_candidates.Pop();
        }
        if (topk_u > 0) {
            while (top_candidates.Size() > topk_u) {
                top_candidates.Pop();
            }
        }
    }

    while (not top_candidates.Empty()) {
        auto [d, local_id] = top_candidates.Top();
        heap->Push(d, ids[local_id]);
        top_candidates.Pop();
    }
    while (heap->Size() > topk_u) {
        heap->Pop();
    }
}

}  // namespace vsag
