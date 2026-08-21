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

#include <algorithm>
#include <limits>

#include "attr/executor/executor.h"
#include "impl/reasoning/search_reasoning.h"
#include "impl/searcher/basic_searcher.h"
#include "vsag_exception.h"

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
    if (graph_fresh and bucket_size >= graph_build_threshold_ and
        not param.use_rabitq_heap_search) {
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
    const auto bucket_size = bucket->GetBucketSize(bucket_id);
    if (bucket_size == 0) {
        return;
    }
    const auto& graph = bucket_graphs_[bucket_id];
    const auto* ids = bucket->GetInnerIds(bucket_id);
    InnerIdType entry = 0;
    while (entry < bucket_size && ids[entry] == std::numeric_limits<InnerIdType>::max()) {
        ++entry;
    }
    if (entry == bucket_size) {
        return;
    }

    Filter* attr_filter = nullptr;
    if (not param.executors.empty() && static_cast<uint64_t>(thread_id) < param.executors.size() &&
        param.executors[thread_id] != nullptr) {
        param.executors[thread_id]->Clear();
        attr_filter = param.executors[thread_id]->Run(bucket_id);
    }
    InnerSearchParam search_param = param;
    search_param.ep = entry;
    uint64_t result_limit = std::numeric_limits<uint64_t>::max();
    if (search_param.search_mode == KNN_SEARCH) {
        CHECK_ARGUMENT(topk > 0, "topk must be greater than 0");
        result_limit = static_cast<uint64_t>(topk);
        search_param.ef = std::max(search_param.ef, result_limit);
    } else if (topk > 0) {
        result_limit = static_cast<uint64_t>(topk);
    }
    search_param.ef = std::min<uint64_t>(search_param.ef, bucket_size);
    search_param.topk = topk;
    if (search_param.search_mode == RANGE_SEARCH) {
        // BasicSearcher uses a non-positive range limit to mean unlimited; clamp only positive
        // limits because InnerSearchParam stores this field as int.
        search_param.range_search_limit_size =
            topk > 0 ? static_cast<int>(std::min<int64_t>(topk, std::numeric_limits<int>::max()))
                     : 0;
    }

    // BucketInterface can create computers only from raw queries, not local IDs. Search only
    // needs query distances, so use the no-factory provider and make pairwise misuse explicit.
    BucketDistanceProvider distance_provider(bucket, bucket_id, computer, ids, buckets_per_data);
    auto visited = std::make_shared<VisitedList>(bucket_size, allocator_);
    QueryContext query_context =
        param.query_context == nullptr ? QueryContext{} : *param.query_context;
    query_context.reasoning_ctx = reasoning_ctx;
    BasicSearcher searcher(allocator_);
    auto top_candidates = searcher.Search(
        graph, distance_provider, visited, search_param, attr_filter, &query_context);
    while (not top_candidates->Empty()) {
        const auto [distance, local_id] = top_candidates->Top();
        heap->Push(distance, ids[local_id]);
        top_candidates->Pop();
    }
    while (heap->Size() > result_limit) {
        heap->Pop();
    }
}

}  // namespace vsag
