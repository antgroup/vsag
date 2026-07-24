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

#pragma once

#include <memory>

#include "container_types.h"
#include "datacell/graph_interface.h"
#include "flat_bucket_searcher.h"
#include "impl/heap/distance_heap.h"
#include "impl/heap/standard_heap.h"
#include "ivf_bucket_searcher.h"
#include "typing.h"
#include "utils/visited_list.h"

namespace vsag {

class GraphBucketSearcher : public IVFBucketSearcher {
public:
    GraphBucketSearcher(int64_t graph_build_threshold,
                        const Vector<GraphInterfacePtr>& bucket_graphs,
                        Allocator* allocator);

    void
    Search(BucketIdType bucket_id,
           const BucketInterfacePtr& bucket,
           const ComputerInterfacePtr& computer,
           const InnerSearchParam& param,
           int64_t thread_id,
           int64_t topk,
           BucketIdType buckets_per_data,
           DistHeapPtr& heap,
           Vector<float>& dist,
           ReasoningContext* reasoning_ctx) const override;

private:
    void
    search_graph(BucketIdType bucket_id,
                 const BucketInterfacePtr& bucket,
                 const ComputerInterfacePtr& computer,
                 const InnerSearchParam& param,
                 int64_t thread_id,
                 int64_t topk,
                 BucketIdType buckets_per_data,
                 DistHeapPtr& heap,
                 ReasoningContext* reasoning_ctx) const;

    int64_t graph_build_threshold_;
    // Owned by IVF and must outlive this searcher.
    const Vector<GraphInterfacePtr>& bucket_graphs_;
    Allocator* allocator_;
    std::shared_ptr<FlatBucketSearcher> flat_searcher_;
};

}  // namespace vsag
