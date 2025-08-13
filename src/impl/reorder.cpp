
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

#include "reorder.h"

namespace vsag {

DistHeapPtr
Reorder::ReorderByFlatten(const DistHeapPtr& input,
                          const FlattenInterfacePtr& flatten,
                          const float* query,
                          Allocator* allocator,
                          int64_t topk) {
    auto reorder_heap = DistanceHeap::MakeInstanceBySize<true, true>(allocator, topk);
    auto computer = flatten->FactoryComputer(query);
    size_t candidate_size = input->Size();
    const auto* candidate_result = input->GetData();
    Vector<InnerIdType> ids(candidate_size, allocator);
    Vector<float> dists(candidate_size, allocator);
    for (int i = 0; i < candidate_size; ++i) {
        ids[i] = candidate_result[i].second;
    }
    flatten->Query(dists.data(), computer, ids.data(), candidate_size);
    for (int i = 0; i < candidate_size; ++i) {
        if (reorder_heap->Size() < topk || dists[i] < reorder_heap->Top().first) {
            reorder_heap->Push(dists[i], candidate_result[i].second);
            if (reorder_heap->Size() > topk) {
                reorder_heap->Pop();
            }
        }
    }
    return reorder_heap;
}

}  // namespace vsag
