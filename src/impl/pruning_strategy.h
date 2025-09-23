
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

#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER2(DistHeap, DistanceHeap);
DEFINE_POINTER(FlattenInterface);
DEFINE_POINTER(GraphInterface);
DEFINE_POINTER(MutexArray);

void
select_edges_by_heuristic(const DistHeapPtr& edges,
                          uint64_t max_size,
                          const FlattenInterfacePtr& flatten,
                          Allocator* allocator,
                          float alpha = 1.0F);

InnerIdType
mutually_connect_new_element(InnerIdType cur_c,
                             const DistHeapPtr& top_candidates,
                             const GraphInterfacePtr& graph,
                             const FlattenInterfacePtr& flatten,
                             const MutexArrayPtr& neighbors_mutexes,
                             Allocator* allocator,
                             float alpha = 1.0F);

}  // namespace vsag
