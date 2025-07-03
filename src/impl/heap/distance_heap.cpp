
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

#include "distance_heap.h"

#include "memmove_heap.h"
#include "standard_heap.h"

namespace vsag {
template <bool max_heap, bool fixed_size>
DistHeapPtr
DistanceHeap::MakeInstanceBySize(Allocator* allocator, int64_t max_size) {
    constexpr static int64_t memmove_maxsize = 10;
    if (max_size < memmove_maxsize) {
        return std::make_shared<MemmoveHeap<max_heap, fixed_size>>(allocator, max_size);
    }
    return std::make_shared<StandardHeap<max_heap, fixed_size>>(allocator, max_size);
}

template DistHeapPtr
DistanceHeap::MakeInstanceBySize<true, true>(Allocator* allocator, int64_t max_size);
template DistHeapPtr
DistanceHeap::MakeInstanceBySize<true, false>(Allocator* allocator, int64_t max_size);
template DistHeapPtr
DistanceHeap::MakeInstanceBySize<false, true>(Allocator* allocator, int64_t max_size);
template DistHeapPtr
DistanceHeap::MakeInstanceBySize<false, false>(Allocator* allocator, int64_t max_size);

DistanceHeap::DistanceHeap(Allocator* allocator) : DistanceHeap(allocator, -1){};

DistanceHeap::DistanceHeap(Allocator* allocator, int64_t max_size)
    : allocator_(allocator), max_size_(max_size){};

void
DistanceHeap::Push(const DistanceRecord& record) {
    return this->Push(record.first, record.second);
}
}  // namespace vsag
