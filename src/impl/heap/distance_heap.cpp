
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

#include "auxiliary_standard_heap.h"
#include "memmove_heap.h"
#include "standard_heap.h"

namespace vsag {
template <bool max_heap, bool fixed_size>
DistHeapPtr
DistanceHeap::MakeInstanceBySize(Allocator* allocator, int64_t max_size) {
    static constexpr int64_t memmove_maxsize = 10;
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

template <bool max_heap, bool fixed_size>
DistHeapPtr
DistanceHeap::MakeInstanceBySizeWithAuxiliary(Allocator* allocator, int64_t max_size) {
    return std::make_shared<AuxiliaryStandardHeap<max_heap, fixed_size>>(allocator, max_size);
}

template DistHeapPtr
DistanceHeap::MakeInstanceBySizeWithAuxiliary<true, true>(Allocator* allocator, int64_t max_size);
template DistHeapPtr
DistanceHeap::MakeInstanceBySizeWithAuxiliary<true, false>(Allocator* allocator, int64_t max_size);
template DistHeapPtr
DistanceHeap::MakeInstanceBySizeWithAuxiliary<false, true>(Allocator* allocator, int64_t max_size);
template DistHeapPtr
DistanceHeap::MakeInstanceBySizeWithAuxiliary<false, false>(Allocator* allocator, int64_t max_size);

DistanceHeap::DistanceHeap(Allocator* allocator) : DistanceHeap(allocator, -1){};

DistanceHeap::DistanceHeap(Allocator* allocator, int64_t max_size)
    : allocator_(allocator), max_size_(max_size){};

void
DistanceHeap::Push(const DistanceRecord& record) {
    return this->Push(record.first, record.second);
}

void
DistanceHeap::PushWithAuxiliary(float dist, InnerIdType id, float auxiliary) {
    (void)auxiliary;
    this->Push(dist, id);
}

void
DistanceHeap::PushWithAuxiliary(float dist,
                                InnerIdType id,
                                float auxiliary,
                                BucketIdType source_bucket_id,
                                InnerIdType source_offset_id,
                                uint64_t source_version) {
    (void)auxiliary;
    (void)source_bucket_id;
    (void)source_offset_id;
    (void)source_version;
    this->Push(dist, id);
}

bool
DistanceHeap::StoresAuxiliary() const {
    return false;
}

const DistanceHeap::AuxiliaryDistanceRecord*
DistanceHeap::GetDataWithAuxiliary() const {
    return nullptr;
}

void
DistanceHeap::Merge(const DistanceHeap& other) {
    if (other.StoresAuxiliary()) {
        const auto* data = other.GetDataWithAuxiliary();
        for (uint64_t i = 0; i < other.Size(); ++i) {
            this->PushWithAuxiliary(data[i].record.first,
                                    data[i].record.second,
                                    data[i].auxiliary,
                                    data[i].source_bucket_id,
                                    data[i].source_offset_id,
                                    data[i].source_version);
        }
        return;
    }

    const auto* data = other.GetData();
    for (uint64_t i = 0; i < other.Size(); ++i) {
        this->Push(data[i]);
    }
}

}  // namespace vsag
