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

#include "auxiliary_standard_heap.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace vsag {

template <bool max_heap, bool fixed_size>
AuxiliaryStandardHeap<max_heap, fixed_size>::AuxiliaryStandardHeap(Allocator* allocator,
                                                                   int64_t max_size)
    : DistanceHeap(allocator, max_size), queue_(allocator), distance_records_(allocator) {
    if (max_size > 0) {
        this->queue_.reserve(static_cast<uint64_t>(max_size));
    }
}

template <bool max_heap, bool fixed_size>
void
AuxiliaryStandardHeap<max_heap, fixed_size>::Push(float dist, InnerIdType id) {
    this->PushWithAuxiliary(dist, id, std::numeric_limits<float>::quiet_NaN());
}

template <bool max_heap, bool fixed_size>
void
AuxiliaryStandardHeap<max_heap, fixed_size>::PushWithAuxiliary(float dist,
                                                               InnerIdType id,
                                                               float auxiliary) {
    this->PushWithAuxiliary(dist,
                            id,
                            auxiliary,
                            -1,
                            std::numeric_limits<InnerIdType>::max(),
                            std::numeric_limits<uint64_t>::max());
}

template <bool max_heap, bool fixed_size>
void
AuxiliaryStandardHeap<max_heap, fixed_size>::PushWithAuxiliary(float dist,
                                                               InnerIdType id,
                                                               float auxiliary,
                                                               BucketIdType source_bucket_id,
                                                               InnerIdType source_offset_id,
                                                               uint64_t source_version) {
    if constexpr (fixed_size) {
        if (this->max_size_ == 0) {
            return;
        }
        if (this->queue_.size() == this->max_size_) {
            if constexpr (max_heap) {
                if (dist > this->queue_.front().record.first) {
                    return;
                }
            } else {
                if (dist < this->queue_.front().record.first) {
                    return;
                }
            }
        }
    }

    this->queue_.emplace_back(AuxiliaryDistanceRecord{
        {dist, id}, auxiliary, source_bucket_id, source_offset_id, 0, source_version});
    if constexpr (max_heap) {
        std::push_heap(this->queue_.begin(), this->queue_.end(), CompareAuxiliaryMax());
    } else {
        std::push_heap(this->queue_.begin(), this->queue_.end(), CompareAuxiliaryMin());
    }
    this->distance_records_dirty_ = true;

    if constexpr (fixed_size) {
        if (this->queue_.size() > this->max_size_) {
            this->Pop();
        }
    }
}

template <bool max_heap, bool fixed_size>
void
AuxiliaryStandardHeap<max_heap, fixed_size>::Pop() {
    if constexpr (max_heap) {
        std::pop_heap(this->queue_.begin(), this->queue_.end(), CompareAuxiliaryMax());
    } else {
        std::pop_heap(this->queue_.begin(), this->queue_.end(), CompareAuxiliaryMin());
    }
    this->queue_.pop_back();
    this->distance_records_dirty_ = true;
}

template <bool max_heap, bool fixed_size>
const DistanceHeap::DistanceRecord*
AuxiliaryStandardHeap<max_heap, fixed_size>::GetData() const {
    if (this->distance_records_dirty_) {
        this->distance_records_.resize(this->queue_.size());
        for (uint64_t i = 0; i < this->queue_.size(); ++i) {
            this->distance_records_[i] = this->queue_[i].record;
        }
        this->distance_records_dirty_ = false;
    }
    return this->distance_records_.data();
}

template class AuxiliaryStandardHeap<true, true>;
template class AuxiliaryStandardHeap<true, false>;
template class AuxiliaryStandardHeap<false, true>;
template class AuxiliaryStandardHeap<false, false>;

}  // namespace vsag
