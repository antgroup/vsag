
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

#include "memmove_heap.h"

namespace vsag {
template <bool max_heap, bool fixed_size>
MemmoveHeap<max_heap, fixed_size>::MemmoveHeap(Allocator* allocator, int64_t max_size)
    : DistanceHeap(allocator, max_size), ordered_buffer_(allocator) {
    if constexpr (fixed_size) {
        ordered_buffer_.resize(max_size + 1);
    }
}

template <bool max_heap, bool fixed_size>
void
MemmoveHeap<max_heap, fixed_size>::Push(float dist, InnerIdType id) {
    using CompareType = typename std::conditional<max_heap, CompareMax, CompareMin>::type;
    if constexpr (fixed_size) {
        if (this->Size() < max_size_ or (dist < this->Top().first) == max_heap) {
            DistanceRecord record = {dist, id};
            auto pos = std::upper_bound(this->ordered_buffer_.begin(),
                                        this->ordered_buffer_.begin() + cur_size_,
                                        record,
                                        CompareType()) -
                       this->ordered_buffer_.begin();
            // NOLINTNEXTLINE(bugprone-undefined-memory-manipulation)
            std::memmove(ordered_buffer_.data() + pos + 1,
                         ordered_buffer_.data() + pos,
                         (max_size_ - pos) * sizeof(DistanceRecord));
            ordered_buffer_[pos] = record;
            this->cur_size_++;
            if (this->Size() > this->max_size_) {
                this->Pop();
            }
        }
    } else {
        DistanceRecord record = {dist, id};
        auto pos =
            std::upper_bound(
                this->ordered_buffer_.begin(), this->ordered_buffer_.end(), record, CompareType()) -
            this->ordered_buffer_.begin();
        ordered_buffer_.emplace_back(record);
        // NOLINTNEXTLINE(bugprone-undefined-memory-manipulation)
        std::memmove(ordered_buffer_.data() + pos + 1,
                     ordered_buffer_.data() + pos,
                     (cur_size_ - pos) * sizeof(DistanceRecord));
        ordered_buffer_[pos] = record;
        cur_size_++;
    }
}

template class MemmoveHeap<true, true>;
template class MemmoveHeap<true, false>;
template class MemmoveHeap<false, true>;
template class MemmoveHeap<false, false>;

}  // namespace vsag
