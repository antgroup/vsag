
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

#include "distance_heap.h"

namespace vsag {
template <bool max_heap = true, bool fixed_size = true>
class MemmoveHeap : public DistanceHeap {
public:
    explicit MemmoveHeap(Allocator* allocator, int64_t max_size);

    ~MemmoveHeap() override = default;

    void
    Push(float dist, InnerIdType id) override;

    [[nodiscard]] const DistanceRecord&
    Top() const override {
        return this->ordered_buffer_[cur_size_ - 1];
    }

    void
    Pop() override {
        cur_size_--;
    }

    [[nodiscard]] uint64_t
    Size() const override {
        return this->cur_size_;
    }

    [[nodiscard]] bool
    Empty() const override {
        return this->cur_size_ == 0;
    }

private:
    Vector<DistanceRecord> ordered_buffer_;

    int64_t cur_size_{0};
};

}  // namespace vsag
