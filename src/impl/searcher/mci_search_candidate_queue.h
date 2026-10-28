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

#include <cstring>

#include "typing.h"

namespace vsag {

struct MCISearchCandidate {
    float distance{0.0F};
    InnerIdType inner_id{0};
    bool expanded{false};
    uint32_t depth{1};
};

inline bool
mci_search_candidate_less(const MCISearchCandidate& lhs, const MCISearchCandidate& rhs) {
    if (lhs.distance != rhs.distance) {
        return lhs.distance < rhs.distance;
    }
    return lhs.inner_id < rhs.inner_id;
}

class MCISearchCandidateQueue {
public:
    explicit MCISearchCandidateQueue(Allocator* allocator)
        : data_(allocator), allocator_(allocator) {
    }

    [[nodiscard]] Allocator*
    GetAllocator() const {
        return allocator_;
    }

    void
    Reset(uint64_t new_capacity) {
        capacity_ = new_capacity;
        size_ = 0;
        current_unexpanded_ = 0;
        if (data_.size() < capacity_ + 1) {
            data_.resize(capacity_ + 1);
        }
    }

    [[nodiscard]] bool
    CanUpdate(float distance) const {
        return size_ < capacity_ or distance < data_[size_ - 1].distance;
    }

    void
    Insert(float distance, InnerIdType inner_id, uint32_t depth) {
        if (capacity_ == 0 or not CanUpdate(distance)) {
            return;
        }
        MCISearchCandidate candidate{distance, inner_id, false, depth};
        uint64_t lo = 0;
        uint64_t hi = size_;
        while (lo < hi) {
            const auto mid = (lo + hi) >> 1U;
            if (mci_search_candidate_less(candidate, data_[mid])) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        if (lo < capacity_ and lo < size_) {
            std::memmove(data_.data() + lo + 1,
                         data_.data() + lo,
                         (size_ - lo) * sizeof(MCISearchCandidate));
        }
        if (lo < capacity_) {
            data_[lo] = candidate;
        }
        if (size_ < capacity_) {
            ++size_;
        }
        if (lo < current_unexpanded_) {
            current_unexpanded_ = lo;
        }
    }

    MCISearchCandidate*
    GetClosestUnexpanded() {
        while (current_unexpanded_ < size_ and data_[current_unexpanded_].expanded) {
            ++current_unexpanded_;
        }
        if (current_unexpanded_ >= size_) {
            return nullptr;
        }
        data_[current_unexpanded_].expanded = true;
        return &data_[current_unexpanded_];
    }

    [[nodiscard]] uint64_t
    Size() const {
        return size_;
    }

    [[nodiscard]] const MCISearchCandidate*
    Data() const {
        return data_.data();
    }

private:
    Vector<MCISearchCandidate> data_;
    Allocator* allocator_{nullptr};
    uint64_t size_{0};
    uint64_t capacity_{0};
    uint64_t current_unexpanded_{0};
};

}  // namespace vsag
