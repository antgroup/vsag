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
class AuxiliaryStandardHeap : public DistanceHeap {
public:
    explicit AuxiliaryStandardHeap(Allocator* allocator, int64_t max_size);

    ~AuxiliaryStandardHeap() override = default;

    void
    Push(float dist, InnerIdType id) override;

    void
    PushWithAuxiliary(float dist, InnerIdType id, float auxiliary) override;

    void
    PushWithAuxiliary(float dist,
                      InnerIdType id,
                      float auxiliary,
                      BucketIdType source_bucket_id,
                      InnerIdType source_offset_id,
                      uint64_t source_version) override;

    [[nodiscard]] const DistanceRecord&
    Top() const override {
        return this->queue_.front().record;
    }

    void
    Pop() override;

    [[nodiscard]] uint64_t
    Size() const override {
        return this->queue_.size();
    }

    [[nodiscard]] bool
    Empty() const override {
        return this->queue_.empty();
    }

    [[nodiscard]] const DistanceRecord*
    GetData() const override;

    [[nodiscard]] bool
    StoresAuxiliary() const override {
        return true;
    }

    [[nodiscard]] const AuxiliaryDistanceRecord*
    GetDataWithAuxiliary() const override {
        return this->queue_.data();
    }

private:
    struct CompareAuxiliaryMax {
        constexpr bool
        operator()(const AuxiliaryDistanceRecord& lhs,
                   const AuxiliaryDistanceRecord& rhs) const noexcept {
            return lhs.record.first < rhs.record.first;
        }
    };

    struct CompareAuxiliaryMin {
        constexpr bool
        operator()(const AuxiliaryDistanceRecord& lhs,
                   const AuxiliaryDistanceRecord& rhs) const noexcept {
            return lhs.record.first > rhs.record.first;
        }
    };

    Vector<AuxiliaryDistanceRecord> queue_;
    mutable Vector<DistanceRecord> distance_records_;
    mutable bool distance_records_dirty_{true};
};

}  // namespace vsag
