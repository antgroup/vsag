
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

#include <type_traits>
#include <utility>

#include "typing.h"
#include "vsag/allocator.h"

namespace vsag {
class DistanceHeap;
using DistHeapPtr = std::shared_ptr<DistanceHeap>;

class DistanceHeap {
public:
    using DistanceRecord = std::pair<float, InnerIdType>;

    template <class HeapImpl>
    HeapImpl&
    GetImpl() {
        return static_cast<HeapImpl>(*this);
    }

    struct CompareMax {
        constexpr bool
        operator()(DistanceRecord const& a, DistanceRecord const& b) const noexcept {
            return a.first < b.first;
        }
    };

    struct CompareMin {
        constexpr bool
        operator()(DistanceRecord const& a, DistanceRecord const& b) const noexcept {
            return a.first > b.first;
        }
    };

public:
    template <bool max_heap, bool fixed_size>
    static DistHeapPtr
    MakeInstanceBySize(Allocator* allocator, int64_t max_size);

public:
    explicit DistanceHeap(Allocator* allocator);

    explicit DistanceHeap(Allocator* allocator, int64_t max_size);

    virtual ~DistanceHeap() = default;

    virtual void
    Push(const DistanceRecord& record);

    virtual void
    Push(float dist, InnerIdType id) = 0;

    [[nodiscard]] virtual const DistanceRecord&
    Top() const = 0;

    virtual void
    Pop() = 0;

    [[nodiscard]] virtual uint64_t
    Size() const = 0;

    [[nodiscard]] virtual bool
    Empty() const = 0;

protected:
    Allocator* allocator_{nullptr};
    int64_t max_size_{-1};
};

using DistHeapPtr = std::shared_ptr<DistanceHeap>;

}  // namespace vsag
