
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
#include <limits>

#include "prefetch.h"
#include "resource_object.h"
#include "resource_object_pool.h"
#include "typing.h"
#include "vsag/allocator.h"

namespace vsag {

class VisitedList : public ResourceObject {
public:
    using VisitedListType = uint16_t;

public:
    explicit VisitedList(InnerIdType max_size, Allocator* allocator)
        : max_size_(max_size), allocator_(allocator) {
        this->list_ = reinterpret_cast<VisitedListType*>(
            allocator_->Allocate((uint64_t)max_size * sizeof(VisitedListType)));
        memset(list_, 0, max_size_ * sizeof(VisitedListType));
        tag_ = 1;
    }

    ~VisitedList() override {
        allocator_->Deallocate(list_);
    }

    inline void
    Set(const InnerIdType& id) {
        this->list_[id] = this->tag_;
    }

    inline bool
    Get(const InnerIdType& id) {
        return this->list_[id] == this->tag_;
    }

    inline void
    Prefetch(const InnerIdType& id) {
        PrefetchLines(this->list_ + id, 64);
    }

    void
    Reset() override {
        if (tag_ == std::numeric_limits<VisitedListType>::max()) {
            memset(list_, 0, max_size_ * sizeof(VisitedListType));
            tag_ = 0;
        }
        ++tag_;
    }

private:
    Allocator* const allocator_{nullptr};

    VisitedListType* list_{nullptr};

    VisitedListType tag_{1};

    const InnerIdType max_size_{0};
};

using VisitedListPool = ResourceObjectPool<VisitedList>;
using VisitedListPtr = std::shared_ptr<VisitedList>;

}  // namespace vsag
