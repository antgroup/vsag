
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

#include "visited_list.h"

#include <limits>

namespace vsag {
VisitedList::VisitedList(InnerIdType max_size, Allocator* allocator)
    : max_size_(max_size), allocator_(allocator) {
    this->list_ = reinterpret_cast<VisitedListType*>(
        allocator_->Allocate((uint64_t)max_size * sizeof(VisitedListType)));
    memset(list_, 0, max_size_ * sizeof(VisitedListType));
    tag_ = 1;
}

VisitedList::~VisitedList() {
    allocator_->Deallocate(list_);
}

void
VisitedList::Reset() {
    if (tag_ == std::numeric_limits<VisitedListType>::max()) {
        memset(list_, 0, max_size_ * sizeof(VisitedListType));
        tag_ = 0;
    }
    ++tag_;
}
}  // namespace vsag
