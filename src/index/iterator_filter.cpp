
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

#include "iterator_filter.h"

#include "logger.h"

namespace vsag {

IteratorFilterContext::~IteratorFilterContext() {
    allocator_->Deallocate(list_);
}

tl::expected<void, Error>
IteratorFilterContext::init(InnerIdType max_size, int64_t ef_search, Allocator* allocator) {
    if (ef_search == 0 || max_size == 0) {
        LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                              "failed to init: max size or ef_search is empty");
    }
    try {
        ef_search_ = ef_search;
        allocator_ = allocator;
        max_size_ = max_size;
        discard_ = std::make_unique<MaxHeap>(allocator);
        list_ = reinterpret_cast<VisitedListType*>(
            allocator_->Allocate((uint64_t)max_size * sizeof(VisitedListType)));
        memset(list_, 0, max_size * sizeof(VisitedListType));
    } catch (const std::bad_alloc& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::NO_ENOUGH_MEMORY,
                              "failed to init iterator filter(not enough memory): ",
                              e.what());
    }
    return {};
}

void
IteratorFilterContext::AddDiscardNode(float dis, uint32_t inner_id) {
    if (discard_->size() >= ef_search_) {
        if (discard_->top().first > dis) {
            discard_->pop();
            discard_->emplace(dis, inner_id);
        }
    } else {
        discard_->emplace(dis, inner_id);
    }
}

uint32_t
IteratorFilterContext::GetTopID() {
    return discard_->top().second;
}

float
IteratorFilterContext::GetTopDist() {
    return discard_->top().first;
}

void
IteratorFilterContext::PopDiscard() {
    discard_->pop();
}

bool
IteratorFilterContext::Empty() {
    return discard_->empty();
}

bool
IteratorFilterContext::IsFirstUsed() const {
    return is_first_used_;
}

void
IteratorFilterContext::SetOFFFirstUsed() {
    is_first_used_ = false;
}

void
IteratorFilterContext::SetPoint(InnerIdType inner_id) {
    list_[inner_id] = 1;
}

bool
IteratorFilterContext::CheckPoint(InnerIdType inner_id) {
    return list_[inner_id] == 0;
}

int64_t
IteratorFilterContext::GetDiscardElementNum() {
    return int64_t(discard_->size());
}

};  // namespace vsag
