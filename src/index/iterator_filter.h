
// Copyright 2024-present the vsag project
//
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
#include <queue>

#include "typing.h"
#include "utils/visited_list.h"
#include "vsag/allocator.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/iterator_context.h"

namespace vsag {

class IteratorFilterContext : public IteratorContext {
public:
    using VisitedListType = uint16_t;

public:
    IteratorFilterContext() : is_first_used_(true){};
    ~IteratorFilterContext();

    tl::expected<void, Error>
    init(InnerIdType max_size, int64_t ef_search, Allocator* allocator);
    void
    AddDiscardNode(float dis, uint32_t id) override;

    virtual uint32_t
    GetTopID() override;
    virtual float
    GetTopDist() override;
    virtual void
    PopDiscard() override;
    virtual bool
    Empty() override;
    virtual bool
    IsFirstUsed() override;
    virtual void
    SetOFFFirstUsed() override;
    void
    SetPoint(uint32_t id) override;
    bool
    CheckPoint(uint32_t id) override;
    int64_t
    GetDiscardElementNum() override;

private:
    int64_t ef_search_;
    bool is_first_used_;
    uint32_t max_size_;
    Allocator* allocator_;
    VisitedListType* list_{nullptr};
    std::unique_ptr<std::priority_queue<std::pair<float, uint32_t>>> discard_;
};

};  // namespace vsag