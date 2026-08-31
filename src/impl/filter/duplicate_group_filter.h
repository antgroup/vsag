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

#include <utility>

#include "datacell/graph_interface.h"
#include "vsag/filter.h"

namespace vsag {

class DuplicateGroupFilter final : public Filter {
public:
    DuplicateGroupFilter(FilterPtr filter, GraphInterfacePtr graph)
        : filter_(std::move(filter)), graph_(std::move(graph)) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        const auto group_id = graph_->GetGroupId(static_cast<InnerIdType>(id));
        if (filter_->CheckValid(group_id)) {
            return true;
        }
        return graph_->AnyDuplicateId(group_id, [this](InnerIdType duplicate_id) {
            return filter_->CheckValid(duplicate_id);
        });
    }

    [[nodiscard]] float
    ValidRatio() const override {
        // A duplicate group can only increase the effective valid ratio. Returning the source
        // ratio is conservative: it preserves the existing skip policy while CheckValid keeps a
        // representative traversable when only one of its aliases passes the filter.
        return filter_->ValidRatio();
    }

private:
    FilterPtr filter_;
    GraphInterfacePtr graph_;
};

inline FilterPtr
MakeDuplicateGroupFilter(const FilterPtr& filter,
                         const GraphInterfacePtr& graph,
                         bool consider_duplicate) {
    if (not consider_duplicate or filter == nullptr) {
        return filter;
    }
    return std::make_shared<DuplicateGroupFilter>(filter, graph);
}

}  // namespace vsag
