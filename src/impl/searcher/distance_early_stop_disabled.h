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

#include "impl/inner_search_param.h"
#include "query_context.h"
#include "vsag/allocator.h"

namespace vsag {

// Per-traversal state of the distance early stop. The disabled specialization is empty, so a
// traversal without early stop carries no extra state.
template <bool enabled>
class DistanceEarlyStop {
public:
    DistanceEarlyStop(Allocator* /*allocator*/,
                      const InnerSearchParam& /*param*/,
                      const QueryContext* /*ctx*/) {
    }

    void
    Record(float /*score*/) {
    }
};

// Defined in distance_early_stop.h. The traversal headers include only this file, so the
// translation units that compile the default traversals do not contain the enabled state: its
// code would count towards GCC's per-unit inlining limits and change how those traversals are
// compiled.
template <>
class DistanceEarlyStop<true>;

}  // namespace vsag
