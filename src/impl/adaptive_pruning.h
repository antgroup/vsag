
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

#include <functional>

#include "parameter.h"

namespace vsag {

struct AdaptivePruningParameter {
    bool enabled{false};
    float adjust_step{0.06F};
    bool apply_to_reverse{false};
    bool apply_to_upper{false};

    void
    FromJson(const JsonType& json);
    JsonType
    ToJson() const;
    void
    Validate(float alpha) const;
};

enum class AdaptivePruningBranch { SINGLE, RELAX, TIGHTEN };

struct AdaptivePruningStats {
    uint64_t initial_accepted{0};
    uint64_t initial_rejected{0};
    uint64_t distance_calls{0};
    float second_alpha{0};
    AdaptivePruningBranch branch{AdaptivePruningBranch::SINGLE};
};

using PruningCandidate = std::pair<float, InnerIdType>;

// Pure L2 selector. Candidates are normalized in place; the returned list is nearest first.
// The caller decides whether this policy applies to its graph layer and selection role.
Vector<PruningCandidate>
select_edges_adaptive(Vector<PruningCandidate>& candidates,
                      InnerIdType center,
                      uint64_t target_degree,
                      float alpha,
                      const AdaptivePruningParameter& parameter,
                      const std::function<float(InnerIdType, InnerIdType)>& distance,
                      Allocator* allocator,
                      AdaptivePruningStats* stats = nullptr);

}  // namespace vsag
