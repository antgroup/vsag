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
#include <utility>

#include "typing.h"

namespace vsag {

enum class PiPNNAdaptivePruningBranch { SINGLE, RELAX, TIGHTEN };

struct PiPNNAdaptivePruningStats {
    uint64_t initial_accepted{0};
    uint64_t initial_rejected{0};
    uint64_t distance_calls{0};
    float second_alpha{0};
    PiPNNAdaptivePruningBranch branch{PiPNNAdaptivePruningBranch::SINGLE};
};

void
validate_pipnn_adaptive_pruning(float alpha, float adjust_step);

// Adapt PR #2933's per-node schedule to PiPNN's already sorted, unique reservoir.
// Return positions in `ordered`, in ascending order, preserving PiPNN's BF16-distance / ID-gap
// tie ordering. Candidate IDs are local IDs and are passed unchanged to the distance callback.
Vector<uint64_t>
select_pipnn_edges_adaptive(const Vector<std::pair<float, uint32_t>>& ordered,
                            uint64_t target_degree,
                            float alpha,
                            float adjust_step,
                            const std::function<float(uint32_t, uint32_t)>& distance,
                            Allocator* allocator,
                            PiPNNAdaptivePruningStats* stats = nullptr);

}  // namespace vsag
