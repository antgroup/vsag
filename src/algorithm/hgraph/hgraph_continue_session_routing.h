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

#include <set>
#include <vector>

#include "typing.h"
#include "vsag_exception.h"

namespace vsag {

// Internal routing primitive shared by the session and deterministic bridge tests.
// Only the unrankable component is traversed; finite routing remains greedy.
template <typename Score, typename Neighbors>
InnerIdType
FindFiniteSessionRoute(InnerIdType entry,
                       Score score,
                       Neighbors neighbors,
                       uint64_t max_bridge_vertices = 65536) {
    if (is_finite_distance(score(entry))) {
        return entry;
    }
    std::set<InnerIdType> seen{entry};
    std::vector<InnerIdType> bridges{entry};
    for (uint64_t cursor = 0; cursor < bridges.size(); ++cursor) {
        for (auto id : neighbors(bridges[cursor])) {
            if (seen.count(id) != 0) {
                continue;
            }
            // Bound routing scratch/work without silently dropping reachable finite exits.
            if (seen.size() >= max_bridge_vertices) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "session non-finite routing bridge limit exceeded");
            }
            seen.insert(id);
            if (is_finite_distance(score(id))) {
                return id;
            }
            bridges.push_back(id);
        }
    }
    return entry;
}
}  // namespace vsag
