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

#include <algorithm>
#include <atomic>
#include <utility>
#include <vector>

#include "basic_types.h"
#include "typing.h"

namespace vsag {

/// Final serial coverage pass shared by both full-build paths, after all workers have joined.
/// Relax clique distance constraints for remaining seeds, using the seed and graph neighbors.
/// VisitNeighbors(seed, visitor) must stop when visitor returns false. Coverage counters must
/// describe the memberships already stored in cliques; only newly stored members are counted.
template <typename Count, typename VisitNeighbors>
uint64_t
// NOLINTNEXTLINE(readability-identifier-naming)
EnsureMCICliqueCoverage(uint64_t total,
                        uint64_t max_degree,
                        uint64_t clique_max,
                        std::vector<std::atomic<Count>>& coverage,
                        VisitNeighbors visit_neighbors,
                        Vector<Vector<InnerIdType>>& cliques,
                        Allocator* allocator) {
    uint64_t repaired = 0;
    const auto candidate_cap = std::max<uint64_t>(1, std::min(max_degree, total));
    const auto member_cap = std::max<uint64_t>(1, clique_max);
    for (uint64_t raw_seed = 0; raw_seed < total; ++raw_seed) {
        const auto seed = static_cast<InnerIdType>(raw_seed);
        if (coverage[seed].load(std::memory_order_relaxed) != 0) {
            continue;
        }
        Vector<InnerIdType> clique(allocator);
        clique.reserve(candidate_cap);
        clique.push_back(seed);
        if (clique.size() < candidate_cap) {
            visit_neighbors(seed, [&](InnerIdType neighbor) {
                if (neighbor < total and
                    std::find(clique.begin(), clique.end(), neighbor) == clique.end()) {
                    clique.push_back(neighbor);
                }
                return clique.size() < candidate_cap;
            });
        }
        std::sort(clique.begin(), clique.end());
        if (clique.size() > member_cap) {
            clique.resize(member_cap);
            if (std::find(clique.begin(), clique.end(), seed) == clique.end()) {
                clique.back() = seed;
            }
        }
        cliques.push_back(std::move(clique));
        for (auto id : cliques.back()) {
            coverage[id].fetch_add(1, std::memory_order_relaxed);
        }
        ++repaired;
    }
    return repaired;
}

}  // namespace vsag
