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
#include <boost/dynamic_bitset.hpp>
#include <cstdint>
#include <utility>
#include <vector>

#include "basic_types.h"
#include "utils/float_utils.h"

namespace vsag::mci {

using CliqueBits = boost::dynamic_bitset<uint64_t>;

// Find a threshold-sized witness in the seed's induced neighborhood. Positions are
// in coverage priority order. A greedy coloring bounds the largest possible extension.
inline bool
FindCliqueWitness(const std::vector<CliqueBits>& adjacency,
                  CliqueBits possible,
                  CliqueBits excluded,
                  size_t threshold,
                  std::vector<size_t>& selected) {
    if (selected.size() == threshold) {
        return true;
    }
    const size_t needed = threshold - selected.size();
    if (possible.count() < needed) {
        return false;
    }
    auto left = possible;
    for (size_t colors = 0; colors < needed; ++colors) {
        if (left.none()) {
            return false;
        }
        auto available = left;
        for (auto v = available.find_first(); v != CliqueBits::npos; v = available.find_first()) {
            left.reset(v);
            available.reset(v);
            available -= adjacency[v];
        }
    }
    const auto pivots = possible | excluded;
    auto pivot = pivots.find_first();
    size_t best = 0;
    for (auto v = pivot; v != CliqueBits::npos; v = pivots.find_next(v)) {
        const auto common = (possible & adjacency[v]).count();
        if (common > best) {
            pivot = v;
            best = common;
        }
    }
    const auto branches = possible - adjacency[pivot];
    for (auto v = branches.find_first(); v != CliqueBits::npos; v = branches.find_next(v)) {
        selected.push_back(v);
        if (FindCliqueWitness(
                adjacency, possible & adjacency[v], excluded & adjacency[v], threshold, selected)) {
            return true;
        }
        selected.pop_back();
        possible.reset(v);
        excluded.set(v);
    }
    return false;
}

// A witness of k neighbors needs at least k-1 edges per vertex in its induced graph.
// Peel low-degree vertices before picking a pivot; their edges cannot support a witness.
inline CliqueBits
CliqueCore(const std::vector<CliqueBits>& adjacency, uint64_t minimum_degree) {
    CliqueBits active(adjacency.size());
    active.set();
    std::vector<uint64_t> degrees;
    std::vector<uint64_t> removed;
    for (uint64_t i = 0; i < adjacency.size(); ++i) {
        degrees.push_back(adjacency[i].count());
        if (degrees[i] < minimum_degree) {
            active.reset(i);
            removed.push_back(i);
        }
    }
    for (uint64_t i = 0; i < removed.size(); ++i) {
        const auto affected = active & adjacency[removed[i]];
        for (auto v = affected.find_first(); v != CliqueBits::npos; v = affected.find_next(v)) {
            if (--degrees[v] < minimum_degree) {
                active.reset(v);
                removed.push_back(v);
            }
        }
    }
    return active;
}

// Seek the first witness among uncovered neighbors. Every candidate in extension_order
// can extend it, including covered and highly subscribed vertices. A zero-size witness
// grows the seed alone at the final radius.
template <typename PairDistance>
std::vector<InnerIdType>
FindMaximalSeedNeighbors(const std::vector<InnerIdType>& extension_order,
                         const std::vector<InnerIdType>& witness_order,
                         const std::vector<float>& seed_distances,
                         float radius,
                         uint64_t threshold,
                         PairDistance&& distance) {
    std::vector<InnerIdType> eligible;
    for (auto id : witness_order) {
        if (seed_distances[id] < radius) {
            eligible.push_back(id);
        }
    }
    if (eligible.size() < threshold) {
        return {};
    }
    std::vector<InnerIdType> chosen;
    auto connected = [&](InnerIdType a, InnerIdType b) {
        const auto value = distance(a, b);
        return IsFiniteFloatBits(value) and value <= radius;
    };
    auto fits = [&](InnerIdType id) {
        return std::all_of(
            chosen.begin(), chosen.end(), [&](InnerIdType other) { return connected(id, other); });
    };
    for (uint64_t i = 0; chosen.size() < threshold; ++i) {
        if (eligible.size() - i < threshold - chosen.size()) {
            break;
        }
        if (fits(eligible[i])) {
            chosen.push_back(eligible[i]);
        }
    }
    if (chosen.size() < threshold) {
        // A greedy miss does not imply that no witness exists at this radius.
        std::vector<CliqueBits> adjacency(eligible.size(), CliqueBits(eligible.size()));
        for (uint64_t i = 0; i < eligible.size(); ++i) {
            for (uint64_t j = 0; j < i; ++j) {
                if (connected(eligible[i], eligible[j])) {
                    adjacency[i].set(j);
                    adjacency[j].set(i);
                }
            }
        }
        std::vector<size_t> positions;
        auto possible = CliqueCore(adjacency, threshold - 1);
        if (not FindCliqueWitness(adjacency,
                                  std::move(possible),
                                  CliqueBits(eligible.size()),
                                  threshold,
                                  positions)) {
            return {};
        }
        chosen.clear();
        for (auto pos : positions) {
            chosen.push_back(eligible[pos]);
        }
    }
    std::vector<uint8_t> included(seed_distances.size(), 0);
    for (auto id : chosen) {
        included[id] = 1;
    }
    // A vertex rejected by one member remains incompatible as the clique grows.
    // One pass therefore certifies inclusion maximality over the full pool.
    for (auto id : extension_order) {
        if (not included[id] and seed_distances[id] < radius and fits(id)) {
            chosen.push_back(id);
        }
    }
    return chosen;
}

}  // namespace vsag::mci
