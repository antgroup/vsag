
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

#include "proximity_scorer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace vsag {

namespace {

// Find the minimum distance between two sorted position lists.
// For unordered mode: returns min |a - b| over all (a, b) pairs.
// For ordered mode: returns min distance considering forward/reverse penalty.
//   Forward (a < b): dist = b - a
//   Reverse (a > b): dist = (a - b) * 2
uint32_t
min_distance_between_lists(const std::vector<uint16_t>& list_a,
                           const std::vector<uint16_t>& list_b,
                           bool ordered) {
    uint32_t min_dist = std::numeric_limits<uint32_t>::max();

    // Both lists are expected to be in insertion order (ascending for positions
    // from a single document scan). Use two-pointer merge for efficiency.
    if (!ordered) {
        // Unordered: classic sorted merge to find min |a - b|
        uint64_t i = 0;
        uint64_t j = 0;
        while (i < list_a.size() && j < list_b.size()) {
            uint32_t a = list_a[i];
            uint32_t b = list_b[j];
            uint32_t dist = (a > b) ? (a - b) : (b - a);
            if (dist < min_dist) {
                min_dist = dist;
            }
            if (min_dist == 0) {
                break;
            }
            if (a < b) {
                ++i;
            } else {
                ++j;
            }
        }
    } else {
        // Ordered: need to check all pairs to find best forward or penalized reverse
        for (uint16_t a : list_a) {
            for (uint16_t b : list_b) {
                uint32_t dist;
                if (a <= b) {
                    // Forward: term_i appears before term_j, as expected
                    dist = b - a;
                } else {
                    // Reverse: term_i appears after term_j, penalty ×2
                    dist = (a - b) * 2;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
        }
    }

    return min_dist;
}

}  // namespace

float
compute_pairwise_proximity(const std::vector<std::vector<uint16_t>>& position_lists, bool ordered) {
    float boost = 0.0f;
    uint64_t n = position_lists.size();

    for (uint64_t i = 0; i < n; ++i) {
        if (position_lists[i].empty()) {
            continue;
        }
        for (uint64_t j = i + 1; j < n; ++j) {
            if (position_lists[j].empty()) {
                continue;
            }
            uint32_t dist =
                min_distance_between_lists(position_lists[i], position_lists[j], ordered);
            if (dist < std::numeric_limits<uint32_t>::max()) {
                boost += 1.0f / static_cast<float>(dist + 1);
            }
        }
    }

    return boost;
}

void
extract_positions_from_token_sequence(const uint32_t* token_sequence,
                                      uint32_t seq_len,
                                      const uint32_t* ids,
                                      uint32_t ids_len,
                                      uint32_t max_positions_per_term,
                                      std::vector<std::vector<uint16_t>>& out_positions) {
    out_positions.clear();
    out_positions.resize(ids_len);

    if (token_sequence == nullptr || seq_len == 0 || ids == nullptr || ids_len == 0) {
        return;
    }

    // Build a map from term_id → index in ids[] for O(1) lookup
    std::unordered_map<uint32_t, std::vector<uint32_t>> term_to_indices;
    for (uint32_t i = 0; i < ids_len; ++i) {
        term_to_indices[ids[i]].push_back(i);
    }

    // Scan token sequence and collect positions
    for (uint32_t pos = 0; pos < seq_len; ++pos) {
        auto it = term_to_indices.find(token_sequence[pos]);
        if (it != term_to_indices.end()) {
            for (uint32_t idx : it->second) {
                if (out_positions[idx].size() < max_positions_per_term) {
                    out_positions[idx].push_back(static_cast<uint16_t>(pos));
                }
            }
        }
    }
}

}  // namespace vsag
