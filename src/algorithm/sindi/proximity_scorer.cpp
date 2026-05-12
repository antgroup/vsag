
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
#include <functional>
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

bool
check_phrase_constraint(const std::vector<std::vector<uint16_t>>& phrase_term_positions,
                        uint32_t slop,
                        bool ordered) {
    uint64_t n = phrase_term_positions.size();
    if (n == 0) {
        return true;
    }

    // All terms must be present
    for (uint64_t i = 0; i < n; ++i) {
        if (phrase_term_positions[i].empty()) {
            return false;
        }
    }

    if (n == 1) {
        return true;
    }

    uint32_t max_span = slop + static_cast<uint32_t>(n) - 1;

    if (!ordered) {
        // Unordered: find minimum span window covering one position from each term.
        // Use multi-pointer approach: merge all positions with term index,
        // sort by position, then slide a window that covers all terms.
        struct PosEntry {
            uint16_t pos;
            uint64_t term_idx;
        };
        std::vector<PosEntry> all_positions;
        for (uint64_t i = 0; i < n; ++i) {
            for (auto pos : phrase_term_positions[i]) {
                all_positions.push_back({pos, i});
            }
        }
        std::sort(all_positions.begin(),
                  all_positions.end(),
                  [](const PosEntry& a, const PosEntry& b) { return a.pos < b.pos; });

        // Sliding window: track count of each term in window
        std::vector<uint32_t> term_count(n, 0);
        uint64_t terms_covered = 0;
        uint64_t left = 0;

        for (uint64_t right = 0; right < all_positions.size(); ++right) {
            auto idx = all_positions[right].term_idx;
            if (term_count[idx] == 0) {
                terms_covered++;
            }
            term_count[idx]++;

            // Shrink window from left while all terms still covered
            while (terms_covered == n) {
                uint32_t span = all_positions[right].pos - all_positions[left].pos;
                if (span <= max_span) {
                    return true;
                }
                auto left_idx = all_positions[left].term_idx;
                term_count[left_idx]--;
                if (term_count[left_idx] == 0) {
                    terms_covered--;
                }
                left++;
            }
        }
        return false;
    } else {
        // Ordered: find positions p0 < p1 < ... < p_{n-1} with span <= max_span.
        // Use recursive/backtracking with pruning, or multi-pointer approach.
        // For small n (typically 2-5 terms), brute force over sorted positions is fine.

        // Build sorted position lists (should already be sorted from extraction)
        // Use a greedy approach: for each starting position of term 0,
        // find the smallest valid position for term 1 > prev, etc.
        std::function<bool(uint64_t, uint16_t, uint16_t)> find_ordered;
        find_ordered = [&](uint64_t term_idx, uint16_t min_pos, uint16_t start_pos) -> bool {
            if (term_idx == n) {
                return true;  // all terms placed
            }
            const auto& positions = phrase_term_positions[term_idx];
            for (auto pos : positions) {
                if (pos < min_pos) {
                    continue;
                }
                if (pos - start_pos > max_span) {
                    break;  // positions are sorted, further ones will exceed span too
                }
                if (find_ordered(term_idx + 1, pos + 1, start_pos)) {
                    return true;
                }
            }
            return false;
        };

        // Try each position of term 0 as the starting point
        for (auto start : phrase_term_positions[0]) {
            if (find_ordered(1, start + 1, start)) {
                return true;
            }
        }
        return false;
    }
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
