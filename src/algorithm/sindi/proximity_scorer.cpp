
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
min_distance_between_lists(const PosSpan& list_a, const PosSpan& list_b, bool ordered) {
    uint32_t min_dist = std::numeric_limits<uint32_t>::max();

    // Both lists are expected to be in insertion order (ascending for positions
    // from a single document scan). Use two-pointer merge for efficiency.
    if (!ordered) {
        // Unordered: classic sorted merge to find min |a - b|
        uint64_t i = 0;
        uint64_t j = 0;
        while (i < list_a.size && j < list_b.size) {
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
        // Ordered: two-pointer merge to find best forward or penalized reverse distance.
        uint64_t i = 0;
        uint64_t j = 0;
        while (i < list_a.size && j < list_b.size) {
            uint32_t a = list_a[i];
            uint32_t b = list_b[j];
            uint32_t dist;
            if (a <= b) {
                dist = b - a;
                ++i;
            } else {
                dist = (a - b) * 2;
                ++j;
            }
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
    }

    return min_dist;
}

}  // namespace

float
all_pairwise_proximity(const std::vector<PosSpan>& position_lists, bool ordered) {
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

float
adjacent_pairwise_proximity(const std::vector<PosSpan>& position_lists, bool ordered) {
    float boost = 0.0f;
    uint64_t n = position_lists.size();

    // Only score adjacent pairs (i, i+1) in query order → n-1 pairs.
    for (uint64_t i = 0; i + 1 < n; ++i) {
        if (position_lists[i].empty() || position_lists[i + 1].empty()) {
            continue;
        }
        uint32_t dist =
            min_distance_between_lists(position_lists[i], position_lists[i + 1], ordered);
        if (dist < std::numeric_limits<uint32_t>::max()) {
            boost += 1.0f / static_cast<float>(dist + 1);
        }
    }

    return boost;
}

namespace {

// File-private helper. It lives in an anonymous namespace so it has *internal
// linkage* (visible only inside this translation unit, the modern replacement
// for a file-level `static` function); it is the shared core of the two
// sloppy_phrase_match overloads below and is not meant to be called elsewhere.
//
// --- What "sloppy phrase" means ---
// For a query phrase like "A B C", a document still matches even if the terms
// are not perfectly consecutive, as long as their layout is within a bounded
// "slop" of the ideal. This mirrors Lucene's SloppyPhraseMatcher.
//
// --- The normalization trick ---
// Query term i has an ideal offset i (A=0, B=1, C=2). For a real position `pos`
// of term i we store norm = pos - i. If the terms are perfectly consecutive,
// every term's norm collapses to the SAME value, so the spread
// (max_norm - min_norm) is 0; the more the layout deviates, the larger the
// spread. A reversed pair produces a smaller/negative norm, which widens the
// spread and therefore "costs" extra slop -- exactly Lucene's behavior.
//
// --- The sweep ---
// `all_norms` holds one {norm, term_idx} entry per position of every term (the
// caller builds it). We sort it by norm, then slide a window [left, right] over
// the sorted entries, looking for the smallest window that contains at least
// one entry from EVERY term (terms_covered == n). `term_count[t]` tracks how
// many entries of term t are currently inside the window; it must be sized to n
// and is used purely as counting scratch. As soon as a covering window has
// spread <= slop we return true.
//
// --- Worked example ---
// query "A B C" (ideal offsets 0,1,2); document has A@0, B@2, C@4.
//   norms:  A: 0-0=0,  B: 2-1=1,  C: 4-2=2
//   sorted: [ (norm=0,A), (norm=1,B), (norm=2,C) ]
//   the first window covering {A,B,C} is [0..2], spread = 2 - 0 = 2.
//   => matches when slop >= 2, fails when slop < 2.
//
// `all_norms` is sorted in place. Returns true iff some covering window has
// (max_norm - min_norm) <= slop.
bool
sloppy_phrase_sweep(std::vector<NormEntry>& all_norms,
                    std::vector<uint32_t>& term_count,
                    uint64_t n,
                    uint32_t slop) {
    std::sort(all_norms.begin(), all_norms.end(), [](const NormEntry& a, const NormEntry& b) {
        return a.norm < b.norm;
    });

    uint64_t terms_covered = 0;
    uint64_t left = 0;
    for (uint64_t right = 0; right < all_norms.size(); ++right) {
        auto idx = all_norms[right].term_idx;
        if (term_count[idx] == 0) {
            terms_covered++;
        }
        term_count[idx]++;

        while (terms_covered == n) {
            int32_t distance = all_norms[right].norm - all_norms[left].norm;
            if (distance <= static_cast<int32_t>(slop)) {
                return true;
            }
            auto left_idx = all_norms[left].term_idx;
            term_count[left_idx]--;
            if (term_count[left_idx] == 0) {
                terms_covered--;
            }
            left++;
        }
    }
    return false;
}

}  // namespace

bool
sloppy_phrase_match(const std::vector<std::vector<uint16_t>>& phrase_term_positions,
                    uint32_t slop) {
    uint64_t n = phrase_term_positions.size();
    if (n <= 1) {
        return true;
    }

    // All terms must be present.
    for (uint64_t i = 0; i < n; ++i) {
        if (phrase_term_positions[i].empty()) {
            return false;
        }
    }

    // Normalize each position by its query offset (term index): norm = pos - i.
    // Reversals produce smaller/negative norms, widening the window (i.e. they
    // cost extra slop) exactly as Lucene's SloppyPhraseMatcher does.
    std::vector<NormEntry> all_norms;
    for (uint64_t i = 0; i < n; ++i) {
        for (auto pos : phrase_term_positions[i]) {
            all_norms.push_back({static_cast<int32_t>(pos) - static_cast<int32_t>(i), i});
        }
    }
    std::vector<uint32_t> term_count(n, 0);
    return sloppy_phrase_sweep(all_norms, term_count, n, slop);
}

bool
sloppy_phrase_match(const std::vector<PosSpan>& phrase_term_positions,
                    uint32_t slop,
                    std::vector<NormEntry>& norm_buffer,
                    std::vector<uint32_t>& count_buffer) {
    uint64_t n = phrase_term_positions.size();
    if (n <= 1) {
        return true;
    }

    for (uint64_t i = 0; i < n; ++i) {
        if (phrase_term_positions[i].empty()) {
            return false;
        }
    }

    norm_buffer.clear();
    for (uint64_t i = 0; i < n; ++i) {
        const auto& span = phrase_term_positions[i];
        for (uint32_t k = 0; k < span.size; ++k) {
            norm_buffer.push_back({static_cast<int32_t>(span[k]) - static_cast<int32_t>(i), i});
        }
    }
    count_buffer.assign(n, 0);
    return sloppy_phrase_sweep(norm_buffer, count_buffer, n, slop);
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

    // Scan token sequence and collect positions. Positions are stored as uint16_t.
    constexpr uint32_t kMaxStorablePosition =
        static_cast<uint32_t>(std::numeric_limits<uint16_t>::max()) + 1;
    uint32_t limit = std::min(seq_len, kMaxStorablePosition);
    for (uint32_t pos = 0; pos < limit; ++pos) {
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
