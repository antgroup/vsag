
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

#include <cstdint>
#include <vector>

namespace vsag {

// Compute pairwise proximity boost for a set of query terms' position lists.
//
// For each pair of terms (i, j) where i < j, finds the minimum distance between
// their position lists and accumulates 1/(min_dist + 1).
//
// When ordered=true, the distance for reversed pairs (pos_i > pos_j) is doubled
// as a penalty (similar to Lucene's slop cost for reversals).
//
// Returns 0.0 if fewer than 2 non-empty position lists are provided.
float
compute_pairwise_proximity(const std::vector<std::vector<uint16_t>>& position_lists, bool ordered);

// Extract per-term positions from a raw token sequence.
//
// For each term in ids[0..ids_len), scans token_sequence to find all positions
// where that term appears. Caps each term's positions at max_positions_per_term.
// Only terms present in ids[] are extracted; other tokens are ignored.
//
// out_positions is resized to ids_len, with out_positions[i] containing the
// positions for ids[i].
void
extract_positions_from_token_sequence(const uint32_t* token_sequence,
                                      uint32_t seq_len,
                                      const uint32_t* ids,
                                      uint32_t ids_len,
                                      uint32_t max_positions_per_term,
                                      std::vector<std::vector<uint16_t>>& out_positions);

}  // namespace vsag
