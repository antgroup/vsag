
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

#include <cmath>
#include <vector>

#include "unittest.h"

#define REQUIRE_APPROX(a, b) REQUIRE(std::abs((a) - (b)) < 1e-6f)

using namespace vsag;

TEST_CASE("ProximityScorer Empty Input", "[ut][ProximityScorer]") {
    // No position lists at all
    std::vector<std::vector<uint16_t>> empty;
    float boost = compute_pairwise_proximity(empty, false);
    REQUIRE(boost == 0.0f);
}

TEST_CASE("ProximityScorer Single Term", "[ut][ProximityScorer]") {
    // Only one term present — no pairs, boost should be 0
    std::vector<std::vector<uint16_t>> single = {{5, 10, 20}};
    float boost = compute_pairwise_proximity(single, false);
    REQUIRE(boost == 0.0f);
}

TEST_CASE("ProximityScorer Two Terms Adjacent", "[ut][ProximityScorer]") {
    // term A at pos 0, term B at pos 1 → dist=1 → 1/(1+1) = 0.5
    std::vector<std::vector<uint16_t>> positions = {{0}, {1}};
    float boost = compute_pairwise_proximity(positions, false);
    REQUIRE_APPROX(boost, 0.5f);
}

TEST_CASE("ProximityScorer Two Terms Same Position", "[ut][ProximityScorer]") {
    // term A at pos 5, term B at pos 5 → dist=0 → 1/(0+1) = 1.0
    std::vector<std::vector<uint16_t>> positions = {{5}, {5}};
    float boost = compute_pairwise_proximity(positions, false);
    REQUIRE_APPROX(boost, 1.0f);
}

TEST_CASE("ProximityScorer Two Terms Far Apart", "[ut][ProximityScorer]") {
    // term A at pos 0, term B at pos 100 → dist=100 → 1/101 ≈ 0.0099
    std::vector<std::vector<uint16_t>> positions = {{0}, {100}};
    float boost = compute_pairwise_proximity(positions, false);
    REQUIRE_APPROX(boost, 1.0f / 101.0f);
}

TEST_CASE("ProximityScorer Two Terms Multiple Positions", "[ut][ProximityScorer]") {
    // term A at [0, 50, 200], term B at [3, 48, 300]
    // min_dist = min(|0-3|, |0-48|, |0-300|, |50-3|, |50-48|, |50-300|, |200-3|, ...) = |50-48| = 2
    // But the algorithm should find min_dist using sorted merge: min over all pairs = 2
    // boost = 1/(2+1) = 0.333...
    std::vector<std::vector<uint16_t>> positions = {{0, 50, 200}, {3, 48, 300}};
    float boost = compute_pairwise_proximity(positions, false);
    REQUIRE_APPROX(boost, 1.0f / 3.0f);
}

TEST_CASE("ProximityScorer Three Terms Pairwise", "[ut][ProximityScorer]") {
    // term A at pos 0, term B at pos 1, term C at pos 200
    // pairs: (A,B) dist=1 → 0.5, (A,C) dist=200 → 1/201, (B,C) dist=199 → 1/200
    // total ≈ 0.5 + 0.00498 + 0.005 ≈ 0.51
    std::vector<std::vector<uint16_t>> positions = {{0}, {1}, {200}};
    float boost = compute_pairwise_proximity(positions, false);
    float expected = 1.0f / 2.0f + 1.0f / 201.0f + 1.0f / 200.0f;
    REQUIRE_APPROX(boost, expected);
}

TEST_CASE("ProximityScorer Ordered Forward", "[ut][ProximityScorer]") {
    // ordered=true, term A at pos 0, term B at pos 5
    // A before B in query order, pos_A < pos_B → forward → dist = 5
    // boost = 1/(5+1) = 1/6
    std::vector<std::vector<uint16_t>> positions = {{0}, {5}};
    float boost = compute_pairwise_proximity(positions, true);
    REQUIRE_APPROX(boost, 1.0f / 6.0f);
}

TEST_CASE("ProximityScorer Ordered Reverse Penalty", "[ut][ProximityScorer]") {
    // ordered=true, term A at pos 10, term B at pos 5
    // A before B in query order, but pos_A > pos_B → reverse → dist = (10-5)*2 = 10
    // boost = 1/(10+1) = 1/11
    std::vector<std::vector<uint16_t>> positions = {{10}, {5}};
    float boost = compute_pairwise_proximity(positions, true);
    REQUIRE_APPROX(boost, 1.0f / 11.0f);
}

TEST_CASE("ProximityScorer Ordered vs Unordered", "[ut][ProximityScorer]") {
    // Same positions, ordered should give lower boost due to penalty
    std::vector<std::vector<uint16_t>> positions = {{10}, {5}};
    float boost_unordered = compute_pairwise_proximity(positions, false);
    float boost_ordered = compute_pairwise_proximity(positions, true);
    // Unordered: dist=5, boost=1/6
    // Ordered: dist=10 (reverse penalty), boost=1/11
    REQUIRE(boost_unordered > boost_ordered);
    REQUIRE_APPROX(boost_unordered, 1.0f / 6.0f);
    REQUIRE_APPROX(boost_ordered, 1.0f / 11.0f);
}

TEST_CASE("ProximityScorer Multiple Positions Min Distance", "[ut][ProximityScorer]") {
    // term A at [10, 100], term B at [12, 95]
    // Unordered min_dist: min(|10-12|, |10-95|, |100-12|, |100-95|) = min(2,85,88,5) = 2
    // boost = 1/3
    std::vector<std::vector<uint16_t>> positions = {{10, 100}, {12, 95}};
    float boost = compute_pairwise_proximity(positions, false);
    REQUIRE_APPROX(boost, 1.0f / 3.0f);
}

TEST_CASE("ProximityScorer Ordered Multiple Positions Best Forward", "[ut][ProximityScorer]") {
    // ordered=true, term A at [10, 100], term B at [12, 95]
    // Forward pairs (pos_A < pos_B): (10,12) dist=2, (10,95) dist=85, (100, -)  none forward
    // Reverse pairs (pos_A > pos_B): (100,12) dist=88*2=176, (100,95) dist=5*2=10
    // Min dist overall: 2 (forward pair 10→12)
    // boost = 1/3
    std::vector<std::vector<uint16_t>> positions = {{10, 100}, {12, 95}};
    float boost = compute_pairwise_proximity(positions, true);
    REQUIRE_APPROX(boost, 1.0f / 3.0f);
}

TEST_CASE("ProximityScorer Empty Position List Skipped", "[ut][ProximityScorer]") {
    // term A has positions, term B has empty (term not present in doc)
    // This pair contributes nothing
    std::vector<std::vector<uint16_t>> positions = {{5, 10}, {}};
    float boost = compute_pairwise_proximity(positions, false);
    REQUIRE(boost == 0.0f);
}

// ===== extract_positions_from_token_sequence tests =====

TEST_CASE("ExtractPositions Basic", "[ut][ProximityScorer]") {
    // Document: "苹果(10) 公司(20) 发布(30) 新款(40) 苹果(10) 手机(50) iphone(60)"
    uint32_t token_seq[] = {10, 20, 30, 40, 10, 50, 60};
    uint32_t ids[] = {10, 20, 60};
    std::vector<std::vector<uint16_t>> out;

    extract_positions_from_token_sequence(token_seq, 7, ids, 3, 64, out);

    REQUIRE(out.size() == 3);
    // term 10 (苹果): positions [0, 4]
    REQUIRE(out[0].size() == 2);
    REQUIRE(out[0][0] == 0);
    REQUIRE(out[0][1] == 4);
    // term 20 (公司): positions [1]
    REQUIRE(out[1].size() == 1);
    REQUIRE(out[1][0] == 1);
    // term 60 (iphone): positions [6]
    REQUIRE(out[2].size() == 1);
    REQUIRE(out[2][0] == 6);
}

TEST_CASE("ExtractPositions Ignores Non-ID Tokens", "[ut][ProximityScorer]") {
    // token_seq has terms 30, 40, 50 that are NOT in ids
    uint32_t token_seq[] = {30, 40, 50, 10, 30};
    uint32_t ids[] = {10};
    std::vector<std::vector<uint16_t>> out;

    extract_positions_from_token_sequence(token_seq, 5, ids, 1, 64, out);

    REQUIRE(out.size() == 1);
    REQUIRE(out[0].size() == 1);
    REQUIRE(out[0][0] == 3);  // term 10 at pos 3
}

TEST_CASE("ExtractPositions Cap Truncation", "[ut][ProximityScorer]") {
    // term 10 appears 10 times, but cap=4
    uint32_t token_seq[] = {10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
    uint32_t ids[] = {10};
    std::vector<std::vector<uint16_t>> out;

    extract_positions_from_token_sequence(token_seq, 10, ids, 1, 4, out);

    REQUIRE(out.size() == 1);
    REQUIRE(out[0].size() == 4);  // capped at 4
    REQUIRE(out[0][0] == 0);
    REQUIRE(out[0][1] == 1);
    REQUIRE(out[0][2] == 2);
    REQUIRE(out[0][3] == 3);
}

TEST_CASE("ExtractPositions Null Token Sequence", "[ut][ProximityScorer]") {
    uint32_t ids[] = {10, 20};
    std::vector<std::vector<uint16_t>> out;

    extract_positions_from_token_sequence(nullptr, 0, ids, 2, 64, out);

    REQUIRE(out.size() == 2);
    REQUIRE(out[0].empty());
    REQUIRE(out[1].empty());
}

TEST_CASE("ExtractPositions Empty IDs", "[ut][ProximityScorer]") {
    uint32_t token_seq[] = {10, 20, 30};
    std::vector<std::vector<uint16_t>> out;

    extract_positions_from_token_sequence(token_seq, 3, nullptr, 0, 64, out);

    REQUIRE(out.empty());
}

TEST_CASE("ExtractPositions Duplicate Term in IDs", "[ut][ProximityScorer]") {
    // Same term_id appears twice in ids (unusual but should handle)
    uint32_t token_seq[] = {10, 20, 10, 30};
    uint32_t ids[] = {10, 10};
    std::vector<std::vector<uint16_t>> out;

    extract_positions_from_token_sequence(token_seq, 4, ids, 2, 64, out);

    REQUIRE(out.size() == 2);
    // Both entries get the same positions
    REQUIRE(out[0].size() == 2);
    REQUIRE(out[0][0] == 0);
    REQUIRE(out[0][1] == 2);
    REQUIRE(out[1].size() == 2);
    REQUIRE(out[1][0] == 0);
    REQUIRE(out[1][1] == 2);
}

TEST_CASE("ExtractPositions Term Not In Sequence", "[ut][ProximityScorer]") {
    // ids contains a term that doesn't appear in token_seq at all
    uint32_t token_seq[] = {10, 20, 30};
    uint32_t ids[] = {10, 99};
    std::vector<std::vector<uint16_t>> out;

    extract_positions_from_token_sequence(token_seq, 3, ids, 2, 64, out);

    REQUIRE(out.size() == 2);
    REQUIRE(out[0].size() == 1);
    REQUIRE(out[0][0] == 0);
    REQUIRE(out[1].empty());  // term 99 not found
}
