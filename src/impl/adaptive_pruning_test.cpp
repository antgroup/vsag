
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

#include "adaptive_pruning.h"

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <limits>

#include "unittest.h"
#include "vsag/engine.h"

namespace vsag {

TEST_CASE("Adaptive pruning relaxation boundaries", "[ut][adaptive_pruning]") {
    auto allocator = Engine::CreateDefaultAllocator();
    AdaptivePruningParameter p;
    p.enabled = true;
    for (uint64_t initial : {1, 2, 3, 4, 5}) {
        CAPTURE(initial);
        Vector<PruningCandidate> candidates(allocator.get());
        for (InnerIdType id = 1; id <= 7; ++id) {
            candidates.emplace_back(1.0F, id);
        }
        AdaptivePruningStats stats;
        auto selected = select_edges_adaptive(
            candidates,
            0,
            6,
            1.06F,
            p,
            [initial](InnerIdType, InnerIdType right) { return right <= initial ? 10.0F : 0.0F; },
            allocator.get(),
            &stats);
        CHECK(selected.size() == initial);
        CHECK(stats.initial_accepted == initial);
        CHECK(stats.initial_rejected == 7 - initial);
        CHECK(stats.branch == AdaptivePruningBranch::RELAX);
        const float expected[] = {0, 1.24F, 1.18F, 1.18F, 1.12F, 1.12F};
        CHECK(stats.second_alpha == Catch::Approx(expected[initial]));
    }
}

TEST_CASE("Adaptive pruning tightening boundaries", "[ut][adaptive_pruning]") {
    auto allocator = Engine::CreateDefaultAllocator();
    AdaptivePruningParameter p;
    p.enabled = true;
    for (uint64_t rejected : {0, 4, 5, 9, 10}) {
        CAPTURE(rejected);
        Vector<PruningCandidate> candidates(allocator.get());
        for (InnerIdType id = 1; id <= rejected + 2; ++id) {
            candidates.emplace_back(1.0F, id);
        }
        AdaptivePruningStats stats;
        auto selected = select_edges_adaptive(
            candidates,
            0,
            2,
            1.06F,
            p,
            [rejected](InnerIdType, InnerIdType right) {
                return right == rejected + 2 ? 10.0F : 0.0F;
            },
            allocator.get(),
            &stats);
        CHECK(selected.size() == 2);
        CHECK(stats.initial_rejected == rejected);
        CHECK(stats.branch == AdaptivePruningBranch::TIGHTEN);
        const float expected = rejected >= 10 ? 1.06F : (rejected >= 5 ? 1.0F : 0.94F);
        CHECK(stats.second_alpha == Catch::Approx(expected));
    }
}

TEST_CASE("Adaptive pruning revisits rejects and preserves tail", "[ut][adaptive_pruning]") {
    auto allocator = Engine::CreateDefaultAllocator();
    AdaptivePruningParameter p;
    p.enabled = true;
    AdaptivePruningStats stats;
    Vector<PruningCandidate> candidates(allocator.get());
    SECTION("Relaxation admits a previously rejected candidate") {
        candidates = {{1.0F, 1}, {1.1F, 2}};
        auto result = select_edges_adaptive(
            candidates,
            0,
            2,
            1.0F,
            p,
            [](InnerIdType, InnerIdType) { return 1.0F; },
            allocator.get(),
            &stats);
        REQUIRE(result.size() == 2);
        CHECK(stats.branch == AdaptivePruningBranch::RELAX);
        CHECK(stats.filled == 0);
    }
    SECTION("Tightening considers previously unscanned candidates") {
        candidates = {{1.0F, 1}, {2.0F, 2}, {3.0F, 3}};
        auto result = select_edges_adaptive(
            candidates,
            0,
            2,
            1.06F,
            p,
            [](InnerIdType, InnerIdType right) { return right == 2 ? 2.0F : 4.0F; },
            allocator.get(),
            &stats);
        REQUIRE(result.size() == 2);
        CHECK(result[0].second == 1);
        CHECK(result[1].second == 3);
    }
    SECTION("Tightening falls back to baseline without unconditional fill") {
        candidates = {{1.0F, 1}, {2.0F, 2}};
        auto result = select_edges_adaptive(
            candidates,
            0,
            2,
            1.06F,
            p,
            [](InnerIdType, InnerIdType) { return 2.0F; },
            allocator.get(),
            &stats);
        REQUIRE(result.size() == 2);
        CHECK(result[1].second == 2);
        CHECK(stats.filled == 0);
    }
}

TEST_CASE("Adaptive pruning normalization and optional fill", "[ut][adaptive_pruning]") {
    auto allocator = Engine::CreateDefaultAllocator();
    AdaptivePruningParameter p;
    p.enabled = true;
    const bool fill = GENERATE(false, true);
    p.fill_rejected = fill;
    p.adjust_step = GENERATE(0.0F, 0.06F);
    Vector<PruningCandidate> candidates(allocator.get());
    candidates = {{2.0F, 3}, {1.0F, 2}, {0.0F, 0}, {1.0F, 1}, {3.0F, 1}};
    AdaptivePruningStats stats;
    auto result = select_edges_adaptive(
        candidates,
        0,
        8,
        1.06F,
        p,
        [](InnerIdType, InnerIdType) { return 0.0F; },
        allocator.get(),
        &stats);
    REQUIRE(result.size() == (fill ? 3 : 1));
    CHECK(result[0].second == 1);
    CHECK(stats.filled == (fill ? 2 : 0));
    CHECK(stats.second_alpha == Catch::Approx(p.adjust_step == 0 ? 1.06F : 1.24F));
    if (fill) {
        CHECK(result[1].second == 2);
        CHECK(result[2].second == 3);
    }
}

TEST_CASE("Adaptive pruning strict inequality and empty inputs", "[ut][adaptive_pruning]") {
    auto allocator = Engine::CreateDefaultAllocator();
    AdaptivePruningParameter p;
    p.enabled = true;
    p.adjust_step = 0;
    Vector<PruningCandidate> candidates(allocator.get());
    candidates = {{1.0F, 1}, {1.0F, 2}};
    const auto distance = [](InnerIdType, InnerIdType) { return 1.0F; };
    AdaptivePruningStats stats;
    CHECK(select_edges_adaptive(candidates, 0, 2, 1, p, distance, allocator.get(), &stats).size() ==
          2);
    CHECK(stats.distance_calls == 1);
    CHECK(select_edges_adaptive(candidates, 0, 0, 1, p, distance, allocator.get()).empty());
    candidates.clear();
    CHECK(select_edges_adaptive(candidates, 0, 2, 1, p, distance, allocator.get()).empty());
    candidates.emplace_back(0.0F, 0);
    CHECK(select_edges_adaptive(candidates, 0, 2, 1, p, distance, allocator.get()).empty());
}

TEST_CASE("Adaptive pruning rejects invalid numeric inputs", "[ut][adaptive_pruning]") {
    auto allocator = Engine::CreateDefaultAllocator();
    AdaptivePruningParameter p;
    p.enabled = true;
    for (float bad :
         {-1.0F, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        CAPTURE(bad);
        p.adjust_step = bad;
        CHECK_THROWS(p.Validate(1.06F));
        p.adjust_step = 0.06F;
        CHECK_THROWS(p.Validate(bad));
        Vector<PruningCandidate> candidates(allocator.get());
        candidates = {{bad, 1}};
        CHECK_THROWS(select_edges_adaptive(
            candidates,
            0,
            2,
            1.06F,
            p,
            [](InnerIdType, InnerIdType) { return 1.0F; },
            allocator.get()));
        candidates = {{1.0F, 1}, {2.0F, 2}};
        CHECK_THROWS(select_edges_adaptive(
            candidates,
            0,
            2,
            1.06F,
            p,
            [bad](InnerIdType, InnerIdType) { return bad; },
            allocator.get()));
    }
    p.adjust_step = 0.53F;
    CHECK_THROWS(p.Validate(1.06F));
    p.adjust_step = std::numeric_limits<float>::max() / 4;
    CHECK_THROWS(p.Validate(std::numeric_limits<float>::max()));
}

TEST_CASE("Adaptive pruning parameter roundtrip", "[ut][adaptive_pruning]") {
    AdaptivePruningParameter p;
    p.FromJson(JsonType::Parse(R"({"enabled":true,"adjust_step":0.04,"fill_rejected":true})"));
    CHECK(p.enabled);
    CHECK(p.fill_rejected);
    CHECK(p.adjust_step == Catch::Approx(0.04F));
    CHECK_NOTHROW(p.Validate(1.06F));
    AdaptivePruningParameter restored;
    restored.FromJson(p.ToJson());
    CHECK(restored.ToJson().Dump() == p.ToJson().Dump());
    restored.apply_to_reverse = true;
    CHECK_NOTHROW(restored.Validate(1.06F));
    restored.apply_to_reverse = false;
    restored.apply_to_upper = true;
    CHECK_THROWS(restored.Validate(1.06F));
    p.FromJson(JsonType::Parse("{}"));
    CHECK_FALSE(p.enabled);
    CHECK_FALSE(p.fill_rejected);
    CHECK_NOTHROW(p.Validate(-1));
    CHECK_THROWS(p.FromJson(JsonType::Parse("[]")));
}

TEST_CASE("Adaptive pruning tightens real L2 neighbors independently of input order",
          "[ut][adaptive_pruning]") {
    auto allocator = Engine::CreateDefaultAllocator();
    AdaptivePruningParameter policy;
    policy.enabled = true;
    policy.fill_rejected = GENERATE(false, true);
    // Squared L2: d(0,1)=1, d(0,2)=d(1,2)=1.25, d(0,3)=4, d(1,3)=9.
    // The first pass takes {1,2}; tightening rejects 2 and must inspect the tail for 3.
    const float vectors[][2] = {{0, 0}, {1, 0}, {0.5F, 1}, {-2, 0}};
    const auto distance = [&](InnerIdType left, InnerIdType right) {
        const float dx = vectors[left][0] - vectors[right][0];
        const float dy = vectors[left][1] - vectors[right][1];
        return dx * dx + dy * dy;
    };
    Vector<InnerIdType> order(allocator.get());
    order = {1, 2, 3};
    do {
        Vector<PruningCandidate> candidates(allocator.get());
        for (auto id : order) {
            candidates.emplace_back(distance(0, id), id);
        }
        candidates.emplace_back(0, 0);               // A self edge must not consume a slot.
        candidates.emplace_back(distance(0, 2), 2);  // Nor may a duplicate candidate.
        AdaptivePruningStats stats;
        const auto selected = select_edges_adaptive(
            candidates, 0, 2, 1.06F, policy, distance, allocator.get(), &stats);
        REQUIRE(selected.size() == 2);
        CHECK(selected[0].second == 1);
        CHECK(selected[1].second == 3);
        CHECK(stats.branch == AdaptivePruningBranch::TIGHTEN);
        CHECK(stats.second_alpha == Catch::Approx(0.94F));
        CHECK(stats.filled == 0);
    } while (std::next_permutation(order.begin(), order.end()));
}

TEST_CASE("Adaptive pruning handles singleton degree and coincident vectors",
          "[ut][adaptive_pruning]") {
    auto allocator = Engine::CreateDefaultAllocator();
    AdaptivePruningParameter policy;
    policy.enabled = true;
    const uint64_t degree = GENERATE(1, 3);
    Vector<PruningCandidate> candidates(allocator.get());
    candidates = {{0, 3}, {0, 1}, {0, 2}, {0, 0}};
    AdaptivePruningStats stats;
    const auto selected = select_edges_adaptive(
        candidates,
        0,
        degree,
        1.06F,
        policy,
        [](InnerIdType, InnerIdType) { return 0.0F; },
        allocator.get(),
        &stats);
    REQUIRE(selected.size() == degree);
    for (uint64_t i = 0; i < degree; ++i) {
        CHECK(selected[i].second == i + 1);
    }
    CHECK(stats.filled == 0);
    CHECK(stats.distance_calls == degree * (degree - 1));
    policy.enabled = false;
    CHECK_THROWS(select_edges_adaptive(
        candidates,
        0,
        degree,
        1.06F,
        policy,
        [](InnerIdType, InnerIdType) { return 0.0F; },
        allocator.get()));
}

}  // namespace vsag
