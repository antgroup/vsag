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

#include "pipnn_adaptive_pruning.h"

#include <array>
#include <catch2/catch_approx.hpp>
#include <limits>

#include "impl/allocator/safe_allocator.h"
#include "unittest.h"

namespace vsag {

TEST_CASE("PiPNN adaptive relaxation follows PR2933 count boundaries", "[ut][pipnn_adaptive]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    for (uint64_t initial : {1, 2, 3, 4, 5}) {
        CAPTURE(initial);
        Vector<std::pair<float, uint32_t>> candidates(allocator.get());
        for (uint32_t id = 1; id <= 7; ++id) {
            candidates.emplace_back(1.0F, id);
        }
        PiPNNAdaptivePruningStats stats;
        const auto selected = select_pipnn_edges_adaptive(
            candidates,
            6,
            1.06F,
            0.06F,
            [initial](uint32_t, uint32_t rhs) { return rhs <= initial ? 10.0F : 0.0F; },
            allocator.get(),
            &stats);
        CHECK(selected.size() == initial);
        CHECK(stats.initial_accepted == initial);
        CHECK(stats.initial_rejected == 7 - initial);
        CHECK(stats.branch == PiPNNAdaptivePruningBranch::RELAX);
        const float expected[] = {0, 1.24F, 1.18F, 1.18F, 1.12F, 1.12F};
        CHECK(stats.second_alpha == Catch::Approx(expected[initial]));
    }
}

TEST_CASE("PiPNN adaptive tightening counts only scanned rejects", "[ut][pipnn_adaptive]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    for (uint32_t rejected : {0, 4, 5, 9, 10}) {
        CAPTURE(rejected);
        Vector<std::pair<float, uint32_t>> candidates(allocator.get());
        for (uint32_t id = 1; id <= rejected + 5; ++id) {
            candidates.emplace_back(1.0F, id);
        }
        PiPNNAdaptivePruningStats stats;
        const auto selected = select_pipnn_edges_adaptive(
            candidates,
            2,
            1.06F,
            0.06F,
            [rejected](uint32_t, uint32_t rhs) { return rhs == rejected + 2 ? 10.0F : 0.0F; },
            allocator.get(),
            &stats);
        CHECK(selected.size() == 2);
        CHECK(stats.initial_rejected == rejected);
        CHECK(stats.branch == PiPNNAdaptivePruningBranch::TIGHTEN);
        CHECK(stats.second_alpha ==
              Catch::Approx(rejected >= 10 ? 1.06F : (rejected >= 5 ? 1.0F : 0.94F)));
    }
}

TEST_CASE("PiPNN adaptive revisits rejects and original tail", "[ut][pipnn_adaptive]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    Vector<std::pair<float, uint32_t>> candidates(allocator.get());
    PiPNNAdaptivePruningStats stats;
    SECTION("relaxation can admit a rejected neighbor") {
        candidates = {{1.0F, 1}, {1.1F, 2}};
        const auto result = select_pipnn_edges_adaptive(
            candidates,
            2,
            1.0F,
            0.06F,
            [](uint32_t, uint32_t) { return 1.0F; },
            allocator.get(),
            &stats);
        REQUIRE(result.size() == 2);
        CHECK(stats.branch == PiPNNAdaptivePruningBranch::RELAX);
    }
    SECTION("tightening admits an unscanned tail candidate with real squared L2") {
        // u=(0,0), a=(1,0), b=(0.5,1), c=(-2,0); the first pass stops at a,b.
        candidates = {{1.0F, 9}, {1.25F, 2}, {4.0F, 7}};
        const auto result = select_pipnn_edges_adaptive(
            candidates,
            2,
            1.06F,
            0.06F,
            [](uint32_t lhs, uint32_t rhs) {
                if (lhs == 9 and rhs == 2) {
                    return 1.25F;
                }
                if (lhs == 9 and rhs == 7) {
                    return 9.0F;
                }
                return 7.25F;
            },
            allocator.get(),
            &stats);
        REQUIRE(result.size() == 2);
        CHECK(result[0] == 0);
        CHECK(result[1] == 2);
        CHECK(stats.initial_rejected == 0);
    }
    SECTION("fallback uses tightened rejects even when first pass rejected nothing") {
        candidates = {{1.0F, 1}, {2.0F, 2}};
        const auto result = select_pipnn_edges_adaptive(
            candidates,
            2,
            1.06F,
            0.06F,
            [](uint32_t, uint32_t) { return 2.0F; },
            allocator.get(),
            &stats);
        REQUIRE(result.size() == 2);
        CHECK(result[1] == 1);
        CHECK(stats.initial_rejected == 0);
        CHECK(stats.distance_calls == 3);
    }
}

TEST_CASE("PiPNN adaptive relaxation preserves occlusion and degree limits",
          "[ut][pipnn_adaptive]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    // With source u=(0,0), b and c are both occluded by a at alpha=1.2.
    std::array<std::array<float, 2>, 3> points{{{1.0F, 0.0F}, {0.6F, 0.8F}, {0.6F, 0.81F}}};
    uint64_t target_degree = 3;
    SECTION("newly accepted neighbors still occlude later candidates") {
        // Relaxation admits b, which must reject the nearby c.
    }
    SECTION("relaxation stops at the degree cap") {
        // c would survive comparison with both a and b, but there is no slot left.
        points[2] = {0.6F, -0.8F};
        target_degree = 2;
    }
    Vector<std::pair<float, uint32_t>> candidates(allocator.get());
    for (uint32_t id = 0; id < points.size(); ++id) {
        const auto& point = points[id];
        candidates.emplace_back(point[0] * point[0] + point[1] * point[1], id);
    }
    const auto distance = [&](uint32_t lhs, uint32_t rhs) {
        const float dx = points[lhs][0] - points[rhs][0];
        const float dy = points[lhs][1] - points[rhs][1];
        return dx * dx + dy * dy;
    };
    PiPNNAdaptivePruningStats stats;
    const auto selected = select_pipnn_edges_adaptive(
        candidates, target_degree, 1.2F, 0.06F, distance, allocator.get(), &stats);
    CHECK(stats.branch == PiPNNAdaptivePruningBranch::RELAX);
    CHECK(stats.initial_accepted == 1);
    CHECK(stats.initial_rejected == 2);
    CHECK(stats.second_alpha == Catch::Approx(1.32F));
    REQUIRE(selected.size() == 2);
    CHECK(selected[0] == 0);
    CHECK(selected[1] == 1);
}

TEST_CASE("PiPNN adaptive preserves caller order and never fills rejects", "[ut][pipnn_adaptive]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    Vector<std::pair<float, uint32_t>> candidates(allocator.get());
    candidates = {{1.0F, 9}, {1.0F, 2}, {2.0F, 3}};
    const float step = GENERATE(0.0F, 0.06F);
    auto result = select_pipnn_edges_adaptive(
        candidates, 8, 1.06F, step, [](uint32_t, uint32_t) { return 0.0F; }, allocator.get());
    REQUIRE(result.size() == 1);
    CHECK(result[0] == 0);
    // Equal distances retain the existing PiPNN ID-gap order, not ascending candidate ID.
    result = select_pipnn_edges_adaptive(
        candidates, 3, 1.0F, 0.0F, [](uint32_t, uint32_t) { return 2.0F; }, allocator.get());
    REQUIRE(result.size() == 3);
    CHECK(result[0] == 0);
    CHECK(result[1] == 1);
    CHECK(result[2] == 2);
}

TEST_CASE("PiPNN adaptive strict comparison and empty input", "[ut][pipnn_adaptive]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    Vector<std::pair<float, uint32_t>> candidates(allocator.get());
    candidates = {{1.0F, 1}, {1.0F, 2}};
    const auto distance = [](uint32_t, uint32_t) { return 1.0F; };
    PiPNNAdaptivePruningStats stats;
    CHECK(select_pipnn_edges_adaptive(candidates, 2, 1, 0, distance, allocator.get(), &stats)
              .size() == 2);
    CHECK(stats.branch == PiPNNAdaptivePruningBranch::SINGLE);
    CHECK(stats.distance_calls == 1);
    CHECK(select_pipnn_edges_adaptive(candidates, 0, 1, 0.06F, distance, allocator.get(), &stats)
              .empty());
    CHECK(stats.distance_calls == 0);
    candidates.clear();
    CHECK(select_pipnn_edges_adaptive(candidates, 2, 1, 0.06F, distance, allocator.get()).empty());
}

TEST_CASE("PiPNN adaptive rejects invalid thresholds and distances", "[ut][pipnn_adaptive]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    for (const float bad :
         {-1.0F, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        CHECK_THROWS(validate_pipnn_adaptive_pruning(1.06F, bad));
        CHECK_THROWS(validate_pipnn_adaptive_pruning(bad, 0.06F));
        Vector<std::pair<float, uint32_t>> candidates(allocator.get());
        candidates = {{bad, 1}};
        CHECK_THROWS(select_pipnn_edges_adaptive(
            candidates, 2, 1.06F, 0.06F, [](uint32_t, uint32_t) { return 1.0F; }, allocator.get()));
        candidates = {{1.0F, 1}, {2.0F, 2}};
        CHECK_THROWS(select_pipnn_edges_adaptive(
            candidates,
            2,
            1.06F,
            0.06F,
            [bad](uint32_t, uint32_t) { return bad; },
            allocator.get()));
    }
    CHECK_THROWS(validate_pipnn_adaptive_pruning(1.06F, 0.53F));
    CHECK_THROWS(validate_pipnn_adaptive_pruning(std::numeric_limits<float>::max(),
                                                 std::numeric_limits<float>::max() / 4));
}

}  // namespace vsag
