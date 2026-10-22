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

#include "distance_early_stop.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "unittest.h"
#include "vsag_exception.h"

using namespace vsag;

namespace {

InnerSearchParam
make_early_stop_param(int64_t k) {
    InnerSearchParam param;
    param.rerank_topk = k;
    return param;
}

QueryContext
make_early_stop_context(float alpha) {
    QueryContext ctx;
    ctx.distance_early_stop_alpha = alpha;
    return ctx;
}

// alpha * max(0, k-th smallest recorded finite score), compared once k finite scores exist.
bool
reference_should_stop(std::vector<float> finite_scores, uint64_t k, float alpha, float next) {
    if (finite_scores.size() < k or not is_finite_distance(next)) {
        return false;
    }
    std::nth_element(finite_scores.begin(),
                     finite_scores.begin() + static_cast<int64_t>(k - 1),
                     finite_scores.end());
    return static_cast<double>(next) >=
           static_cast<double>(alpha) * std::max(0.0F, finite_scores[k - 1]);
}

}  // namespace

TEST_CASE("Distance early stop matches a sorted reference", "[ut][distance_early_stop]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const std::vector<float> special{std::numeric_limits<float>::infinity(),
                                     -std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::quiet_NaN(),
                                     0.0F,
                                     -0.0F};
    std::mt19937 rng(20261012);
    // Scores come from a small grid, so ties at the k-th position are frequent; negative scores
    // exercise the max(0, k-th) clamp.
    std::uniform_int_distribution<int> grid(-4, 40);
    std::uniform_int_distribution<int> pick(0, 9);
    std::uniform_int_distribution<uint64_t> special_index(0, 4);
    auto draw = [&]() {
        if (pick(rng) == 0) {
            return special[special_index(rng)];
        }
        return static_cast<float>(grid(rng)) * 0.25F;
    };
    for (const uint64_t k : {1, 2, 10}) {
        for (const float alpha : {1.0F, 1.1F, 1.5F}) {
            CAPTURE(k, alpha);
            const auto ctx = make_early_stop_context(alpha);
            DistanceEarlyStop<true> early_stop(
                allocator.get(), make_early_stop_param(static_cast<int64_t>(k)), &ctx);
            std::vector<float> finite_scores;
            for (uint64_t step = 0; step < 400; ++step) {
                const float score = draw();
                early_stop.Record(score);
                if (is_finite_distance(score)) {
                    finite_scores.push_back(score);
                }
                const float next = draw();
                REQUIRE(early_stop.ShouldStop(next) ==
                        reference_should_stop(finite_scores, k, alpha, next));
            }
        }
    }
}

TEST_CASE("Distance early stop waits for k results and keeps ties", "[ut][distance_early_stop]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const auto ctx = make_early_stop_context(1.5F);
    DistanceEarlyStop<true> early_stop(allocator.get(), make_early_stop_param(2), &ctx);
    early_stop.Record(1.0F);
    REQUIRE_FALSE(early_stop.ShouldStop(100.0F));
    early_stop.Record(std::numeric_limits<float>::quiet_NaN());
    early_stop.Record(std::numeric_limits<float>::infinity());
    REQUIRE_FALSE(early_stop.ShouldStop(100.0F));
    early_stop.Record(2.0F);
    REQUIRE(early_stop.ShouldStop(3.0F));
    REQUIRE_FALSE(early_stop.ShouldStop(2.9F));
    REQUIRE_FALSE(early_stop.ShouldStop(std::numeric_limits<float>::infinity()));
    // An equal score does not replace the k-th result; a smaller one tightens the threshold.
    early_stop.Record(2.0F);
    REQUIRE_FALSE(early_stop.ShouldStop(2.9F));
    early_stop.Record(1.0F);
    REQUIRE(early_stop.ShouldStop(1.5F));
    // A zero k-th score makes every finite non-negative distance a stop.
    early_stop.Record(0.0F);
    early_stop.Record(0.0F);
    REQUIRE(early_stop.ShouldStop(0.0F));
}

TEST_CASE("Distance early stop rejects traversals it cannot serve", "[ut][distance_early_stop]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const auto ctx = make_early_stop_context(1.5F);
    REQUIRE_THROWS_AS(DistanceEarlyStop<true>(allocator.get(), make_early_stop_param(0), &ctx),
                      VsagException);
    auto custom = make_early_stop_param(2);
    custom.distance_batch_func = [](const int64_t* /*ids*/, uint64_t /*count*/, float* /*out*/) {};
    REQUIRE_THROWS_AS(DistanceEarlyStop<true>(allocator.get(), custom, &ctx), VsagException);
    // The alpha comes from the query context and must be at least 1.
    REQUIRE_THROWS_AS(DistanceEarlyStop<true>(allocator.get(), make_early_stop_param(2), nullptr),
                      VsagException);
    const auto no_alpha = make_early_stop_context(0.0F);
    REQUIRE_THROWS_AS(DistanceEarlyStop<true>(allocator.get(), make_early_stop_param(2), &no_alpha),
                      VsagException);
}
