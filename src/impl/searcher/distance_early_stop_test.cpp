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

using namespace vsag;

namespace {

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
            DistanceEarlyStop early_stop(allocator.get(), static_cast<int64_t>(k), alpha);
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
