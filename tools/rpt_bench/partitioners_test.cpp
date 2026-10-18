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

#include "partitioners.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <string>
#include <vector>

namespace vsag::rpt_bench {

TEST_CASE("KMeans benchmark rejects unsupported cluster counts", "[rpt_bench][kmeans]") {
    const float data = 0.0F;
    const auto max_clusters = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    const std::array<uint64_t, 3> counts = {
        max_clusters + 1,
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1,
        std::numeric_limits<uint64_t>::max()};
    for (uint64_t count : counts) {
        CAPTURE(count);
        // Invalid requests must return before reading data or allocating cluster storage.
        PartitionRequest request{1, count, &data, 1, 0};
        auto result = partition_by_kmeans(request);
        REQUIRE(result.error == "cluster count " + std::to_string(count) +
                                    " exceeds KMeans maximum " + std::to_string(max_clusters));
        REQUIRE(result.partitions.empty());
    }
}

TEST_CASE("KMeans benchmark retains every vector for supported requests", "[rpt_bench][kmeans]") {
    const std::array<float, 4> data = {0.0F, 1.0F, 10.0F, 11.0F};
    constexpr uint64_t count = 4;
    PartitionRequest request{1, count, data.data(), 2, 0};
    auto result = partition_by_kmeans(request);
    REQUIRE(result.error.empty());
    REQUIRE(result.partitions.size() == 2);

    std::vector<int64_t> ids;
    for (const auto& partition : result.partitions) {
        REQUIRE_FALSE(partition.empty());
        ids.insert(ids.end(), partition.begin(), partition.end());
    }
    std::sort(ids.begin(), ids.end());
    REQUIRE(ids == std::vector<int64_t>{0, 1, 2, 3});
}

}  // namespace vsag::rpt_bench
