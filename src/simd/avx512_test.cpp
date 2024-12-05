
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

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <iostream>

#include "catch2/catch_approx.hpp"
#include "cpuinfo.h"
#include "fixtures.h"
#include "simd.h"
#include "simd_status.h"

TEST_CASE("avx512 int8", "[ut][simd][avx]") {
#if defined(ENABLE_AVX512)
    if (vsag::SimdStatus::SupportAVX512()) {
        auto common_dims = fixtures::get_common_used_dims();
        for (size_t dim : common_dims) {
            auto vectors = fixtures::generate_vectors(2, dim);
            fixtures::dist_t distance_512 = vsag::INT8InnerProduct512ResidualsAVX512Distance(
                vectors.data(), vectors.data() + dim, &dim);
            fixtures::dist_t distance_256 = vsag::INT8InnerProduct256ResidualsAVX512Distance(
                vectors.data(), vectors.data() + dim, &dim);
            fixtures::dist_t expected_distance =
                vsag::INT8InnerProductDistance(vectors.data(), vectors.data() + dim, &dim);
            REQUIRE(distance_512 == expected_distance);
            REQUIRE(distance_256 == expected_distance);
        }
    }
#endif
}
