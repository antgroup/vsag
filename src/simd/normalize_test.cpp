
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

#include "normalize.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "simd_status.h"

using namespace vsag;

TEST_CASE("Normalize Compute", "[ut][simd]") {
    auto dims = fixtures::get_common_used_dims();
    int64_t count = 100;
    for (auto& dim : dims) {
        auto vec1 = fixtures::generate_vectors(count, dim);
        std::vector<float> tmp_value(dim * 4);
        std::vector<float> zero_centroid(dim, 0);
        for (uint64_t i = 0; i < count; ++i) {
            auto gt_self_centroid = generic::NormalizeWithCentroid(
                vec1.data() + i * dim, vec1.data() + i * dim, tmp_value.data(), dim);
            REQUIRE(std::abs(gt_self_centroid - 1) < 1e-5);
            auto gt_zero_centroid = generic::NormalizeWithCentroid(
                vec1.data() + i * dim, zero_centroid.data(), tmp_value.data(), dim);
            auto gt = generic::Normalize(vec1.data() + i * dim, tmp_value.data(), dim);
            REQUIRE(gt_zero_centroid == gt);

            if (SimdStatus::SupportSSE()) {
                auto sse = sse::Normalize(vec1.data() + i * dim, tmp_value.data() + dim, dim);
                REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sse));
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(tmp_value[j]) ==
                            fixtures::dist_t(tmp_value[j + dim * 1]));
                }
            }
            if (SimdStatus::SupportAVX2()) {
                auto avx2 = avx2::Normalize(vec1.data() + i * dim, tmp_value.data() + dim * 2, dim);
                REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx2));
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(tmp_value[j]) ==
                            fixtures::dist_t(tmp_value[j + dim * 2]));
                }
            }
            if (SimdStatus::SupportAVX512()) {
                auto avx512 =
                    avx512::Normalize(vec1.data() + i * dim, tmp_value.data() + dim * 3, dim);
                REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx512));
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(tmp_value[j]) ==
                            fixtures::dist_t(tmp_value[j + dim * 3]));
                }
            }
            if (SimdStatus::SupportNEON()) {
                auto neon =
                    neon::Normalize(vec1.data() + i * dim, tmp_value.data() + dim * 3, dim);
                REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(neon));
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(tmp_value[j]) ==
                            fixtures::dist_t(tmp_value[j + dim * 3]));
                }
            }
        }
    }
}

#define BENCHMARK_SIMD_COMPUTE(Simd, Comp)                                 \
    BENCHMARK_ADVANCED(#Simd #Comp) {                                      \
        for (int i = 0; i < count; ++i) {                                  \
            Simd::Comp(vec1.data() + i * dim, vec2.data() + i * dim, dim); \
        }                                                                  \
        return;                                                            \
    }

TEST_CASE("Normalize Benchmark", "[ut][simd][!benchmark]") {
    int64_t count = 500;
    int64_t dim = 128;
    auto vec1 = fixtures::generate_vectors(count * 2, dim);
    std::vector<float> vec2(vec1.begin() + count, vec1.end());
    BENCHMARK_SIMD_COMPUTE(generic, Normalize);
    BENCHMARK_SIMD_COMPUTE(sse, Normalize);
    BENCHMARK_SIMD_COMPUTE(avx2, Normalize);
    BENCHMARK_SIMD_COMPUTE(avx512, Normalize);
    BENCHMARK_SIMD_COMPUTE(neon, Normalize);
}
