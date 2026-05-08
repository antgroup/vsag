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

#include "simd_status.h"
#include "unittest.h"

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
                auto neon = neon::Normalize(vec1.data() + i * dim, tmp_value.data() + dim * 3, dim);
                REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(neon));
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(tmp_value[j]) ==
                            fixtures::dist_t(tmp_value[j + dim * 3]));
                }
            }
            if (SimdStatus::SupportSVE()) {
                auto sve = sve::Normalize(vec1.data() + i * dim, tmp_value.data() + dim * 3, dim);
                REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sve));
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(tmp_value[j]) ==
                            fixtures::dist_t(tmp_value[j + dim * 3]));
                }
            }
        }
    }
}

TEST_CASE("NormalizeWithCentroid Compute", "[ut][simd]") {
    auto dims = fixtures::get_common_used_dims();
    int64_t count = 100;
    for (auto& dim : dims) {
        auto vec1 = fixtures::generate_vectors(count, dim);
        auto centroid = fixtures::generate_vectors(1, dim);
        std::vector<float> tmp_value(dim * 7);
        for (uint64_t i = 0; i < count; ++i) {
            auto gt = generic::NormalizeWithCentroid(
                vec1.data() + i * dim, centroid.data(), tmp_value.data(), dim);

            if (SimdStatus::SupportSSE()) {
                auto sse = sse::NormalizeWithCentroid(
                    vec1.data() + i * dim, centroid.data(), tmp_value.data() + dim, dim);
                REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sse));
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(tmp_value[j]) ==
                            fixtures::dist_t(tmp_value[j + dim * 1]));
                }
            }
            if (SimdStatus::SupportAVX()) {
                auto avx = avx::NormalizeWithCentroid(
                    vec1.data() + i * dim, centroid.data(), tmp_value.data() + dim * 2, dim);
                REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx));
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(tmp_value[j]) ==
                            fixtures::dist_t(tmp_value[j + dim * 2]));
                }
            }
            if (SimdStatus::SupportAVX2()) {
                auto avx2 = avx2::NormalizeWithCentroid(
                    vec1.data() + i * dim, centroid.data(), tmp_value.data() + dim * 3, dim);
                REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx2));
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(tmp_value[j]) ==
                            fixtures::dist_t(tmp_value[j + dim * 3]));
                }
            }
            if (SimdStatus::SupportAVX512()) {
                auto avx512 = avx512::NormalizeWithCentroid(
                    vec1.data() + i * dim, centroid.data(), tmp_value.data() + dim * 4, dim);
                REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx512));
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(tmp_value[j]) ==
                            fixtures::dist_t(tmp_value[j + dim * 4]));
                }
            }
            if (SimdStatus::SupportNEON()) {
                auto neon = neon::NormalizeWithCentroid(
                    vec1.data() + i * dim, centroid.data(), tmp_value.data() + dim * 5, dim);
                REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(neon));
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(tmp_value[j]) ==
                            fixtures::dist_t(tmp_value[j + dim * 5]));
                }
            }
            if (SimdStatus::SupportSVE()) {
                auto sve = sve::NormalizeWithCentroid(
                    vec1.data() + i * dim, centroid.data(), tmp_value.data() + dim * 6, dim);
                REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sve));
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(tmp_value[j]) ==
                            fixtures::dist_t(tmp_value[j + dim * 6]));
                }
            }
        }
    }
}

TEST_CASE("NormalizeWithCentroid Self Centroid", "[ut][simd]") {
    auto dims = fixtures::get_common_used_dims();
    int64_t count = 10;
    for (auto& dim : dims) {
        auto vec1 = fixtures::generate_vectors(count, dim);
        std::vector<float> normalized(dim * 7);
        std::vector<float> recovered(dim * 7);
        for (uint64_t i = 0; i < count; ++i) {
            auto* input = vec1.data() + i * dim;

            auto gt = generic::NormalizeWithCentroid(input, input, normalized.data(), dim);
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(1.0f));
            for (int j = 0; j < dim; ++j) {
                REQUIRE(fixtures::dist_t(normalized[j]) == fixtures::dist_t(0.0f));
            }
            generic::InverseNormalizeWithCentroid(
                normalized.data(), input, recovered.data(), dim, gt);
            for (int j = 0; j < dim; ++j) {
                REQUIRE(fixtures::dist_t(recovered[j]) == fixtures::dist_t(input[j]));
            }

            if (SimdStatus::SupportSSE()) {
                auto sse = sse::NormalizeWithCentroid(input, input, normalized.data() + dim, dim);
                REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sse));
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(normalized[j + dim]) == fixtures::dist_t(0.0f));
                }
                sse::InverseNormalizeWithCentroid(
                    normalized.data() + dim, input, recovered.data() + dim, dim, sse);
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(recovered[j + dim]) == fixtures::dist_t(input[j]));
                }
            }
            if (SimdStatus::SupportAVX()) {
                auto avx =
                    avx::NormalizeWithCentroid(input, input, normalized.data() + dim * 2, dim);
                REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx));
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(normalized[j + dim * 2]) == fixtures::dist_t(0.0f));
                }
                avx::InverseNormalizeWithCentroid(
                    normalized.data() + dim * 2, input, recovered.data() + dim * 2, dim, avx);
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(recovered[j + dim * 2]) == fixtures::dist_t(input[j]));
                }
            }
            if (SimdStatus::SupportAVX2()) {
                auto avx2 =
                    avx2::NormalizeWithCentroid(input, input, normalized.data() + dim * 3, dim);
                REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx2));
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(normalized[j + dim * 3]) == fixtures::dist_t(0.0f));
                }
                avx2::InverseNormalizeWithCentroid(
                    normalized.data() + dim * 3, input, recovered.data() + dim * 3, dim, avx2);
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(recovered[j + dim * 3]) == fixtures::dist_t(input[j]));
                }
            }
            if (SimdStatus::SupportAVX512()) {
                auto avx512 =
                    avx512::NormalizeWithCentroid(input, input, normalized.data() + dim * 4, dim);
                REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx512));
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(normalized[j + dim * 4]) == fixtures::dist_t(0.0f));
                }
                avx512::InverseNormalizeWithCentroid(
                    normalized.data() + dim * 4, input, recovered.data() + dim * 4, dim, avx512);
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(recovered[j + dim * 4]) == fixtures::dist_t(input[j]));
                }
            }
            if (SimdStatus::SupportNEON()) {
                auto neon =
                    neon::NormalizeWithCentroid(input, input, normalized.data() + dim * 5, dim);
                REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(neon));
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(normalized[j + dim * 5]) == fixtures::dist_t(0.0f));
                }
                neon::InverseNormalizeWithCentroid(
                    normalized.data() + dim * 5, input, recovered.data() + dim * 5, dim, neon);
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(recovered[j + dim * 5]) == fixtures::dist_t(input[j]));
                }
            }
            if (SimdStatus::SupportSVE()) {
                auto sve =
                    sve::NormalizeWithCentroid(input, input, normalized.data() + dim * 6, dim);
                REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sve));
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(normalized[j + dim * 6]) == fixtures::dist_t(0.0f));
                }
                sve::InverseNormalizeWithCentroid(
                    normalized.data() + dim * 6, input, recovered.data() + dim * 6, dim, sve);
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(recovered[j + dim * 6]) == fixtures::dist_t(input[j]));
                }
            }
        }
    }
}

TEST_CASE("InverseNormalizeWithCentroid Compute", "[ut][simd]") {
    auto dims = fixtures::get_common_used_dims();
    int64_t count = 100;
    for (auto& dim : dims) {
        auto vec1 = fixtures::generate_vectors(count, dim);
        auto centroid = fixtures::generate_vectors(1, dim);
        std::vector<float> tmp_value(dim * 7);
        std::vector<float> recovered(dim * 7);
        for (uint64_t i = 0; i < count; ++i) {
            auto norm = generic::NormalizeWithCentroid(
                vec1.data() + i * dim, centroid.data(), tmp_value.data(), dim);

            generic::InverseNormalizeWithCentroid(
                tmp_value.data(), centroid.data(), recovered.data(), dim, norm);
            for (int j = 0; j < dim; ++j) {
                REQUIRE(fixtures::dist_t(vec1[i * dim + j]) == fixtures::dist_t(recovered[j]));
            }

            if (SimdStatus::SupportSSE()) {
                sse::InverseNormalizeWithCentroid(
                    tmp_value.data(), centroid.data(), recovered.data() + dim, dim, norm);
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(vec1[i * dim + j]) ==
                            fixtures::dist_t(recovered[j + dim * 1]));
                }
            }
            if (SimdStatus::SupportAVX()) {
                avx::InverseNormalizeWithCentroid(
                    tmp_value.data(), centroid.data(), recovered.data() + dim * 2, dim, norm);
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(vec1[i * dim + j]) ==
                            fixtures::dist_t(recovered[j + dim * 2]));
                }
            }
            if (SimdStatus::SupportAVX2()) {
                avx2::InverseNormalizeWithCentroid(
                    tmp_value.data(), centroid.data(), recovered.data() + dim * 3, dim, norm);
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(vec1[i * dim + j]) ==
                            fixtures::dist_t(recovered[j + dim * 3]));
                }
            }
            if (SimdStatus::SupportAVX512()) {
                avx512::InverseNormalizeWithCentroid(
                    tmp_value.data(), centroid.data(), recovered.data() + dim * 4, dim, norm);
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(vec1[i * dim + j]) ==
                            fixtures::dist_t(recovered[j + dim * 4]));
                }
            }
            if (SimdStatus::SupportNEON()) {
                neon::InverseNormalizeWithCentroid(
                    tmp_value.data(), centroid.data(), recovered.data() + dim * 5, dim, norm);
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(vec1[i * dim + j]) ==
                            fixtures::dist_t(recovered[j + dim * 5]));
                }
            }
            if (SimdStatus::SupportSVE()) {
                sve::InverseNormalizeWithCentroid(
                    tmp_value.data(), centroid.data(), recovered.data() + dim * 6, dim, norm);
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(fixtures::dist_t(vec1[i * dim + j]) ==
                            fixtures::dist_t(recovered[j + dim * 6]));
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
    if (SimdStatus::SupportSSE()) {
        BENCHMARK_SIMD_COMPUTE(sse, Normalize);
    }
    if (SimdStatus::SupportAVX2()) {
        BENCHMARK_SIMD_COMPUTE(avx2, Normalize);
    }
    if (SimdStatus::SupportAVX512()) {
        BENCHMARK_SIMD_COMPUTE(avx512, Normalize);
    }
    if (SimdStatus::SupportNEON()) {
        BENCHMARK_SIMD_COMPUTE(neon, Normalize);
    }
    if (SimdStatus::SupportSVE()) {
        BENCHMARK_SIMD_COMPUTE(sve, Normalize);
    }
}
