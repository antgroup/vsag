
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

#include "basic_func.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "simd_status.h"

using namespace vsag;

#define TEST_ACCURACY(Func)                                                            \
    {                                                                                  \
        float gt, sse, avx, avx2, avx512, neon;                                        \
        gt = generic::Func(vec1.data() + i * dim, vec2.data() + i * dim, &dim);        \
        if (SimdStatus::SupportSSE()) {                                                \
            sse = sse::Func(vec1.data() + i * dim, vec2.data() + i * dim, &dim);       \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sse));                    \
        }                                                                              \
        if (SimdStatus::SupportAVX()) {                                                \
            avx = avx::Func(vec1.data() + i * dim, vec2.data() + i * dim, &dim);       \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx));                    \
        }                                                                              \
        if (SimdStatus::SupportAVX2()) {                                               \
            avx2 = avx2::Func(vec1.data() + i * dim, vec2.data() + i * dim, &dim);     \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx2));                   \
        }                                                                              \
        if (SimdStatus::SupportAVX512()) {                                             \
            avx512 = avx512::Func(vec1.data() + i * dim, vec2.data() + i * dim, &dim); \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx512));                 \
        }                                                                              \
        if (SimdStatus::SupportNEON()) {                                               \
            neon = neon::Func(vec1.data() + i * dim, vec2.data() + i * dim, &dim);     \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(neon));                   \
        }                                                                              \
    };

TEST_CASE("L2Sqr & InnerProduct SIMD Compute", "[ut][simd]") {
    auto dims = fixtures::get_common_used_dims(8, 217);
    int64_t count = 100;
    for (const auto& dim2 : dims) {
        uint64_t dim = dim2;
        auto vec1 = fixtures::generate_vectors(count * 2, dim);
        std::vector<float> vec2(vec1.begin() + count, vec1.end());
        for (uint64_t i = 0; i < count; ++i) {
            TEST_ACCURACY(L2Sqr);
            TEST_ACCURACY(InnerProduct);
            TEST_ACCURACY(InnerProductDistance);
        }
    }
}

TEST_CASE("Int8 SIMD Compute", "[ut][simd]") {
    auto dims = fixtures::get_common_used_dims(8, 217);
    int64_t count = 100;
    for (const auto& dim2 : dims) {
        uint64_t dim = dim2;
        auto vec1 = fixtures::generate_int8_codes(count * 2, dim);
        std::vector<int8_t> vec2(vec1.begin() + count, vec1.end());
        for (uint64_t i = 0; i < count; ++i) {
            TEST_ACCURACY(INT8InnerProduct);
            TEST_ACCURACY(INT8InnerProductDistance);
        }
    }
}

TEST_CASE("PQ Calculation", "[ut][simd]") {
    size_t dim = 256;
    float single_dim_value = 0.571;
    float results_expected[256]{0.0f};
    float results[256]{0.0f};
    auto vectors = fixtures::generate_vectors(1, dim);

    generic::PQDistanceFloat256(vectors.data(), single_dim_value, results_expected);
    auto check_func = [&]() {
        for (int i = 0; i < dim; ++i) {
            REQUIRE(std::abs(results_expected[i] - results[i]) < 0.001);
        }
        memset(results, 0, 256 * sizeof(float));
    };
    if (SimdStatus::SupportSSE()) {
        sse::PQDistanceFloat256(vectors.data(), single_dim_value, results);
        check_func();
    }
    if (SimdStatus::SupportAVX()) {
        avx::PQDistanceFloat256(vectors.data(), single_dim_value, results);
        check_func();
    }
    if (SimdStatus::SupportAVX2()) {
        avx2::PQDistanceFloat256(vectors.data(), single_dim_value, results);
        check_func();
    }
    if (SimdStatus::SupportAVX512()) {
        avx512::PQDistanceFloat256(vectors.data(), single_dim_value, results);
        check_func();
    }
    if (SimdStatus::SupportNEON()) {
        neon::PQDistanceFloat256(vectors.data(), single_dim_value, results);
        check_func();
    }
}
