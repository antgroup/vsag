
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

#include "simd/int8_simd.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "simd_status.h"

using namespace vsag;

#define TEST_INT8_COMPUTE_ACCURACY(Func)                                              \
    {                                                                                 \
        float gt, sse, avx, avx2, avx512, neon, sve;                                  \
        gt = generic::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim);        \
        if (SimdStatus::SupportSSE()) {                                               \
            sse = sse::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim);       \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sse));                   \
        }                                                                             \
        if (SimdStatus::SupportAVX()) {                                               \
            avx = avx::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim);       \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx));                   \
        }                                                                             \
        if (SimdStatus::SupportAVX2()) {                                              \
            avx2 = avx2::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim);     \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx2));                  \
        }                                                                             \
        if (SimdStatus::SupportAVX512()) {                                            \
            avx512 = avx512::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim); \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx512));                \
        }                                                                             \
        if (SimdStatus::SupportNEON()) {                                              \
            neon = neon::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim);     \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(neon));                  \
        }                                                                             \
        if (SimdStatus::SupportSVE()) {                                               \
            sve = sve::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim);       \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sve));                   \
        }                                                                             \
    };

TEST_CASE("INT8 SIMD Compute", "[ut][simd][int8]") {
    const std::vector<int64_t> dims = {8, 16, 32, 256};
    int64_t count = 100;
    for (const auto& dim : dims) {
        auto vec1 = fixtures::generate_int8_codes(count * 2, dim);
        std::vector<int8_t> vec2(vec1.begin() + count * dim, vec1.end());
        for (uint64_t i = 0; i < count; ++i) {
            TEST_INT8_COMPUTE_ACCURACY(INT8ComputeL2Sqr);
            TEST_INT8_COMPUTE_ACCURACY(INT8ComputeIP);
        }
        //TODO(lc): Add batch compute func test
        for (uint64_t i = 0; i < count; i += 4) {
        }
    }
}

#define BENCHMARK_ONE(Simd, Func)                                          \
    BENCHMARK_ADVANCED(#Simd #Func) {                                      \
        for (int i = 0; i < count; ++i) {                                  \
            Simd::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim); \
        }                                                                  \
    }

#define TEST_ALL_BACKENDS(func)      \
    BENCHMARK_ONE(generic, func);    \
    if (SimdStatus::SupportSSE())    \
        BENCHMARK_ONE(sse, func);    \
    if (SimdStatus::SupportAVX())    \
        BENCHMARK_ONE(avx, func);    \
    if (SimdStatus::SupportAVX2())   \
        BENCHMARK_ONE(avx2, func);   \
    if (SimdStatus::SupportAVX512()) \
        BENCHMARK_ONE(avx512, func); \
    if (SimdStatus::SupportNEON())   \
        BENCHMARK_ONE(neon, func);   \
    if (SimdStatus::SupportSVE())    \
        BENCHMARK_ONE(sve, func);

TEST_CASE("INT8 Benchmark", "[ut][simd][int8][!benchmark]") {
    int64_t count = 500;
    int64_t dim = 128;
    auto vec1 = fixtures::generate_int8_codes(count, dim);
    auto vec2 = fixtures::generate_int8_codes(count, dim);

    TEST_ALL_BACKENDS(INT8ComputeL2Sqr);
    TEST_ALL_BACKENDS(INT8ComputeIP);
}
