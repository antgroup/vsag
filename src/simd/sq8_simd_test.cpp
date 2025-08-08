
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

#include "sq8_simd.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "simd_status.h"

using namespace vsag;

#define TEST_ACCURACY(Func)                                                                 \
    {                                                                                       \
        auto gt = generic::Func(                                                            \
            vec1.data() + i * dim, vec2.data() + i * dim, lb.data(), diff.data(), dim);     \
        if (SimdStatus::SupportSSE()) {                                                     \
            auto sse = sse::Func(                                                           \
                vec1.data() + i * dim, vec2.data() + i * dim, lb.data(), diff.data(), dim); \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sse));                         \
        }                                                                                   \
        if (SimdStatus::SupportAVX()) {                                                     \
            auto avx = avx::Func(                                                           \
                vec1.data() + i * dim, vec2.data() + i * dim, lb.data(), diff.data(), dim); \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx));                         \
        }                                                                                   \
        if (SimdStatus::SupportAVX2()) {                                                    \
            auto avx2 = avx2::Func(                                                         \
                vec1.data() + i * dim, vec2.data() + i * dim, lb.data(), diff.data(), dim); \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx2));                        \
        }                                                                                   \
        if (SimdStatus::SupportAVX512()) {                                                  \
            auto avx512 = avx512::Func(                                                     \
                vec1.data() + i * dim, vec2.data() + i * dim, lb.data(), diff.data(), dim); \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx512));                      \
        }                                                                                   \
        if (SimdStatus::SupportNEON()) {                                                    \
            auto neon = neon::Func(                                                         \
                vec1.data() + i * dim, vec2.data() + i * dim, lb.data(), diff.data(), dim); \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(neon));                        \
        }                                                                                   \
    }

TEST_CASE("SQ8 SIMD Compute Codes", "[ut][simd]") {
    auto dims = fixtures::get_common_used_dims();
    int64_t count = 100;
    std::vector<uint8_t> vec1, vec2;
    for (const auto& dim : dims) {
        auto vec = fixtures::generate_vectors(count * 2, dim);
        vec1.resize(count * dim);
        std::transform(vec.begin(), vec.begin() + count * dim, vec1.begin(), [](float x) {
            return static_cast<uint8_t>(x * 255.0);
        });
        vec2.resize(count * dim);
        std::transform(vec.begin() + count * dim, vec.end(), vec2.begin(), [](float x) {
            return static_cast<uint8_t>(x * 255.0);
        });
        auto lb = fixtures::generate_vectors(1, dim, true, 186);
        auto diff = fixtures::generate_vectors(1, dim, true, 657);
        for (uint64_t i = 0; i < count; ++i) {
            TEST_ACCURACY(SQ8ComputeCodesIP);
            TEST_ACCURACY(SQ8ComputeCodesL2Sqr);
        }
    }
}

TEST_CASE("SQ8 SIMD Compute", "[ut][simd]") {
    auto dims = fixtures::get_common_used_dims();
    int64_t count = 100;
    for (const auto& dim : dims) {
        auto vec1 = fixtures::generate_vectors(count * 2, dim);
        std::vector<uint8_t> vec2(count * dim);
        std::transform(vec1.begin() + count * dim, vec1.end(), vec2.begin(), [](float x) {
            return uint64_t(x * 255.0);
        });
        auto lb = fixtures::generate_vectors(1, dim, true, 183);
        auto diff = fixtures::generate_vectors(1, dim, true, 657);
        for (uint64_t i = 0; i < count; ++i) {
            TEST_ACCURACY(SQ8ComputeIP);
            TEST_ACCURACY(SQ8ComputeL2Sqr);
        }
    }
}

#define BENCHMARK_SIMD_COMPUTE(Simd, Comp)                                                         \
    BENCHMARK_ADVANCED(#Simd #Comp) {                                                              \
        for (int i = 0; i < count; ++i) {                                                          \
            Simd::Comp(vec1.data() + i * dim, vec2.data() + i * dim, lb.data(), diff.data(), dim); \
        }                                                                                          \
        return;                                                                                    \
    }

TEST_CASE("SQ8 SIMD Compute Benchmark", "[ut][simd][!benchmark]") {
    const std::vector<int64_t> dims = {256};
    int64_t count = 200;
    int64_t dim = 256;

    auto vec1 = fixtures::generate_vectors(count * 2, dim);
    std::vector<uint8_t> vec2(count * dim);
    std::transform(vec1.begin() + count * dim, vec1.end(), vec2.begin(), [](float x) {
        return uint64_t(x * 255.0);
    });
    auto lb = fixtures::generate_vectors(1, dim, true, 180);
    auto diff = fixtures::generate_vectors(1, dim, true, 6217);
    BENCHMARK_SIMD_COMPUTE(generic, SQ8ComputeIP);
    if (SimdStatus::SupportSSE()) {
        BENCHMARK_SIMD_COMPUTE(sse, SQ8ComputeIP);
    }
    if (SimdStatus::SupportAVX()) {
        BENCHMARK_SIMD_COMPUTE(avx, SQ8ComputeIP);
    }
    if (SimdStatus::SupportAVX2()) {
        BENCHMARK_SIMD_COMPUTE(avx2, SQ8ComputeIP);
    }
    if (SimdStatus::SupportAVX512()) {
        BENCHMARK_SIMD_COMPUTE(avx512, SQ8ComputeIP);
    }
    if (SimdStatus::SupportNEON()) {
        BENCHMARK_SIMD_COMPUTE(neon, SQ8ComputeIP);
    }
}
