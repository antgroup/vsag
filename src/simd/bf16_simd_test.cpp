
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

#include "bf16_simd.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_all.hpp>

#include "fixtures.h"
#include "simd_status.h"

using namespace vsag;

std::vector<uint8_t>
encode_bf16(const std::vector<float>& data, const int64_t count) {
    std::vector<uint8_t> result(count * 2);
    auto* bf16 = reinterpret_cast<uint16_t*>(result.data());
    for (int64_t i = 0; i < count; ++i) {
        bf16[i] = generic::FloatToBF16(data[i]);
    }
    return result;
}

TEST_CASE("Encode & Decode BF16", "[ut][simd]") {
    auto vec_fp32 = fixtures::generate_vectors(10, 100);
    for (int64_t i = 0; i < 1000; ++i) {
        uint16_t bf16 = generic::FloatToBF16(vec_fp32[i]);
        float decode_fp32 = generic::BF16ToFloat(bf16);
        REQUIRE(std::abs(decode_fp32 - vec_fp32[i]) < 5e-4);
    }
}

#define TEST_ACCURACY(Func)                                                           \
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

TEST_CASE("BF16 SIMD Compute", "[ut][simd]") {
    int64_t dim = GENERATE(1, 8, 16, 32, 256);
    int64_t count = 100;

    auto vec1_fp32 = fixtures::generate_vectors(count, dim, false, 39);
    auto vec1 = encode_bf16(vec1_fp32, count * dim);
    auto vec2_fp32 = fixtures::generate_vectors(count, dim, false, 87);
    auto vec2 = encode_bf16(vec2_fp32, count * dim);
    for (uint64_t j = 0; j < count; ++j) {
        auto i = j * 2;
        TEST_ACCURACY(BF16ComputeIP);
        TEST_ACCURACY(BF16ComputeL2Sqr);
    }
}

#define BENCHMARK_SIMD_COMPUTE(Simd, Comp)                                         \
    BENCHMARK_ADVANCED(#Simd #Comp) {                                              \
        for (int i = 0; i < count; ++i) {                                          \
            Simd::Comp(vec1.data() + i * 2 * dim, vec2.data() + i * 2 * dim, dim); \
        }                                                                          \
        return;                                                                    \
    }

TEST_CASE("BF16 Benchmark", "[ut][simd][!benchmark]") {
    int64_t count = 500;
    int64_t dim = 128;
    auto vec1_fp32 = fixtures::generate_vectors(count, dim, false, 37);
    auto vec1 = encode_bf16(vec1_fp32, count * dim);
    auto vec2_fp32 = fixtures::generate_vectors(count, dim, false, 86);
    auto vec2 = encode_bf16(vec2_fp32, count * dim);
    BENCHMARK_SIMD_COMPUTE(generic, BF16ComputeIP);
    if (SimdStatus::SupportSSE()) {
        BENCHMARK_SIMD_COMPUTE(sse, BF16ComputeIP);
    }
    if (SimdStatus::SupportAVX()) {
        BENCHMARK_SIMD_COMPUTE(avx, BF16ComputeIP);
    }
    if (SimdStatus::SupportAVX2()) {
        BENCHMARK_SIMD_COMPUTE(avx2, BF16ComputeIP);
    }
    if (SimdStatus::SupportAVX512()) {
        BENCHMARK_SIMD_COMPUTE(avx512, BF16ComputeIP);
    }
    if (SimdStatus::SupportNEON()) {
        BENCHMARK_SIMD_COMPUTE(neon, BF16ComputeIP);
    }
    if (SimdStatus::SupportSVE()) {
        BENCHMARK_SIMD_COMPUTE(sve, BF16ComputeIP);
    }

    BENCHMARK_SIMD_COMPUTE(generic, BF16ComputeL2Sqr);
    if (SimdStatus::SupportSSE()) {
        BENCHMARK_SIMD_COMPUTE(sse, BF16ComputeL2Sqr);
    }
    if (SimdStatus::SupportAVX()) {
        BENCHMARK_SIMD_COMPUTE(avx, BF16ComputeL2Sqr);
    }
    if (SimdStatus::SupportAVX2()) {
        BENCHMARK_SIMD_COMPUTE(avx2, BF16ComputeL2Sqr);
    }
    if (SimdStatus::SupportAVX512()) {
        BENCHMARK_SIMD_COMPUTE(avx512, BF16ComputeL2Sqr);
    }
    if (SimdStatus::SupportNEON()) {
        BENCHMARK_SIMD_COMPUTE(neon, BF16ComputeL2Sqr);
    }
    if (SimdStatus::SupportSVE()) {
        BENCHMARK_SIMD_COMPUTE(sve, BF16ComputeL2Sqr);
    }
}
