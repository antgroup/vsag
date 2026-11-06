
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

#include "fp16_simd.h"

#include <bit>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_all.hpp>
#include <cmath>
#include <limits>

#include "simd_status.h"
#include "unittest.h"

using namespace vsag;

std::vector<uint8_t>
encode_fp16(const std::vector<float>& data, const int64_t count) {
    std::vector<uint8_t> result(count * 2);
    auto* fp16 = reinterpret_cast<uint16_t*>(result.data());
    for (int64_t i = 0; i < count; ++i) {
        fp16[i] = generic::FloatToFP16(data[i]);
    }
    return result;
}

TEST_CASE("Encode & Decode FP16", "[ut][simd]") {
    auto vec_fp32 = fixtures::generate_vectors(10, 100);
    for (float item : vec_fp32) {
        uint16_t fp16 = generic::FloatToFP16(item);
        float decode_fp32 = generic::FP16ToFloat(fp16);
        REQUIRE(std::abs(decode_fp32 - item) < 3e-4);
    }
    for (float item = -512.0F; item <= 512.0F; item += 1.0F) {
        uint16_t fp16 = generic::FloatToFP16(item);
        float decode_fp32 = generic::FP16ToFloat(fp16);
        REQUIRE(std::abs(decode_fp32 - item) < 1);
    }
}

TEST_CASE("FP16 conversion preserves non-finite values", "[ut][simd][nonfinite]") {
    const auto sign = GENERATE(1.0F, -1.0F);
    const auto infinity = std::copysign(std::numeric_limits<float>::infinity(), sign);
    const auto infinity_code = generic::FloatToFP16(infinity);
    REQUIRE((infinity_code & 0x7FFFU) == 0x7C00U);
    REQUIRE(std::isinf(generic::FP16ToFloat(infinity_code)));
    REQUIRE(std::signbit(generic::FP16ToFloat(infinity_code)) == std::signbit(infinity));

    const auto nan_payload = GENERATE(0x7FC00000U, 0x7F800001U);
    const auto nan = std::bit_cast<float>(nan_payload | (sign < 0 ? 0x80000000U : 0U));
    const auto nan_code = generic::FloatToFP16(nan);
    REQUIRE((nan_code & 0x7C00U) == 0x7C00U);
    REQUIRE((nan_code & 0x03FFU) != 0);
    REQUIRE(std::isnan(generic::FP16ToFloat(nan_code)));
    REQUIRE(std::signbit(generic::FP16ToFloat(nan_code)) == std::signbit(nan));
}

TEST_CASE("FP16 distance kernels preserve NaN across SIMD tails", "[ut][simd][nonfinite]") {
    const auto dim = GENERATE(1, 7, 16, 17, 64);
    std::vector<uint16_t> query(dim, 0x3C00U);
    std::vector<uint16_t> codes(dim, 0x3C00U);
    codes.back() = generic::FloatToFP16(std::numeric_limits<float>::quiet_NaN());
    const auto* query_bytes = reinterpret_cast<const uint8_t*>(query.data());
    const auto* code_bytes = reinterpret_cast<const uint8_t*>(codes.data());
    REQUIRE(std::isnan(generic::FP16ComputeIP(query_bytes, code_bytes, dim)));
    REQUIRE(std::isnan(generic::FP16ComputeL2Sqr(query_bytes, code_bytes, dim)));
    REQUIRE(std::isnan(FP16ComputeIP(query_bytes, code_bytes, dim)));
    REQUIRE(std::isnan(FP16ComputeL2Sqr(query_bytes, code_bytes, dim)));
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

TEST_CASE("FP16 SIMD Compute", "[ut][simd]") {
    int64_t dim = GENERATE(1, 8, 16, 32, 256);
    int64_t count = 100;

    auto vec1_fp32 = fixtures::generate_vectors(count, dim, false, 39);
    auto vec1 = encode_fp16(vec1_fp32, count * dim);
    auto vec2_fp32 = fixtures::generate_vectors(count, dim, false, 87);
    auto vec2 = encode_fp16(vec2_fp32, count * dim);
    for (uint64_t j = 0; j < count; ++j) {
        auto i = j * 2;
        TEST_ACCURACY(FP16ComputeIP);
        TEST_ACCURACY(FP16ComputeL2Sqr);
    }
}

#define BENCHMARK_SIMD_COMPUTE(Simd, Comp)                                         \
    BENCHMARK_ADVANCED(#Simd #Comp) {                                              \
        for (int i = 0; i < count; ++i) {                                          \
            Simd::Comp(vec1.data() + i * 2 * dim, vec2.data() + i * 2 * dim, dim); \
        }                                                                          \
        return;                                                                    \
    }

TEST_CASE("FP16 Benchmark", "[ut][simd][!benchmark]") {
    int64_t count = 500;
    int64_t dim = 128;
    auto vec1_fp32 = fixtures::generate_vectors(count, dim, false, 37);
    auto vec1 = encode_fp16(vec1_fp32, count * dim);
    auto vec2_fp32 = fixtures::generate_vectors(count, dim, false, 86);
    auto vec2 = encode_fp16(vec2_fp32, count * dim);
    BENCHMARK_SIMD_COMPUTE(generic, FP16ComputeIP);
    if (SimdStatus::SupportSSE()) {
        BENCHMARK_SIMD_COMPUTE(sse, FP16ComputeIP);
    }
    if (SimdStatus::SupportAVX2()) {
        BENCHMARK_SIMD_COMPUTE(avx2, FP16ComputeIP);
    }
    if (SimdStatus::SupportAVX512()) {
        BENCHMARK_SIMD_COMPUTE(avx512, FP16ComputeIP);
    }
    if (SimdStatus::SupportNEON()) {
        BENCHMARK_SIMD_COMPUTE(neon, FP16ComputeIP);
    }
    if (SimdStatus::SupportSVE()) {
        BENCHMARK_SIMD_COMPUTE(sve, FP16ComputeIP);
    }

    BENCHMARK_SIMD_COMPUTE(generic, FP16ComputeL2Sqr);
    if (SimdStatus::SupportSSE()) {
        BENCHMARK_SIMD_COMPUTE(sse, FP16ComputeL2Sqr);
    }
    if (SimdStatus::SupportAVX2()) {
        BENCHMARK_SIMD_COMPUTE(avx2, FP16ComputeL2Sqr);
    }
    if (SimdStatus::SupportAVX512()) {
        BENCHMARK_SIMD_COMPUTE(avx512, FP16ComputeL2Sqr);
    }
    if (SimdStatus::SupportNEON()) {
        BENCHMARK_SIMD_COMPUTE(neon, FP16ComputeL2Sqr);
    }
    if (SimdStatus::SupportSVE()) {
        BENCHMARK_SIMD_COMPUTE(sve, FP16ComputeL2Sqr);
    }
}
