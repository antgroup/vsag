
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

TEST_CASE("FP16 IEEE conversion boundaries", "[ut][simd]") {
    const std::vector<std::pair<uint16_t, float>> cases = {
        {0x0000, 0.0F},
        {0x8000, -0.0F},
        {0x0001, std::ldexp(1.0F, -24)},
        {0x8001, -std::ldexp(1.0F, -24)},
        {0x03FF, std::ldexp(1023.0F, -24)},
        {0x83FF, -std::ldexp(1023.0F, -24)},
        {0x0400, std::ldexp(1.0F, -14)},
        {0x8400, -std::ldexp(1.0F, -14)},
        {0x7BFF, 65504.0F},
        {0xFBFF, -65504.0F},
        {0x7C00, std::numeric_limits<float>::infinity()},
        {0xFC00, -std::numeric_limits<float>::infinity()},
    };
    for (const auto& [bits, value] : cases) {
        CAPTURE(bits);
        const float decoded = generic::FP16ToFloat(bits);
        REQUIRE(decoded == value);
        REQUIRE(std::signbit(decoded) == std::signbit(value));
        REQUIRE(generic::FloatToFP16(value) == bits);
    }
    for (uint16_t bits : {0x7C01, 0x7E00, 0x7FFF, 0xFC01, 0xFE00, 0xFFFF}) {
        CAPTURE(bits);
        const float decoded = generic::FP16ToFloat(bits);
        REQUIRE(std::isnan(decoded));
        REQUIRE(std::signbit(decoded) == ((bits & 0x8000) != 0));
        REQUIRE(generic::FloatToFP16(decoded) == (bits | 0x0200));
    }
    REQUIRE((generic::FloatToFP16(std::numeric_limits<float>::quiet_NaN()) & 0x7FFF) > 0x7C00);
}

TEST_CASE("FP16 conversion rounds to nearest even", "[ut][simd]") {
    REQUIRE(generic::FloatToFP16(std::ldexp(1.0F, -25)) == 0x0000);
    REQUIRE(generic::FloatToFP16(-std::ldexp(1.0F, -25)) == 0x8000);
    REQUIRE(generic::FloatToFP16(std::nextafter(std::ldexp(1.0F, -25), 1.0F)) == 0x0001);
    REQUIRE(generic::FloatToFP16(std::ldexp(3.0F, -25)) == 0x0002);
    REQUIRE(generic::FloatToFP16(std::ldexp(2047.0F, -25)) == 0x0400);
    REQUIRE(generic::FloatToFP16(1.0F + std::ldexp(1.0F, -11)) == 0x3C00);
    REQUIRE(generic::FloatToFP16(1.0F + std::ldexp(3.0F, -11)) == 0x3C02);
    REQUIRE(generic::FloatToFP16(65519.0F) == 0x7BFF);
    REQUIRE(generic::FloatToFP16(65520.0F) == 0x7C00);
    REQUIRE(generic::FloatToFP16(-65520.0F) == 0xFC00);
    REQUIRE(generic::FloatToFP16(std::numeric_limits<float>::max()) == 0x7C00);
    REQUIRE(generic::FloatToFP16(std::numeric_limits<float>::denorm_min()) == 0x0000);
    REQUIRE(generic::FloatToFP16(-std::numeric_limits<float>::denorm_min()) == 0x8000);
}

TEST_CASE("FP16 finite bit patterns roundtrip", "[ut][simd]") {
    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        if ((bits & 0x7C00) == 0x7C00) {
            continue;
        }
        CAPTURE(bits);
        const uint32_t exponent = (bits >> 10) & 0x1F;
        const uint32_t mantissa = bits & 0x3FF;
        float expected = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -24)
                                       : std::ldexp(static_cast<float>(1024 + mantissa),
                                                    static_cast<int>(exponent) - 25);
        if ((bits & 0x8000) != 0) {
            expected = -expected;
        }
        const float decoded = generic::FP16ToFloat(static_cast<uint16_t>(bits));
        REQUIRE(decoded == expected);
        REQUIRE(std::signbit(decoded) == std::signbit(expected));
        REQUIRE(generic::FloatToFP16(decoded) == bits);
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

TEST_CASE("FP16 SIMD conversion boundaries", "[ut][simd]") {
    const int64_t dim = GENERATE(1, 7, 8, 15, 16, 17, 32);
    const uint16_t boundary_bits[] = {
        0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x0400, 0x8400};
    std::vector<uint8_t> vec1(dim * 2);
    std::vector<uint8_t> vec2(dim * 2);
    auto* query = reinterpret_cast<uint16_t*>(vec1.data());
    auto* codes = reinterpret_cast<uint16_t*>(vec2.data());
    for (int64_t j = 0; j < dim; ++j) {
        query[j] = boundary_bits[j % 8];
        codes[j] = 0x3C00;  // Exactly 1.0, keeping subnormal contributions observable.
    }
    const int64_t i = 0;
    TEST_ACCURACY(FP16ComputeIP);
    TEST_ACCURACY(FP16ComputeL2Sqr);
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
