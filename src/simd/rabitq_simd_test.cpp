
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

#include "rabitq_simd.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "fp32_simd.h"

using namespace vsag;

#define TEST_ACCURACY_FP32(Func)                                 \
    {                                                            \
        float generic, sse, avx, avx2, avx512, neon;             \
        generic = generic::Func(query, base, dim, inv_sqrt_d);   \
        REQUIRE(std::abs(gt - generic) < 1e-4);                  \
        if (SimdStatus::SupportSSE()) {                          \
            sse = sse::Func(query, base, dim, inv_sqrt_d);       \
            REQUIRE(std::abs(gt - sse) < 1e-4);                  \
        }                                                        \
        if (SimdStatus::SupportAVX()) {                          \
            avx = avx::Func(query, base, dim, inv_sqrt_d);       \
            REQUIRE(std::abs(gt - avx) < 1e-4);                  \
        }                                                        \
        if (SimdStatus::SupportAVX2()) {                         \
            avx2 = avx2::Func(query, base, dim, inv_sqrt_d);     \
            REQUIRE(std::abs(gt - avx2) < 1e-4);                 \
        }                                                        \
        if (SimdStatus::SupportAVX512()) {                       \
            avx512 = avx512::Func(query, base, dim, inv_sqrt_d); \
            REQUIRE(std::abs(gt - avx512) < 1e-4);               \
        }                                                        \
        if (SimdStatus::SupportNEON()) {                         \
            neon = neon::Func(query, base, dim, inv_sqrt_d);     \
            REQUIRE(std::abs(gt - neon) < 1e-4);                 \
        }                                                        \
    };

#define TEST_ACCURACY_SQ4(Func)                                        \
    {                                                                  \
        float gt, avx512;                                              \
        gt = generic::Func(codes.data(), bits.data(), dim);            \
        if (SimdStatus::SupportAVX512()) {                             \
            avx512 = avx512::Func(codes.data(), bits.data(), dim);     \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx512)); \
        }                                                              \
    };

#define BENCHMARK_SIMD_COMPUTE_SQ4(Simd, Comp)          \
    BENCHMARK_ADVANCED(#Simd #Comp) {                   \
        for (int i = 0; i < count; ++i) {               \
            Simd::Comp(codes.data(), bits.data(), dim); \
        }                                               \
        return;                                         \
    }

TEST_CASE("RaBitQ SQ4U-BQ Compute Benchmark", "[ut][simd][!benchmark]") {
    std::vector<uint8_t> codes = {0xFF,
                                  0xFF,  // [1111 1111, 1111 1111]
                                  0x0F,
                                  0x0F,  // [0000 1111, 0000 1111]
                                  0xF0,
                                  0xF0,  // [1111 0000, 1111 0000]
                                  0x00,
                                  0x00};  // [0000 0000, 0000 0000]
    codes.resize(64);
    std::vector<uint8_t> bits = {0xAA, 0x55};  // [1010 1010, 0101 0101]
    bits.resize(64);

    int count = 10000;
    int dim = 32;
    BENCHMARK_SIMD_COMPUTE_SQ4(generic, RaBitQSQ4UBinaryIP);
    BENCHMARK_SIMD_COMPUTE_SQ4(avx512, RaBitQSQ4UBinaryIP);
}

TEST_CASE("RaBitQ SQ4U-BQ Compute Codes", "[ut][simd]") {
    std::vector<uint8_t> codes = {0xFF,
                                  0xFF,  // [1111 1111, 1111 1111]
                                  0x0F,
                                  0x0F,  // [0000 1111, 0000 1111]
                                  0xF0,
                                  0xF0,  // [1111 0000, 1111 0000]
                                  0x00,
                                  0x00};  // [0000 0000, 0000 0000]
    codes.resize(64);
    std::vector<uint8_t> bits = {0xAA, 0x55};  // [1010 1010, 0101 0101]
    bits.resize(64);

    for (auto dim = 0; dim < 17; dim++) {
        uint32_t result = generic::RaBitQSQ4UBinaryIP(codes.data(), bits.data(), dim);
        TEST_ACCURACY_SQ4(RaBitQSQ4UBinaryIP);
        if (dim == 0) {
            REQUIRE(result == 0);
        } else if (dim <= 8) {
            // 4 * 1 + 4 * 2 + 2 * 4 + 2 * 8
            REQUIRE(result == 36);
        } else {
            // 8 * 1 + 4 * 2 + 4 * 4 + 0 * 8
            REQUIRE(result == 32);
        }
    }
}

TEST_CASE("RaBitQ FP32-BQ SIMD Compute Codes", "[ut][simd]") {
    auto dims = fixtures::get_common_used_dims();
    int64_t count = 100;
    for (const auto& dim : dims) {
        uint32_t code_size = (dim + 7) / 8;
        float inv_sqrt_d = 1.0f / sqrt(dim);
        std::vector<float> queries;
        std::vector<uint8_t> bases;
        std::tie(queries, bases) = fixtures::GenerateBinaryVectorsAndCodes(count, dim);
        for (uint64_t i = 0; i < count; ++i) {
            auto* query = queries.data() + i * dim;
            auto* base = bases.data() + i * code_size;

            auto gt = FP32ComputeIP(query, query, dim);
            TEST_ACCURACY_FP32(RaBitQFloatBinaryIP);
        }
    }
}

#define BENCHMARK_SIMD_COMPUTE(Simd, Comp)                                                      \
    BENCHMARK_ADVANCED(#Simd #Comp) {                                                           \
        for (int i = 0; i < count; ++i) {                                                       \
            Simd::Comp(                                                                         \
                queries.data() + i * dim, bases.data() + i * code_size, dim, 1.0f / sqrt(dim)); \
        }                                                                                       \
        return;                                                                                 \
    }

TEST_CASE("RaBitQ FP32-BQ SIMD Compute Benchmark", "[ut][simd][!benchmark]") {
    int64_t count = 100;
    int64_t dim = 256;

    uint32_t code_size = (dim + 7) / 8;
    std::vector<float> queries;
    std::vector<uint8_t> bases;
    std::tie(queries, bases) = fixtures::GenerateBinaryVectorsAndCodes(count, dim);

    BENCHMARK_SIMD_COMPUTE(generic, RaBitQFloatBinaryIP);
    BENCHMARK_SIMD_COMPUTE(sse, RaBitQFloatBinaryIP);
    BENCHMARK_SIMD_COMPUTE(avx, RaBitQFloatBinaryIP);
    BENCHMARK_SIMD_COMPUTE(avx2, RaBitQFloatBinaryIP);
    BENCHMARK_SIMD_COMPUTE(avx512, RaBitQFloatBinaryIP);
    BENCHMARK_SIMD_COMPUTE(neon, RaBitQFloatBinaryIP);
}
