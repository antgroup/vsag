
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

#include "fixtures.h"
#include "simd_status.h"

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

#define TEST_ACCURACY(Func)                                                           \
    {                                                                                 \
        float gt, sse, avx, avx2, avx512, neon;                                       \
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
}
