
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

#include "sq8_uniform_simd.h"

#include <catch2/catch_test_macros.hpp>

#include "catch2/benchmark/catch_benchmark.hpp"
#include "fixtures.h"
#include "simd_status.h"

using namespace vsag;

#ifndef ENABLE_SSE
namespace sse = generic;
#endif

#ifndef ENABLE_AVX2
namespace avx2 = sse;
#endif

#ifndef ENABLE_AVX512
namespace avx512 = avx2;
#endif

#define TEST_ACCURACY(Func)                                                                      \
    {                                                                                            \
        auto gt =                                                                                \
            generic::Func(codes1.data() + i * code_size, codes2.data() + i * code_size, dim);    \
        if (SimdStatus::SupportSSE()) {                                                          \
            auto sse =                                                                           \
                sse::Func(codes1.data() + i * code_size, codes2.data() + i * code_size, dim);    \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sse));                              \
        }                                                                                        \
        if (SimdStatus::SupportAVX2()) {                                                         \
            auto avx2 =                                                                          \
                avx2::Func(codes1.data() + i * code_size, codes2.data() + i * code_size, dim);   \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx2));                             \
        }                                                                                        \
        if (SimdStatus::SupportAVX512()) {                                                         \
            auto avx512 =                                                                        \
                avx512::Func(codes1.data() + i * code_size, codes2.data() + i * code_size, dim); \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx512));                           \
        }                                                                                        \
    }

TEST_CASE("SQ8 Uniform SIMD Compute Codes", "[SQ8 Uniform SIMD]") {
    auto dims = fixtures::get_common_used_dims();
    int64_t count = 100;
    for (const auto& dim : dims) {
        uint32_t code_size = dim;
        auto codes1 = fixtures::generate_uint8_codes(count, dim, 114);
        auto codes2 = fixtures::generate_uint8_codes(count, dim, 514);
        for (uint64_t i = 0; i < count; ++i) {
            TEST_ACCURACY(SQ8UniformComputeCodesIP);
        }
    }
}

#define BENCHMARK_SIMD_COMPUTE(Simd, Comp)                                                 \
    BENCHMARK_ADVANCED(#Simd #Comp) {                                                      \
        for (int i = 0; i < count; ++i) {                                                  \
            Simd::Comp(codes1.data() + i * code_size, codes2.data() + i * code_size, dim); \
        }                                                                                  \
        return;                                                                            \
    }

TEST_CASE("SQ8 Uniform SIMD Compute Benchmark", "[simd][!benchmark]") {
    int64_t count = 100;
    int64_t dim = 256;
    uint32_t code_size = dim;

    auto codes1 = fixtures::generate_uint8_codes(count, dim, 114);
    auto codes2 = fixtures::generate_uint8_codes(count, dim, 514);
    BENCHMARK_SIMD_COMPUTE(generic, SQ8UniformComputeCodesIP);
    BENCHMARK_SIMD_COMPUTE(sse, SQ8UniformComputeCodesIP);
    BENCHMARK_SIMD_COMPUTE(avx2, SQ8UniformComputeCodesIP);
    BENCHMARK_SIMD_COMPUTE(avx512, SQ8UniformComputeCodesIP);
}
