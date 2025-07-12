
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

#include "sq4_simd.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "simd_status.h"
using namespace vsag;

#define TEST_ACCURACY(Func)                                        \
    {                                                              \
        auto gt = generic::Func(codes1.data() + i * code_size,     \
                                codes2.data() + i * code_size,     \
                                lb.data(),                         \
                                diff.data(),                       \
                                dim);                              \
        auto sse = sse::Func(codes1.data() + i * code_size,        \
                             codes2.data() + i * code_size,        \
                             lb.data(),                            \
                             diff.data(),                          \
                             dim);                                 \
        auto avx = avx::Func(codes1.data() + i * code_size,        \
                             codes2.data() + i * code_size,        \
                             lb.data(),                            \
                             diff.data(),                          \
                             dim);                                 \
        auto avx2 = avx2::Func(codes1.data() + i * code_size,      \
                               codes2.data() + i * code_size,      \
                               lb.data(),                          \
                               diff.data(),                        \
                               dim);                               \
        auto avx512 = avx512::Func(codes1.data() + i * code_size,  \
                                   codes2.data() + i * code_size,  \
                                   lb.data(),                      \
                                   diff.data(),                    \
                                   dim);                           \
        auto neon = neon::Func(codes1.data() + i * code_size,      \
                               codes2.data() + i * code_size,      \
                               lb.data(),                          \
                               diff.data(),                        \
                               dim);                               \
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sse));    \
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx));    \
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx2));   \
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx512)); \
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(neon));   \
    }

TEST_CASE("SQ4 SIMD Compute Codes", "[ut][simd]") {
    const std::vector<uint32_t> dims = {1, 8, 16, 32, 97, 129, 256};
    int64_t count = 100;
    for (const auto& dim : dims) {
        uint32_t code_size = (dim + 1) / 2;
        auto codes1 = fixtures::generate_int4_codes(count, dim);
        auto codes2 = fixtures::generate_int4_codes(count, dim);
        auto lb = fixtures::generate_vectors(1, dim, true, 186);
        auto diff = fixtures::generate_vectors(1, dim, true, 657);
        for (uint64_t i = 0; i < count; ++i) {
            TEST_ACCURACY(SQ4ComputeCodesIP);
            TEST_ACCURACY(SQ4ComputeCodesL2Sqr);
        }
    }
}

TEST_CASE("SQ4 SIMD Compute", "[ut][simd]") {
    const std::vector<int64_t> dims = {1, 8, 16, 32, 97, 129, 256};
    int64_t count = 100;
    for (const auto& dim : dims) {
        uint32_t code_size = (dim + 1) / 2;
        auto codes1 = fixtures::generate_vectors(count, dim);
        std::vector<uint8_t> codes2 = fixtures::generate_int4_codes(count, dim);
        auto lb = fixtures::generate_vectors(1, dim, true, 186);
        auto diff = fixtures::generate_vectors(1, dim, true, 657);
        for (uint64_t i = 0; i < count; ++i) {
            TEST_ACCURACY(SQ4ComputeIP);
            TEST_ACCURACY(SQ4ComputeL2Sqr);
        }
    }
}

#define BENCHMARK_SIMD_COMPUTE(Simd, Comp)            \
    BENCHMARK_ADVANCED(#Simd #Comp) {                 \
        for (int i = 0; i < count; ++i) {             \
            Simd::Comp(codes1.data() + i * dim,       \
                       codes2.data() + i * code_size, \
                       lb.data(),                     \
                       diff.data(),                   \
                       dim);                          \
        }                                             \
        return;                                       \
    }

TEST_CASE("SQ4 SIMD Compute Benchmark", "[ut][simd][!benchmark]") {
    const std::vector<int64_t> dims = {256};
    int64_t count = 200;
    int64_t dim = 256;
    uint32_t code_size = (dim + 1) / 2;

    auto codes1 = fixtures::generate_vectors(count, dim);
    std::vector<uint8_t> codes2 = fixtures::generate_int4_codes(count, dim);
    auto lb = fixtures::generate_vectors(1, dim, true, 180);
    auto diff = fixtures::generate_vectors(1, dim, true, 6217);
    BENCHMARK_SIMD_COMPUTE(generic, SQ4ComputeIP);
    if (SimdStatus::SupportSSE()) {
        BENCHMARK_SIMD_COMPUTE(sse, SQ4ComputeIP);
    }
    if (SimdStatus::SupportAVX()) {
        BENCHMARK_SIMD_COMPUTE(avx, SQ4ComputeIP);
    }
    if (SimdStatus::SupportAVX2()) {
        BENCHMARK_SIMD_COMPUTE(avx2, SQ4ComputeIP);
    }
    if (SimdStatus::SupportAVX512()) {
        BENCHMARK_SIMD_COMPUTE(avx512, SQ4ComputeIP);
    }
    if (SimdStatus::SupportNEON()) {
        BENCHMARK_SIMD_COMPUTE(neon, SQ4ComputeIP);
    }
}
