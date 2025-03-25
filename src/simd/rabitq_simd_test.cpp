
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

            auto ip_32_32 = FP32ComputeIP(query, query, dim);
            auto ip_32_1_generic = generic::RaBitQFloatBinaryIP(query, base, dim, inv_sqrt_d);
            REQUIRE(std::abs(ip_32_1_generic - ip_32_32) < 1e-4);

            if (SimdStatus::SupportAVX512()) {
                auto ip_32_1_avx512 = avx512::RaBitQFloatBinaryIP(query, base, dim, inv_sqrt_d);
                REQUIRE(std::abs(ip_32_1_avx512 - ip_32_32) < 1e-4);
            }

            if (SimdStatus::SupportAVX2()) {
                auto ip_32_1_avx2 = avx2::RaBitQFloatBinaryIP(query, base, dim, inv_sqrt_d);
                REQUIRE(std::abs(ip_32_1_avx2 - ip_32_32) < 1e-4);
            }

            if (SimdStatus::SupportAVX()) {
                auto ip_32_1_avx = avx::RaBitQFloatBinaryIP(query, base, dim, inv_sqrt_d);
                REQUIRE(std::abs(ip_32_1_avx - ip_32_32) < 1e-4);
            }

            if (SimdStatus::SupportSSE()) {
                auto ip_32_1_sse = sse::RaBitQFloatBinaryIP(query, base, dim, inv_sqrt_d);
                REQUIRE(std::abs(ip_32_1_sse - ip_32_32) < 1e-4);
            }
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
}
