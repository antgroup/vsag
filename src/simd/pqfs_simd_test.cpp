
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

#include "pqfs_simd.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "simd_status.h"

using namespace vsag;

template <class T>
bool
compare_vector(std::vector<T>& v1, std::vector<T>& v2) {
    if (v1.size() != v2.size()) {
        return false;
    }
    for (size_t i = 0; i < v1.size(); ++i) {
        if (v1[i] != v2[i]) {
            return false;
        }
    }
    return true;
}

#define TEST_ACCURACY(Func)                                                                     \
    {                                                                                           \
        std::vector<int32_t> gt(32, 0);                                                         \
        std::vector<int32_t> sse_data(32, 0);                                                   \
        std::vector<int32_t> avx_data(32, 0);                                                   \
        std::vector<int32_t> avx2_data(32, 0);                                                  \
        std::vector<int32_t> avx512_data(32, 0);                                                \
        std::vector<int32_t> neon_data(32, 0);                                                  \
        std::vector<int32_t> sve_data(32, 0);                                                   \
        generic::Func(lut.data() + i * dim, codes.data() + i * dim, pq_dim, gt.data());         \
        if (SimdStatus::SupportSSE()) {                                                         \
            sse::Func(lut.data() + i * dim, codes.data() + i * dim, pq_dim, sse_data.data());   \
            REQUIRE(compare_vector(gt, sse_data) == true);                                      \
        }                                                                                       \
        if (SimdStatus::SupportAVX()) {                                                         \
            avx::Func(lut.data() + i * dim, codes.data() + i * dim, pq_dim, avx_data.data());   \
            REQUIRE(compare_vector(gt, avx_data) == true);                                      \
        }                                                                                       \
        if (SimdStatus::SupportAVX2()) {                                                        \
            avx2::Func(lut.data() + i * dim, codes.data() + i * dim, pq_dim, avx2_data.data()); \
            REQUIRE(compare_vector(gt, avx2_data) == true);                                     \
        }                                                                                       \
        if (SimdStatus::SupportAVX512()) {                                                      \
            avx512::Func(                                                                       \
                lut.data() + i * dim, codes.data() + i * dim, pq_dim, avx512_data.data());      \
            REQUIRE(compare_vector(gt, avx512_data) == true);                                   \
        }                                                                                       \
        if (SimdStatus::SupportNEON()) {                                                        \
            neon::Func(lut.data() + i * dim, codes.data() + i * dim, pq_dim, neon_data.data()); \
            REQUIRE(compare_vector(gt, neon_data) == true);                                     \
        }                                                                                       \
        if (SimdStatus::SupportSVE()) {                                                         \
            sve::Func(lut.data() + i * dim, codes.data() + i * dim, pq_dim, sve_data.data());   \
            REQUIRE(compare_vector(gt, sve_data) == true);                                      \
        }                                                                                       \
    };

TEST_CASE("PQFastScan SIMD Compute", "[ut][simd]") {
    const std::vector<int64_t> dims = {8, 16, 31, 256};
    int64_t count = 100;
    for (const auto& pq_dim : dims) {
        auto dim = pq_dim * 16;
        auto lut =
            fixtures::generate_uint8_codes(count, pq_dim * 16, fixtures::RandomValue(0, 999));
        auto codes =
            fixtures::generate_uint8_codes(count, pq_dim * 16, fixtures::RandomValue(0, 9999));
        for (uint64_t i = 0; i < count; ++i) {
            TEST_ACCURACY(PQFastScanLookUp32);
        }
    }
}

#define BENCHMARK_SIMD_COMPUTE(Simd, Comp)                                               \
    BENCHMARK_ADVANCED(#Simd #Comp) {                                                    \
        for (int i = 0; i < count; ++i) {                                                \
            Simd::Comp(lut.data() + i * dim, codes.data() + i * dim, pq_dim, gt.data()); \
        }                                                                                \
        return;                                                                          \
    }

TEST_CASE("PQFastScan Benchmark", "[ut][simd][!benchmark]") {
    int64_t count = 500;
    int64_t pq_dim = 128;
    auto dim = pq_dim * 16;
    auto lut = fixtures::generate_uint8_codes(count, pq_dim * 16, fixtures::RandomValue(0, 999));
    auto codes = fixtures::generate_uint8_codes(count, pq_dim * 16, fixtures::RandomValue(0, 9999));
    std::vector<int32_t> gt(32);

    BENCHMARK_SIMD_COMPUTE(generic, PQFastScanLookUp32);
    if (SimdStatus::SupportSSE()) {
        BENCHMARK_SIMD_COMPUTE(sse, PQFastScanLookUp32);
    }
    if (SimdStatus::SupportAVX()) {
        BENCHMARK_SIMD_COMPUTE(avx, PQFastScanLookUp32);
    }
    if (SimdStatus::SupportAVX2()) {
        BENCHMARK_SIMD_COMPUTE(avx2, PQFastScanLookUp32);
    }
    if (SimdStatus::SupportAVX512()) {
        BENCHMARK_SIMD_COMPUTE(avx512, PQFastScanLookUp32);
    }
    if (SimdStatus::SupportNEON()) {
        BENCHMARK_SIMD_COMPUTE(neon, PQFastScanLookUp32);
    }
    if (SimdStatus::SupportSVE()) {
        BENCHMARK_SIMD_COMPUTE(sve, PQFastScanLookUp32);
    }
}
