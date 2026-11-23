
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

#include "simd_status.h"
#include "unittest.h"

using namespace vsag;

#define TEST_ACCURACY_CODES(Func)                                                           \
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
        if (SimdStatus::SupportSVE()) {                                                     \
            auto sve = sve::Func(                                                           \
                vec1.data() + i * dim, vec2.data() + i * dim, lb.data(), diff.data(), dim); \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sve));                         \
        }                                                                                   \
    }

#define TEST_ACCURACY_QUERY(Func)                                                           \
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
        if (SimdStatus::SupportSVE()) {                                                     \
            auto sve = sve::Func(                                                           \
                vec1.data() + i * dim, vec2.data() + i * dim, lb.data(), diff.data(), dim); \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sve));                         \
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
            TEST_ACCURACY_CODES(SQ8ComputeCodesIP);
            TEST_ACCURACY_CODES(SQ8ComputeCodesL2Sqr);
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
            TEST_ACCURACY_QUERY(SQ8ComputeIP);
            TEST_ACCURACY_QUERY(SQ8ComputeL2Sqr);
        }
    }
}

TEST_CASE("SQ8 L2 Batch4 matches four single distances", "[ut][simd][batch4]") {
    for (const uint64_t dim : {1ULL, 7ULL, 16ULL, 17ULL, 31ULL, 32ULL, 960ULL}) {
        auto query = fixtures::generate_vectors(1, dim);
        auto lower_bound = fixtures::generate_vectors(1, dim, true, 183);
        auto diff = fixtures::generate_vectors(1, dim, true, 657);
        std::vector<uint8_t> codes(4 * dim);
        for (uint64_t i = 0; i < codes.size(); ++i) {
            codes[i] = static_cast<uint8_t>((i * 37) % 256);
        }

        // The dispatched kernel and the fallback used on CPUs without a batched kernel.
        for (const SQ8ComputeBatch4Type batch4 :
             {SQ8ComputeL2SqrBatch4, &SQ8ComputeL2SqrBatch4Fallback}) {
            float dist1 = 0, dist2 = 0, dist3 = 0, dist4 = 0;
            batch4(query.data(),
                   codes.data(),
                   codes.data() + dim,
                   codes.data() + 2 * dim,
                   codes.data() + 3 * dim,
                   lower_bound.data(),
                   diff.data(),
                   dim,
                   dist1,
                   dist2,
                   dist3,
                   dist4);
            const float actual[4] = {dist1, dist2, dist3, dist4};
            for (uint64_t i = 0; i < 4; ++i) {
                const float expected = SQ8ComputeL2Sqr(
                    query.data(), codes.data() + i * dim, lower_bound.data(), diff.data(), dim);
                REQUIRE(fixtures::dist_t(actual[i]) == fixtures::dist_t(expected));
            }
        }
    }
}

TEST_CASE("SQ8 IP Batch4 matches four single distances", "[ut][simd][batch4]") {
    for (const uint64_t dim : {1ULL, 7ULL, 16ULL, 17ULL, 31ULL, 32ULL, 768ULL}) {
        auto query = fixtures::generate_vectors(1, dim);
        auto lower_bound = fixtures::generate_vectors(1, dim, true, 183);
        auto diff = fixtures::generate_vectors(1, dim, true, 657);
        std::vector<uint8_t> codes(4 * dim);
        for (uint64_t i = 0; i < codes.size(); ++i) {
            codes[i] = static_cast<uint8_t>((i * 37) % 256);
        }

        // The dispatched kernel and the fallback used on CPUs without a batched kernel.
        for (const SQ8ComputeBatch4Type batch4 :
             {SQ8ComputeIPBatch4, &SQ8ComputeIPBatch4Fallback}) {
            float dist1 = 0, dist2 = 0, dist3 = 0, dist4 = 0;
            batch4(query.data(),
                   codes.data(),
                   codes.data() + dim,
                   codes.data() + 2 * dim,
                   codes.data() + 3 * dim,
                   lower_bound.data(),
                   diff.data(),
                   dim,
                   dist1,
                   dist2,
                   dist3,
                   dist4);
            const float actual[4] = {dist1, dist2, dist3, dist4};
            for (uint64_t i = 0; i < 4; ++i) {
                const float expected = SQ8ComputeIP(
                    query.data(), codes.data() + i * dim, lower_bound.data(), diff.data(), dim);
                REQUIRE(fixtures::dist_t(actual[i]) == fixtures::dist_t(expected));
            }
        }
    }
}

TEST_CASE("SQ8 fused batch4 covers each enabled backend", "[ut][simd][batch4]") {
    struct Backend {
        const char* name;
        bool enabled;
        SQ8ComputeBatch4Type batch_ip;
        SQ8ComputeBatch4Type batch_l2;
    };
    const Backend backends[] = {
        {"SSE", SimdStatus::SupportSSE(), sse::SQ8ComputeIPBatch4, sse::SQ8ComputeL2SqrBatch4},
        {"AVX", SimdStatus::SupportAVX(), avx::SQ8ComputeIPBatch4, avx::SQ8ComputeL2SqrBatch4},
        {"AVX2", SimdStatus::SupportAVX2(), avx2::SQ8ComputeIPBatch4, avx2::SQ8ComputeL2SqrBatch4},
        {"AVX-512",
         SimdStatus::SupportAVX512(),
         avx512::SQ8ComputeIPBatch4,
         avx512::SQ8ComputeL2SqrBatch4},
        {"NEON", SimdStatus::SupportNEON(), neon::SQ8ComputeIPBatch4, neon::SQ8ComputeL2SqrBatch4},
        {"SVE", SimdStatus::SupportSVE(), sve::SQ8ComputeIPBatch4, sve::SQ8ComputeL2SqrBatch4},
    };
    for (const uint64_t dim : {0ULL,
                               1ULL,
                               3ULL,
                               4ULL,
                               5ULL,
                               7ULL,
                               8ULL,
                               9ULL,
                               15ULL,
                               16ULL,
                               17ULL,
                               31ULL,
                               32ULL,
                               33ULL,
                               127ULL,
                               1024ULL}) {
        const auto query = fixtures::generate_vectors(1, dim + 1, true, 183);
        const auto lower = fixtures::generate_vectors(1, dim + 1, true, 186);
        const auto diff = fixtures::generate_vectors(1, dim + 1, true, 657);
        std::vector<uint8_t> codes(4 * dim + 1);
        for (uint64_t i = 0; i < codes.size(); ++i) {
            codes[i] = static_cast<uint8_t>((i * 37 + 19) % 256);
        }
        const auto* q = query.data() + 1;
        const auto* lb = lower.data() + 1;
        const auto* range = diff.data() + 1;
        const auto* base = codes.data() + 1;
        for (const auto& backend : backends) {
            if (!backend.enabled) {
                continue;
            }
            CAPTURE(backend.name, dim);
            for (bool is_l2 : {false, true}) {
                CAPTURE(is_l2);
                const auto batch = is_l2 ? backend.batch_l2 : backend.batch_ip;
                const auto single = is_l2 ? generic::SQ8ComputeL2Sqr : generic::SQ8ComputeIP;
                float distances[4] = {-1.0F, 100.0F, -15.0F, 57.0F};
                batch(q,
                      base,
                      base + dim,
                      base + 2 * dim,
                      base + 3 * dim,
                      lb,
                      range,
                      dim,
                      distances[0],
                      distances[1],
                      distances[2],
                      distances[3]);
                for (uint64_t i = 0; i < 4; ++i) {
                    REQUIRE(fixtures::dist_t(distances[i]) ==
                            fixtures::dist_t(single(q, base + i * dim, lb, range, dim)));
                }
            }
        }
    }
}

#define BENCHMARK_SIMD_COMPUTE_QUERY(Simd, Comp)                                                   \
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
    BENCHMARK_SIMD_COMPUTE_QUERY(generic, SQ8ComputeIP);
    if (SimdStatus::SupportSSE()) {
        BENCHMARK_SIMD_COMPUTE_QUERY(sse, SQ8ComputeIP);
    }
    if (SimdStatus::SupportAVX()) {
        BENCHMARK_SIMD_COMPUTE_QUERY(avx, SQ8ComputeIP);
    }
    if (SimdStatus::SupportAVX2()) {
        BENCHMARK_SIMD_COMPUTE_QUERY(avx2, SQ8ComputeIP);
    }
    if (SimdStatus::SupportAVX512()) {
        BENCHMARK_SIMD_COMPUTE_QUERY(avx512, SQ8ComputeIP);
    }
    if (SimdStatus::SupportNEON()) {
        BENCHMARK_SIMD_COMPUTE_QUERY(neon, SQ8ComputeIP);
    }
    if (SimdStatus::SupportSVE()) {
        BENCHMARK_SIMD_COMPUTE_QUERY(sve, SQ8ComputeIP);
    }
}
