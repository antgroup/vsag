
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

#include <array>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <cmath>

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

TEST_CASE("SQ8 four-code L2 matches the single-pair kernel including vector tails",
          "[ut][simd][sq8_tile]") {
    const uint64_t dim = GENERATE(1U, 15U, 16U, 17U, 33U, 960U);
    const auto stride = dim + 1;
    std::vector<uint8_t> bytes(5 * stride);
    std::vector<float> lower(dim), scale(dim);
    for (uint64_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<uint8_t>((i * 37U + i / stride * 13U) % 256U);
    }
    for (uint64_t i = 0; i < dim; ++i) {
        lower[i] = -static_cast<float>(i % 17) / 3.0F;
        scale[i] = static_cast<float>(i % 13) / 7.0F;
    }
    const auto* source = bytes.data() + 1;
    const uint8_t* targets[] = {source, source + stride, source + 2 * stride, source + 4 * stride};
    float distances[4]{};
    const auto compute = GENERATE(&vsag::SQ8CodeBatch4Fallback, vsag::SQ8ComputeCodesL2SqrBatch4);
    compute(source, targets, lower.data(), scale.data(), dim, distances);
    for (size_t i = 0; i < 4; ++i) {
        const auto expected =
            vsag::SQ8ComputeCodesL2Sqr(source, targets[i], lower.data(), scale.data(), dim);
        REQUIRE(distances[i] == expected);
    }
}

TEST_CASE("SQ8 four-way L2 agrees with affine decoding for FP32 and encoded queries",
          "[ut][simd][sq8_tile]") {
    constexpr uint64_t dim = 33;
    std::array<float, dim> lower{}, scale{}, query{};
    std::array<uint8_t, dim> source{};
    std::array<std::array<uint8_t, dim>, 4> target{};
    std::array<const uint8_t*, 4> target_rows{};
    for (uint64_t i = 0; i < dim; ++i) {
        lower[i] = static_cast<float>(i % 9) - 4.0F;
        scale[i] = static_cast<float>(i % 7) * 0.5F;
        query[i] = lower[i] + 0.25F;
        source[i] = static_cast<uint8_t>((19 * i + 23) % 256);
        for (uint64_t j = 0; j < target.size(); ++j) {
            target[j][i] = static_cast<uint8_t>((37 * i + 71 * j) % 256);
        }
    }
    for (uint64_t j = 0; j < target.size(); ++j) {
        target_rows[j] = target[j].data();
    }

    float from_float[4]{}, from_codes[4]{};
    vsag::SQ8ComputeL2SqrBatch4(query.data(),
                                target_rows[0],
                                target_rows[1],
                                target_rows[2],
                                target_rows[3],
                                lower.data(),
                                scale.data(),
                                dim,
                                from_float[0],
                                from_float[1],
                                from_float[2],
                                from_float[3]);
    vsag::SQ8ComputeCodesL2SqrBatch4(
        source.data(), target_rows.data(), lower.data(), scale.data(), dim, from_codes);
    for (uint64_t j = 0; j < target.size(); ++j) {
        double float_ref = 0, code_ref = 0;
        for (uint64_t i = 0; i < dim; ++i) {
            const auto decode = [&](uint8_t byte) {
                return lower[i] + static_cast<double>(byte) * scale[i] / 255.0;
            };
            const double a = query[i] - decode(target[j][i]);
            const double b = decode(source[i]) - decode(target[j][i]);
            float_ref += a * a;
            code_ref += b * b;
        }
        REQUIRE(std::abs(from_float[j] - float_ref) <= 1e-5 * std::max(1.0, float_ref));
        REQUIRE(std::abs(from_codes[j] - code_ref) <= 1e-5 * std::max(1.0, code_ref));
    }
}
