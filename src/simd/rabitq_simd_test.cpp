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

#include <algorithm>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "fp32_simd.h"
#include "simd_status.h"
#include "unittest.h"

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
        if (SimdStatus::SupportSVE()) {                          \
            auto sve = sve::Func(query, base, dim, inv_sqrt_d);  \
            REQUIRE(std::abs(gt - sve) < 1e-4);                  \
        }                                                        \
    }

#define TEST_ACCURACY_SQ4(Func)                                                \
    {                                                                          \
        float gt, avx512;                                                      \
        gt = generic::Func(codes.data(), bits.data(), dim);                    \
        if (SimdStatus::SupportAVX512()) {                                     \
            avx512 = avx512::Func(codes.data(), bits.data(), dim);             \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx512));         \
        }                                                                      \
        if (SimdStatus::SupportAVX512VPOPCNTDQ()) {                            \
            float res = avx512vpopcntdq::Func(codes.data(), bits.data(), dim); \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(res));            \
        }                                                                      \
        if (SimdStatus::SupportSVE()) {                                        \
            float sve = sve::Func(codes.data(), bits.data(), dim);             \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sve));            \
        }                                                                      \
        if (SimdStatus::SupportNEON()) {                                       \
            float neon = neon::Func(codes.data(), bits.data(), dim);           \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(neon));           \
        }                                                                      \
    }

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
    if (SimdStatus::SupportAVX512()) {
        BENCHMARK_SIMD_COMPUTE_SQ4(avx512, RaBitQSQ4UBinaryIP);
    }
    if (SimdStatus::SupportAVX512VPOPCNTDQ()) {
        BENCHMARK_SIMD_COMPUTE_SQ4(avx512vpopcntdq, RaBitQSQ4UBinaryIP);
    }
    if (SimdStatus::SupportSVE()) {
        BENCHMARK_SIMD_COMPUTE_SQ4(sve, RaBitQSQ4UBinaryIP);
    }
    if (SimdStatus::SupportNEON()) {
        BENCHMARK_SIMD_COMPUTE_SQ4(neon, RaBitQSQ4UBinaryIP);
    }
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

TEST_CASE("RaBitQ FP32-BQ SIMD Batch4 Compute Codes", "[ut][simd]") {
    const std::vector<uint64_t> dims = {0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 960};

    for (const auto dim : dims) {
        const uint64_t code_size = (dim + 7) / 8;
        const float inv_sqrt_d = dim == 0 ? 1.0F : 1.0F / std::sqrt(static_cast<float>(dim));
        std::vector<float> query(dim);
        std::vector<uint8_t> bases(std::max<uint64_t>(1, code_size * 4));
        for (uint64_t d = 0; d < dim; ++d) {
            query[d] = static_cast<float>(static_cast<int>(d % 23) - 11) * 0.03125F;
        }
        for (uint64_t i = 0; i < bases.size(); ++i) {
            bases[i] = static_cast<uint8_t>(31U * i + 17U);
        }

        const auto* bits1 = bases.data();
        const auto* bits2 = bits1 + code_size;
        const auto* bits3 = bits2 + code_size;
        const auto* bits4 = bits3 + code_size;
        float expected[4] = {
            generic::RaBitQFloatBinaryIP(query.data(), bits1, dim, inv_sqrt_d),
            generic::RaBitQFloatBinaryIP(query.data(), bits2, dim, inv_sqrt_d),
            generic::RaBitQFloatBinaryIP(query.data(), bits3, dim, inv_sqrt_d),
            generic::RaBitQFloatBinaryIP(query.data(), bits4, dim, inv_sqrt_d),
        };

        auto check_result = [&expected](const float* result) {
            for (uint32_t i = 0; i < 4; ++i) {
                REQUIRE(std::abs(expected[i] - result[i]) < 1e-4F);
            }
        };

        float result[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        generic::RaBitQFloatBinaryIPBatch4(
            query.data(), bits1, bits2, bits3, bits4, dim, inv_sqrt_d, result);
        check_result(result);
        if (SimdStatus::SupportSSE()) {
            sse::RaBitQFloatBinaryIPBatch4(
                query.data(), bits1, bits2, bits3, bits4, dim, inv_sqrt_d, result);
            check_result(result);
        }
        if (SimdStatus::SupportAVX()) {
            avx::RaBitQFloatBinaryIPBatch4(
                query.data(), bits1, bits2, bits3, bits4, dim, inv_sqrt_d, result);
            check_result(result);
        }
        if (SimdStatus::SupportAVX2()) {
            avx2::RaBitQFloatBinaryIPBatch4(
                query.data(), bits1, bits2, bits3, bits4, dim, inv_sqrt_d, result);
            check_result(result);
        }
        if (SimdStatus::SupportAVX512()) {
            avx512::RaBitQFloatBinaryIPBatch4(
                query.data(), bits1, bits2, bits3, bits4, dim, inv_sqrt_d, result);
            check_result(result);
        }
        if (SimdStatus::SupportNEON()) {
            neon::RaBitQFloatBinaryIPBatch4(
                query.data(), bits1, bits2, bits3, bits4, dim, inv_sqrt_d, result);
            check_result(result);
        }
        if (SimdStatus::SupportSVE()) {
            sve::RaBitQFloatBinaryIPBatch4(
                query.data(), bits1, bits2, bits3, bits4, dim, inv_sqrt_d, result);
            check_result(result);
        }
    }
}

TEST_CASE("RaBitQ FP32 three-bit SIMD Batch4 Compute Codes", "[ut][simd]") {
    const std::vector<uint64_t> dims = {0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 960};

    for (const auto dim : dims) {
        const uint64_t plane_bytes = (dim + 7) / 8;
        std::vector<float> query(dim);
        for (uint64_t d = 0; d < dim; ++d) {
            query[d] = static_cast<float>(static_cast<int>(d % 29) - 14) * 0.03125F;
        }

        for (uint32_t reorder_bits : {0U, 1U, 5U}) {
            std::vector<uint8_t> codes(std::max<uint64_t>(1, plane_bytes * 3 * 4));
            for (uint64_t i = 0; i < codes.size(); ++i) {
                codes[i] = static_cast<uint8_t>(43U * i + 13U * reorder_bits + 5U);
            }

            const auto* bits1 = codes.data();
            const auto* bits2 = bits1 + plane_bytes * 3;
            const auto* bits3 = bits2 + plane_bytes * 3;
            const auto* bits4 = bits3 + plane_bytes * 3;
            float expected[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            generic::RaBitQFloatThreeBitIPBatch4(
                query.data(), bits1, bits2, bits3, bits4, dim, reorder_bits, expected);

            auto check_result = [&expected](const float* result) {
                for (uint32_t i = 0; i < 4; ++i) {
                    REQUIRE(std::abs(expected[i] - result[i]) < 1e-4F);
                }
            };

            float result[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            generic::RaBitQFloatThreeBitIPBatch4(
                query.data(), bits1, bits2, bits3, bits4, dim, reorder_bits, result);
            check_result(result);

            std::vector<float> lookup(std::max<uint64_t>(1, plane_bytes * 256));
            generic::RaBitQFloatBuildByteIPLookupTable(query.data(), dim, lookup.data());
            generic::RaBitQFloatThreeBitIPBatch4ByLookup(
                lookup.data(), bits1, bits2, bits3, bits4, dim, reorder_bits, result);
            check_result(result);
            REQUIRE(std::abs(expected[0] - generic::RaBitQFloatThreeBitIPByLookup(
                                               lookup.data(), bits1, dim, reorder_bits)) < 1e-4F);
            REQUIRE(std::abs(expected[1] - generic::RaBitQFloatThreeBitIPByLookup(
                                               lookup.data(), bits2, dim, reorder_bits)) < 1e-4F);
            REQUIRE(std::abs(expected[2] - generic::RaBitQFloatThreeBitIPByLookup(
                                               lookup.data(), bits3, dim, reorder_bits)) < 1e-4F);
            REQUIRE(std::abs(expected[3] - generic::RaBitQFloatThreeBitIPByLookup(
                                               lookup.data(), bits4, dim, reorder_bits)) < 1e-4F);
            RaBitQFloatThreeBitIPBatch4ByLookup(
                lookup.data(), bits1, bits2, bits3, bits4, dim, reorder_bits, result);
            check_result(result);
            REQUIRE(std::abs(expected[0] - RaBitQFloatThreeBitIPByLookup(
                                               lookup.data(), bits1, dim, reorder_bits)) < 1e-4F);
            REQUIRE(std::abs(expected[1] - RaBitQFloatThreeBitIPByLookup(
                                               lookup.data(), bits2, dim, reorder_bits)) < 1e-4F);
            REQUIRE(std::abs(expected[2] - RaBitQFloatThreeBitIPByLookup(
                                               lookup.data(), bits3, dim, reorder_bits)) < 1e-4F);
            REQUIRE(std::abs(expected[3] - RaBitQFloatThreeBitIPByLookup(
                                               lookup.data(), bits4, dim, reorder_bits)) < 1e-4F);
            if (SimdStatus::SupportSSE()) {
                sse::RaBitQFloatThreeBitIPBatch4(
                    query.data(), bits1, bits2, bits3, bits4, dim, reorder_bits, result);
                check_result(result);
            }
            if (SimdStatus::SupportAVX()) {
                avx::RaBitQFloatThreeBitIPBatch4(
                    query.data(), bits1, bits2, bits3, bits4, dim, reorder_bits, result);
                check_result(result);
            }
            if (SimdStatus::SupportAVX2()) {
                avx2::RaBitQFloatThreeBitIPBatch4(
                    query.data(), bits1, bits2, bits3, bits4, dim, reorder_bits, result);
                check_result(result);
                avx2::RaBitQFloatThreeBitIPBatch4ByLookup(
                    lookup.data(), bits1, bits2, bits3, bits4, dim, reorder_bits, result);
                check_result(result);
                REQUIRE(std::abs(expected[0] - avx2::RaBitQFloatThreeBitIPByLookup(
                                                   lookup.data(), bits1, dim, reorder_bits)) <
                        1e-4F);
                REQUIRE(std::abs(expected[1] - avx2::RaBitQFloatThreeBitIPByLookup(
                                                   lookup.data(), bits2, dim, reorder_bits)) <
                        1e-4F);
                REQUIRE(std::abs(expected[2] - avx2::RaBitQFloatThreeBitIPByLookup(
                                                   lookup.data(), bits3, dim, reorder_bits)) <
                        1e-4F);
                REQUIRE(std::abs(expected[3] - avx2::RaBitQFloatThreeBitIPByLookup(
                                                   lookup.data(), bits4, dim, reorder_bits)) <
                        1e-4F);
            }
            if (SimdStatus::SupportAVX512()) {
                avx512::RaBitQFloatThreeBitIPBatch4(
                    query.data(), bits1, bits2, bits3, bits4, dim, reorder_bits, result);
                check_result(result);
                avx512::RaBitQFloatThreeBitIPBatch4ByLookup(
                    lookup.data(), bits1, bits2, bits3, bits4, dim, reorder_bits, result);
                check_result(result);
                REQUIRE(std::abs(expected[0] - avx512::RaBitQFloatThreeBitIPByLookup(
                                                   lookup.data(), bits1, dim, reorder_bits)) <
                        1e-4F);
                REQUIRE(std::abs(expected[1] - avx512::RaBitQFloatThreeBitIPByLookup(
                                                   lookup.data(), bits2, dim, reorder_bits)) <
                        1e-4F);
                REQUIRE(std::abs(expected[2] - avx512::RaBitQFloatThreeBitIPByLookup(
                                                   lookup.data(), bits3, dim, reorder_bits)) <
                        1e-4F);
                REQUIRE(std::abs(expected[3] - avx512::RaBitQFloatThreeBitIPByLookup(
                                                   lookup.data(), bits4, dim, reorder_bits)) <
                        1e-4F);
            }
            if (SimdStatus::SupportNEON()) {
                neon::RaBitQFloatThreeBitIPBatch4(
                    query.data(), bits1, bits2, bits3, bits4, dim, reorder_bits, result);
                check_result(result);
            }
            if (SimdStatus::SupportSVE()) {
                sve::RaBitQFloatThreeBitIPBatch4(
                    query.data(), bits1, bits2, bits3, bits4, dim, reorder_bits, result);
                check_result(result);
            }
        }
    }
}

TEST_CASE("RaBitQ FP32 two-bit centered SIMD Batch4 Compute Codes", "[ut][simd]") {
    const std::vector<uint64_t> dims = {0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 960};

    for (const auto dim : dims) {
        const uint64_t plane_bytes = (dim + 7) / 8;
        std::vector<float> query(dim);
        for (uint64_t d = 0; d < dim; ++d) {
            query[d] = static_cast<float>(static_cast<int>(d % 29) - 14) * 0.03125F;
        }

        std::vector<uint8_t> codes(std::max<uint64_t>(1, plane_bytes * 2 * 4));
        for (uint64_t i = 0; i < codes.size(); ++i) {
            codes[i] = static_cast<uint8_t>(43U * i + 5U);
        }

        const auto* bits1 = codes.data();
        const auto* bits2 = bits1 + plane_bytes * 2;
        const auto* bits3 = bits2 + plane_bytes * 2;
        const auto* bits4 = bits3 + plane_bytes * 2;
        const uint8_t* all_bits[4] = {bits1, bits2, bits3, bits4};

        float expected[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        for (uint64_t d = 0; d < dim; ++d) {
            const uint64_t byte_idx = d >> 3;
            const uint8_t bit_mask = static_cast<uint8_t>(1U << (d & 7));
            for (uint32_t i = 0; i < 4; ++i) {
                const uint8_t* plane0 = all_bits[i];
                const uint8_t* plane1 = all_bits[i] + plane_bytes;
                float weight = (plane0[byte_idx] & bit_mask) != 0U ? 1.0F : -1.0F;
                weight += (plane1[byte_idx] & bit_mask) != 0U ? 0.5F : -0.5F;
                expected[i] += query[d] * weight;
            }
        }

        auto check_result = [&expected](const float* result) {
            for (uint32_t i = 0; i < 4; ++i) {
                REQUIRE(std::abs(expected[i] - result[i]) < 1e-4F);
            }
        };
        auto check_one = [&](auto func, const uint8_t* bits, float expected_value) {
            REQUIRE(std::abs(expected_value - func(query.data(), bits, dim)) < 1e-4F);
        };

        float result[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        generic::RaBitQFloatTwoBitCenteredIPBatch4(
            query.data(), bits1, bits2, bits3, bits4, dim, result);
        check_result(result);
        check_one(generic::RaBitQFloatTwoBitCenteredIP, bits1, expected[0]);
        check_one(generic::RaBitQFloatTwoBitCenteredIP, bits2, expected[1]);
        check_one(generic::RaBitQFloatTwoBitCenteredIP, bits3, expected[2]);
        check_one(generic::RaBitQFloatTwoBitCenteredIP, bits4, expected[3]);

        RaBitQFloatTwoBitCenteredIPBatch4(query.data(), bits1, bits2, bits3, bits4, dim, result);
        check_result(result);
        check_one(RaBitQFloatTwoBitCenteredIP, bits1, expected[0]);
        check_one(RaBitQFloatTwoBitCenteredIP, bits2, expected[1]);
        check_one(RaBitQFloatTwoBitCenteredIP, bits3, expected[2]);
        check_one(RaBitQFloatTwoBitCenteredIP, bits4, expected[3]);

        if (SimdStatus::SupportAVX2()) {
            avx2::RaBitQFloatTwoBitCenteredIPBatch4(
                query.data(), bits1, bits2, bits3, bits4, dim, result);
            check_result(result);
            check_one(avx2::RaBitQFloatTwoBitCenteredIP, bits1, expected[0]);
            check_one(avx2::RaBitQFloatTwoBitCenteredIP, bits2, expected[1]);
            check_one(avx2::RaBitQFloatTwoBitCenteredIP, bits3, expected[2]);
            check_one(avx2::RaBitQFloatTwoBitCenteredIP, bits4, expected[3]);
        }
        if (SimdStatus::SupportAVX512()) {
            avx512::RaBitQFloatTwoBitCenteredIPBatch4(
                query.data(), bits1, bits2, bits3, bits4, dim, result);
            check_result(result);
            check_one(avx512::RaBitQFloatTwoBitCenteredIP, bits1, expected[0]);
            check_one(avx512::RaBitQFloatTwoBitCenteredIP, bits2, expected[1]);
            check_one(avx512::RaBitQFloatTwoBitCenteredIP, bits3, expected[2]);
            check_one(avx512::RaBitQFloatTwoBitCenteredIP, bits4, expected[3]);
        }
    }
}

TEST_CASE("RaBitQ FP32 three-bit centered SIMD Batch4 Compute Codes", "[ut][simd]") {
    const std::vector<uint64_t> dims = {0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 960};

    for (const auto dim : dims) {
        const uint64_t plane_bytes = (dim + 7) / 8;
        std::vector<float> query(dim);
        float query_sum = 0.0F;
        for (uint64_t d = 0; d < dim; ++d) {
            query[d] = static_cast<float>(static_cast<int>(d % 29) - 14) * 0.03125F;
            query_sum += query[d];
        }

        std::vector<uint8_t> codes(std::max<uint64_t>(1, plane_bytes * 3 * 4));
        for (uint64_t i = 0; i < codes.size(); ++i) {
            codes[i] = static_cast<uint8_t>(43U * i + 5U);
        }

        const auto* bits1 = codes.data();
        const auto* bits2 = bits1 + plane_bytes * 3;
        const auto* bits3 = bits2 + plane_bytes * 3;
        const auto* bits4 = bits3 + plane_bytes * 3;

        float unsigned_ip[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        generic::RaBitQFloatThreeBitIPBatch4(
            query.data(), bits1, bits2, bits3, bits4, dim, 0, unsigned_ip);
        float expected[4] = {unsigned_ip[0] - 3.5F * query_sum,
                             unsigned_ip[1] - 3.5F * query_sum,
                             unsigned_ip[2] - 3.5F * query_sum,
                             unsigned_ip[3] - 3.5F * query_sum};

        auto check_result = [&expected](const float* result) {
            for (uint32_t i = 0; i < 4; ++i) {
                REQUIRE(std::abs(expected[i] - result[i]) < 1e-4F);
            }
        };
        auto check_one = [&](auto func, const uint8_t* bits, float expected_value) {
            REQUIRE(std::abs(expected_value - func(query.data(), bits, dim)) < 1e-4F);
        };

        float result[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        generic::RaBitQFloatThreeBitCenteredIPBatch4(
            query.data(), bits1, bits2, bits3, bits4, dim, result);
        check_result(result);
        check_one(generic::RaBitQFloatThreeBitCenteredIP, bits1, expected[0]);
        check_one(generic::RaBitQFloatThreeBitCenteredIP, bits2, expected[1]);
        check_one(generic::RaBitQFloatThreeBitCenteredIP, bits3, expected[2]);
        check_one(generic::RaBitQFloatThreeBitCenteredIP, bits4, expected[3]);

        RaBitQFloatThreeBitCenteredIPBatch4(query.data(), bits1, bits2, bits3, bits4, dim, result);
        check_result(result);
        check_one(RaBitQFloatThreeBitCenteredIP, bits1, expected[0]);
        check_one(RaBitQFloatThreeBitCenteredIP, bits2, expected[1]);
        check_one(RaBitQFloatThreeBitCenteredIP, bits3, expected[2]);
        check_one(RaBitQFloatThreeBitCenteredIP, bits4, expected[3]);

        if (SimdStatus::SupportAVX2()) {
            avx2::RaBitQFloatThreeBitCenteredIPBatch4(
                query.data(), bits1, bits2, bits3, bits4, dim, result);
            check_result(result);
            check_one(avx2::RaBitQFloatThreeBitCenteredIP, bits1, expected[0]);
            check_one(avx2::RaBitQFloatThreeBitCenteredIP, bits2, expected[1]);
            check_one(avx2::RaBitQFloatThreeBitCenteredIP, bits3, expected[2]);
            check_one(avx2::RaBitQFloatThreeBitCenteredIP, bits4, expected[3]);
        }
        if (SimdStatus::SupportAVX512()) {
            avx512::RaBitQFloatThreeBitCenteredIPBatch4(
                query.data(), bits1, bits2, bits3, bits4, dim, result);
            check_result(result);
            check_one(avx512::RaBitQFloatThreeBitCenteredIP, bits1, expected[0]);
            check_one(avx512::RaBitQFloatThreeBitCenteredIP, bits2, expected[1]);
            check_one(avx512::RaBitQFloatThreeBitCenteredIP, bits3, expected[2]);
            check_one(avx512::RaBitQFloatThreeBitCenteredIP, bits4, expected[3]);
        }
    }
}

TEST_CASE("RaBitQ FP32 multi-bit lookup SIMD Batch4 Compute Codes", "[ut][simd]") {
    constexpr float kLookupTolerance = 1e-3F;
    const std::vector<uint64_t> dims = {0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 960};

    for (const auto dim : dims) {
        const uint64_t plane_bytes = (dim + 7) / 8;
        std::vector<float> query(dim);
        for (uint64_t d = 0; d < dim; ++d) {
            query[d] = static_cast<float>(static_cast<int>(d % 31) - 15) * 0.02625F;
        }

        for (uint32_t filter_bits : {2U, 3U}) {
            for (uint32_t reorder_bits : {0U, 1U, 5U}) {
                std::vector<uint8_t> codes(std::max<uint64_t>(1, plane_bytes * filter_bits * 4));
                for (uint64_t i = 0; i < codes.size(); ++i) {
                    codes[i] =
                        static_cast<uint8_t>(41U * i + 17U * reorder_bits + 7U * filter_bits);
                }

                const auto record_bytes = plane_bytes * filter_bits;
                const auto* bits1 = codes.data();
                const auto* bits2 = bits1 + record_bytes;
                const auto* bits3 = bits2 + record_bytes;
                const auto* bits4 = bits3 + record_bytes;

                auto compute_expected = [&](const uint8_t* bits) {
                    float result = 0.0F;
                    for (uint64_t d = 0; d < dim; ++d) {
                        const uint64_t byte_idx = d >> 3;
                        const uint8_t bit_mask = static_cast<uint8_t>(1U << (d & 7));
                        uint32_t code = 0;
                        for (uint32_t bit = 0; bit < filter_bits; ++bit) {
                            const auto* plane = bits + static_cast<uint64_t>(bit) * plane_bytes;
                            if ((plane[byte_idx] & bit_mask) != 0U) {
                                code += 1U << (reorder_bits + filter_bits - bit - 1);
                            }
                        }
                        result += query[d] * static_cast<float>(code);
                    }
                    return result;
                };

                float expected[4] = {compute_expected(bits1),
                                     compute_expected(bits2),
                                     compute_expected(bits3),
                                     compute_expected(bits4)};
                auto check_result = [&expected](const float* result) {
                    for (uint32_t i = 0; i < 4; ++i) {
                        REQUIRE(std::abs(expected[i] - result[i]) < kLookupTolerance);
                    }
                };

                std::vector<float> lookup(std::max<uint64_t>(1, plane_bytes * 256));
                generic::RaBitQFloatBuildByteIPLookupTable(query.data(), dim, lookup.data());
                auto check_one = [&](auto func, const uint8_t* bits, float expected_value) {
                    REQUIRE(std::abs(expected_value -
                                     func(lookup.data(), bits, dim, reorder_bits, filter_bits)) <
                            kLookupTolerance);
                };

                float result[4] = {0.0F, 0.0F, 0.0F, 0.0F};
                generic::RaBitQFloatMultiBitIPBatch4ByLookup(lookup.data(),
                                                             bits1,
                                                             bits2,
                                                             bits3,
                                                             bits4,
                                                             dim,
                                                             reorder_bits,
                                                             filter_bits,
                                                             result);
                check_result(result);
                check_one(generic::RaBitQFloatMultiBitIPByLookup, bits1, expected[0]);
                check_one(generic::RaBitQFloatMultiBitIPByLookup, bits2, expected[1]);
                check_one(generic::RaBitQFloatMultiBitIPByLookup, bits3, expected[2]);
                check_one(generic::RaBitQFloatMultiBitIPByLookup, bits4, expected[3]);

                RaBitQFloatMultiBitIPBatch4ByLookup(lookup.data(),
                                                    bits1,
                                                    bits2,
                                                    bits3,
                                                    bits4,
                                                    dim,
                                                    reorder_bits,
                                                    filter_bits,
                                                    result);
                check_result(result);
                check_one(RaBitQFloatMultiBitIPByLookup, bits1, expected[0]);
                check_one(RaBitQFloatMultiBitIPByLookup, bits2, expected[1]);
                check_one(RaBitQFloatMultiBitIPByLookup, bits3, expected[2]);
                check_one(RaBitQFloatMultiBitIPByLookup, bits4, expected[3]);

                if (SimdStatus::SupportAVX2()) {
                    avx2::RaBitQFloatMultiBitIPBatch4ByLookup(lookup.data(),
                                                              bits1,
                                                              bits2,
                                                              bits3,
                                                              bits4,
                                                              dim,
                                                              reorder_bits,
                                                              filter_bits,
                                                              result);
                    check_result(result);
                    check_one(avx2::RaBitQFloatMultiBitIPByLookup, bits1, expected[0]);
                    check_one(avx2::RaBitQFloatMultiBitIPByLookup, bits2, expected[1]);
                    check_one(avx2::RaBitQFloatMultiBitIPByLookup, bits3, expected[2]);
                    check_one(avx2::RaBitQFloatMultiBitIPByLookup, bits4, expected[3]);
                }
                if (SimdStatus::SupportAVX512()) {
                    avx512::RaBitQFloatMultiBitIPBatch4ByLookup(lookup.data(),
                                                                bits1,
                                                                bits2,
                                                                bits3,
                                                                bits4,
                                                                dim,
                                                                reorder_bits,
                                                                filter_bits,
                                                                result);
                    check_result(result);
                    check_one(avx512::RaBitQFloatMultiBitIPByLookup, bits1, expected[0]);
                    check_one(avx512::RaBitQFloatMultiBitIPByLookup, bits2, expected[1]);
                    check_one(avx512::RaBitQFloatMultiBitIPByLookup, bits3, expected[2]);
                    check_one(avx512::RaBitQFloatMultiBitIPByLookup, bits4, expected[3]);
                }
            }
        }
    }
}

TEST_CASE("RaBitQ FP32 split-code SIMD Compute Codes", "[ut][simd]") {
    const std::vector<uint64_t> dims = {0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 960};

    for (auto dim : dims) {
        const auto plane_bytes = (dim + 7) / 8;
        std::vector<float> query(dim);
        for (uint64_t d = 0; d < dim; ++d) {
            query[d] = static_cast<float>(static_cast<int>(d % 17) - 8) * 0.03125F;
        }

        for (uint32_t supplement_bits = 0; supplement_bits <= 7; ++supplement_bits) {
            std::vector<uint8_t> one_bit_code(plane_bytes);
            std::vector<uint8_t> supplement_code(
                std::max<uint64_t>(1, plane_bytes * supplement_bits));
            for (uint64_t i = 0; i < one_bit_code.size(); ++i) {
                one_bit_code[i] = static_cast<uint8_t>(37U * i + 11U * supplement_bits + 3U);
            }
            for (uint64_t i = 0; i < supplement_code.size(); ++i) {
                supplement_code[i] = static_cast<uint8_t>(53U * i + 7U * supplement_bits + 19U);
            }

            float expected = 0.0F;
            const uint32_t one_bit_weight = 1U << supplement_bits;
            for (uint64_t d = 0; d < dim; ++d) {
                const auto byte_idx = d >> 3;
                const auto bit_mask = static_cast<uint8_t>(1U << (d & 7));
                uint32_t code = (one_bit_code[byte_idx] & bit_mask) != 0 ? one_bit_weight : 0U;
                for (uint32_t bit = 0; bit < supplement_bits; ++bit) {
                    const auto* plane = supplement_code.data() + uint64_t(bit) * plane_bytes;
                    if ((plane[byte_idx] & bit_mask) != 0) {
                        code += 1U << bit;
                    }
                }
                expected += query[d] * static_cast<float>(code);
            }

            const auto* supplement = supplement_bits == 0 ? nullptr : supplement_code.data();
            auto generic_result = generic::RaBitQFloatSplitCodeIP(
                query.data(), one_bit_code.data(), supplement, dim, supplement_bits);
            REQUIRE(std::abs(expected - generic_result) < 1e-4F);
            if (SimdStatus::SupportSSE()) {
                auto sse_result = sse::RaBitQFloatSplitCodeIP(
                    query.data(), one_bit_code.data(), supplement, dim, supplement_bits);
                REQUIRE(std::abs(expected - sse_result) < 1e-4F);
            }
            if (SimdStatus::SupportAVX()) {
                auto avx_result = avx::RaBitQFloatSplitCodeIP(
                    query.data(), one_bit_code.data(), supplement, dim, supplement_bits);
                REQUIRE(std::abs(expected - avx_result) < 1e-4F);
            }
            if (SimdStatus::SupportAVX2()) {
                auto avx2_result = avx2::RaBitQFloatSplitCodeIP(
                    query.data(), one_bit_code.data(), supplement, dim, supplement_bits);
                REQUIRE(std::abs(expected - avx2_result) < 1e-4F);
            }
            if (SimdStatus::SupportAVX512()) {
                auto avx512_result = avx512::RaBitQFloatSplitCodeIP(
                    query.data(), one_bit_code.data(), supplement, dim, supplement_bits);
                REQUIRE(std::abs(expected - avx512_result) < 1e-4F);
            }
            if (SimdStatus::SupportNEON()) {
                auto neon_result = neon::RaBitQFloatSplitCodeIP(
                    query.data(), one_bit_code.data(), supplement, dim, supplement_bits);
                REQUIRE(std::abs(expected - neon_result) < 1e-4F);
            }
            if (SimdStatus::SupportSVE()) {
                auto sve_result = sve::RaBitQFloatSplitCodeIP(
                    query.data(), one_bit_code.data(), supplement, dim, supplement_bits);
                REQUIRE(std::abs(expected - sve_result) < 1e-4F);
            }
        }
    }
}

TEST_CASE("RaBitQ FP32 supplement-code SIMD Compute Codes", "[ut][simd]") {
    const std::vector<uint64_t> dims = {0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 960};

    for (auto dim : dims) {
        const auto plane_bytes = (dim + 7) / 8;
        std::vector<float> query(dim);
        for (uint64_t d = 0; d < dim; ++d) {
            query[d] = static_cast<float>(static_cast<int>(d % 19) - 9) * 0.03125F;
        }

        for (uint32_t supplement_bits = 0; supplement_bits <= 7; ++supplement_bits) {
            std::vector<uint8_t> supplement_code(
                std::max<uint64_t>(1, plane_bytes * supplement_bits));
            for (uint64_t i = 0; i < supplement_code.size(); ++i) {
                supplement_code[i] = static_cast<uint8_t>(59U * i + 3U * supplement_bits + 23U);
            }

            float expected = 0.0F;
            for (uint64_t d = 0; d < dim; ++d) {
                const auto byte_idx = d >> 3;
                const auto bit_mask = static_cast<uint8_t>(1U << (d & 7));
                uint32_t code = 0;
                for (uint32_t bit = 0; bit < supplement_bits; ++bit) {
                    const auto* plane = supplement_code.data() + uint64_t(bit) * plane_bytes;
                    if ((plane[byte_idx] & bit_mask) != 0) {
                        code += 1U << bit;
                    }
                }
                expected += query[d] * static_cast<float>(code);
            }

            const auto* supplement = supplement_bits == 0 ? nullptr : supplement_code.data();
            auto generic_result = generic::RaBitQFloatSupplementCodeIP(
                query.data(), supplement, dim, supplement_bits);
            REQUIRE(std::abs(expected - generic_result) < 1e-4F);
            if (SimdStatus::SupportSSE()) {
                auto sse_result = sse::RaBitQFloatSupplementCodeIP(
                    query.data(), supplement, dim, supplement_bits);
                REQUIRE(std::abs(expected - sse_result) < 1e-4F);
            }
            if (SimdStatus::SupportAVX()) {
                auto avx_result = avx::RaBitQFloatSupplementCodeIP(
                    query.data(), supplement, dim, supplement_bits);
                REQUIRE(std::abs(expected - avx_result) < 1e-4F);
            }
            if (SimdStatus::SupportAVX2()) {
                auto avx2_result = avx2::RaBitQFloatSupplementCodeIP(
                    query.data(), supplement, dim, supplement_bits);
                REQUIRE(std::abs(expected - avx2_result) < 1e-4F);
            }
            if (SimdStatus::SupportAVX512()) {
                auto avx512_result = avx512::RaBitQFloatSupplementCodeIP(
                    query.data(), supplement, dim, supplement_bits);
                REQUIRE(std::abs(expected - avx512_result) < 1e-4F);
            }
            if (SimdStatus::SupportNEON()) {
                auto neon_result = neon::RaBitQFloatSupplementCodeIP(
                    query.data(), supplement, dim, supplement_bits);
                REQUIRE(std::abs(expected - neon_result) < 1e-4F);
            }
            if (SimdStatus::SupportSVE()) {
                auto sve_result = sve::RaBitQFloatSupplementCodeIP(
                    query.data(), supplement, dim, supplement_bits);
                REQUIRE(std::abs(expected - sve_result) < 1e-4F);
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
    if (SimdStatus::SupportSSE()) {
        BENCHMARK_SIMD_COMPUTE(sse, RaBitQFloatBinaryIP);
    }
    if (SimdStatus::SupportAVX()) {
        BENCHMARK_SIMD_COMPUTE(avx, RaBitQFloatBinaryIP);
    }
    if (SimdStatus::SupportAVX2()) {
        BENCHMARK_SIMD_COMPUTE(avx2, RaBitQFloatBinaryIP);
    }
    if (SimdStatus::SupportAVX512()) {
        BENCHMARK_SIMD_COMPUTE(avx512, RaBitQFloatBinaryIP);
    }
    if (SimdStatus::SupportNEON()) {
        BENCHMARK_SIMD_COMPUTE(neon, RaBitQFloatBinaryIP);
    }
    if (SimdStatus::SupportSVE()) {
        BENCHMARK_SIMD_COMPUTE(sve, RaBitQFloatBinaryIP);
    }
}

TEST_CASE("SIMD test for rescale", "[ut][simd]") {
    auto dims = fixtures::get_common_used_dims();
    int64_t count = 100;

    for (const auto& dim : dims) {
        std::vector<float> gt = fixtures::GenerateVectors<float>(count, dim);

        std::vector<float> avx512_datas(gt.size());
        avx512_datas.assign(gt.begin(), gt.end());
        std::vector<float> avx2_datas(gt.size());
        avx2_datas.assign(gt.begin(), gt.end());
        std::vector<float> avx_datas(gt.size());
        avx_datas.assign(gt.begin(), gt.end());
        std::vector<float> sse_datas(gt.size());
        sse_datas.assign(gt.begin(), gt.end());

        std::vector<float> neon_datas(gt.size());
        neon_datas.assign(gt.begin(), gt.end());
        std::vector<float> sve_datas(gt.size());
        sve_datas.assign(gt.begin(), gt.end());

        for (int i = 0; i < count; i++) {
            auto* gt_data = gt.data() + i * dim;
            auto* avx512_data = avx512_datas.data() + i * dim;
            auto* avx2_data = avx2_datas.data() + i * dim;
            auto* avx_data = avx_datas.data() + i * dim;
            auto* sse_data = sse_datas.data() + i * dim;
            auto* neon_data = neon_datas.data() + i * dim;
            auto* sve_data = sve_datas.data() + i * dim;

            const float delta = 1e-5;
            generic::VecRescale(gt_data, dim, 0.5);

            if (SimdStatus::SupportAVX512()) {
                avx512::VecRescale(avx512_data, dim, 0.5);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - avx512_data[j]) < delta);
                }
            }
            if (SimdStatus::SupportAVX2()) {
                avx2::VecRescale(avx2_data, dim, 0.5);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - avx2_data[j]) < delta);
                }
            }
            if (SimdStatus::SupportAVX()) {
                avx::VecRescale(avx_data, dim, 0.5);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - avx_data[j]) < delta);
                }
            }
            if (SimdStatus::SupportSSE()) {
                sse::VecRescale(sse_data, dim, 0.5);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - sse_data[j]) < delta);
                }
            }
            if (SimdStatus::SupportNEON()) {
                neon::VecRescale(neon_data, dim, 0.5);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - neon_data[j]) < delta);
                }
            }
            if (SimdStatus::SupportSVE()) {
                sve::VecRescale(sve_data, dim, 0.5);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - sve_data[j]) < delta);
                }
            }
        }
    }
}

TEST_CASE("SIMD test for kacs_walk", "[ut][simd]") {
    auto dims = fixtures::get_common_used_dims();
    int64_t count = 100;

    for (const auto& dim : dims) {
        std::vector<float> gt = fixtures::GenerateVectors<float>(count, dim);

        std::vector<float> avx512_datas(gt.size());
        avx512_datas.assign(gt.begin(), gt.end());
        std::vector<float> avx2_datas(gt.size());
        avx2_datas.assign(gt.begin(), gt.end());
        std::vector<float> avx_datas(gt.size());
        avx_datas.assign(gt.begin(), gt.end());
        std::vector<float> sse_datas(gt.size());
        sse_datas.assign(gt.begin(), gt.end());

        std::vector<float> neon_datas(gt.size());
        neon_datas.assign(gt.begin(), gt.end());
        std::vector<float> sve_datas(gt.size());
        sve_datas.assign(gt.begin(), gt.end());

        const float delta = 1e-5;
        for (int i = 0; i < count; i++) {
            auto* gt_data = gt.data() + i * dim;
            generic::KacsWalk(gt_data, dim);

            if (SimdStatus::SupportAVX512()) {
                auto* avx512_data = avx512_datas.data() + i * dim;
                avx512::KacsWalk(avx512_data, dim);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - avx512_data[j]) < delta);
                }
            }
            if (SimdStatus::SupportAVX2()) {
                auto* avx2_data = avx2_datas.data() + i * dim;
                avx2::KacsWalk(avx2_data, dim);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - avx2_data[j]) < delta);
                }
            }
            if (SimdStatus::SupportAVX()) {
                auto* avx_data = avx_datas.data() + i * dim;
                avx::KacsWalk(avx_data, dim);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - avx_data[j]) < delta);
                }
            }
            if (SimdStatus::SupportSSE()) {
                auto* sse_data = sse_datas.data() + i * dim;
                sse::KacsWalk(sse_data, dim);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - sse_data[j]) < delta);
                }
            }
            if (SimdStatus::SupportNEON()) {
                auto* neon_data = neon_datas.data() + i * dim;
                neon::KacsWalk(neon_data, dim);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - neon_data[j]) < delta);
                }
            }
            if (SimdStatus::SupportSVE()) {
                auto* sve_data = sve_datas.data() + i * dim;
                sve::KacsWalk(sve_data, dim);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - sve_data[j]) < delta);
                }
            }
        }
    }
}

TEST_CASE("SIMD test for rotate", "[ut][simd]") {
    auto dims = fixtures::get_common_used_dims();
    int64_t count = 100;

    for (const auto& dim : dims) {
        std::vector<float> gt = fixtures::GenerateVectors<float>(count, dim);

        std::vector<float> avx512_datas(gt.size());
        avx512_datas.assign(gt.begin(), gt.end());
        std::vector<float> avx2_datas(gt.size());
        avx2_datas.assign(gt.begin(), gt.end());
        std::vector<float> avx_datas(gt.size());
        avx_datas.assign(gt.begin(), gt.end());
        std::vector<float> sse_datas(gt.size());
        sse_datas.assign(gt.begin(), gt.end());

        std::vector<float> neon_datas(gt.size());
        neon_datas.assign(gt.begin(), gt.end());
        std::vector<float> sve_datas(gt.size());
        sve_datas.assign(gt.begin(), gt.end());

        const float delta = 1e-5;
        for (int i = 0; i < count; i++) {
            auto* gt_data = gt.data() + i * dim;
            uint64_t tmp_dim = dim;
            uint64_t ret = 0;
            while (tmp_dim > 1) {
                ret++;
                tmp_dim >>= 1;
            }
            uint64_t trunc_dim = 1 << ret;
            int start = dim - trunc_dim;
            generic::FHTRotate(gt_data, trunc_dim);
            generic::FHTRotate(gt_data + start, trunc_dim);

            if (SimdStatus::SupportAVX512()) {
                auto* avx512_data = avx512_datas.data() + i * dim;
                avx512::FHTRotate(avx512_data, trunc_dim);
                avx512::FHTRotate(avx512_data + start, trunc_dim);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - avx512_data[j]) < delta);
                }
            }
            if (SimdStatus::SupportAVX2()) {
                auto* avx2_data = avx2_datas.data() + i * dim;
                avx2::FHTRotate(avx2_data, trunc_dim);
                avx2::FHTRotate(avx2_data + start, trunc_dim);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - avx2_data[j]) < delta);
                }
            }
            if (SimdStatus::SupportAVX()) {
                auto* avx_data = avx_datas.data() + i * dim;
                avx::FHTRotate(avx_data, trunc_dim);
                avx::FHTRotate(avx_data + start, trunc_dim);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - avx_data[j]) < delta);
                }
            }
            if (SimdStatus::SupportSSE()) {
                auto* sse_data = sse_datas.data() + i * dim;
                sse::FHTRotate(sse_data, trunc_dim);
                sse::FHTRotate(sse_data + start, trunc_dim);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - sse_data[j]) < delta);
                }
            }
            if (SimdStatus::SupportNEON()) {
                auto* neon_data = neon_datas.data() + i * dim;
                neon::FHTRotate(neon_data, trunc_dim);
                neon::FHTRotate(neon_data + start, trunc_dim);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - neon_data[j]) < delta);
                }
            }
            if (SimdStatus::SupportSVE()) {
                auto* sve_data = sve_datas.data() + i * dim;
                sve::FHTRotate(sve_data, trunc_dim);
                sve::FHTRotate(sve_data + start, trunc_dim);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - sve_data[j]) < delta);
                }
            }
        }
    }
}

TEST_CASE("SIMD test for flip_sign", "[ut][simd]") {
    auto dims = fixtures::get_common_used_dims();
    int64_t count = 100;

    for (const auto& dim : dims) {
        uint32_t flip_size = (dim + 7) / 8;
        std::vector<float> gt = fixtures::GenerateVectors<float>(count, dim);
        std::vector<uint8_t> flips = fixtures::GenerateVectors<uint8_t>(count, flip_size);

        std::vector<float> sse_datas(gt.size());
        sse_datas.assign(gt.begin(), gt.end());
        std::vector<float> avx_datas(gt.size());
        avx_datas.assign(gt.begin(), gt.end());
        std::vector<float> avx2_datas(gt.size());
        avx2_datas.assign(gt.begin(), gt.end());
        std::vector<float> avx512_datas(gt.size());
        avx512_datas.assign(gt.begin(), gt.end());
        std::vector<float> neon_datas(gt.size());
        neon_datas.assign(gt.begin(), gt.end());
        std::vector<float> sve_datas(gt.size());
        sve_datas.assign(gt.begin(), gt.end());

        const float delta = 1e-5;
        for (int i = 0; i < count; i++) {
            auto* gt_data = gt.data() + i * dim;
            auto* flip = flips.data() + i * flip_size;

            generic::FlipSign(flip, gt_data, dim);

            if (SimdStatus::SupportAVX512()) {
                auto* avx512_data = avx512_datas.data() + i * dim;
                avx512::FlipSign(flip, avx512_data, dim);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - avx512_data[j]) < delta);
                }
            }
            if (SimdStatus::SupportNEON()) {
                auto* neon_data = neon_datas.data() + i * dim;
                neon::FlipSign(flip, neon_data, dim);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - neon_data[j]) < delta);
                }
            }
            if (SimdStatus::SupportSVE()) {
                auto* sve_data = sve_datas.data() + i * dim;
                sve::FlipSign(flip, sve_data, dim);
                for (int j = 0; j < dim; j++) {
                    REQUIRE(std::abs(gt_data[j] - sve_data[j]) < delta);
                }
            }
        }
    }
}

#define BENCHMARK_SIMD_FLIP_SIGN(Simd, Func)                                       \
    BENCHMARK_ADVANCED(#Simd #Func) {                                              \
        for (int i = 0; i < count; ++i) {                                          \
            Simd::Func(flips.data() + i * flip_size, datas.data() + i * dim, dim); \
        }                                                                          \
        return;                                                                    \
    }

TEST_CASE("SIMD FlipSign Benchmark", "[ut][simd][!benchmark]") {
    int64_t count = 100;
    int64_t dim = 256;
    uint32_t flip_size = (dim + 7) / 8;

    std::vector<float> datas = fixtures::GenerateVectors<float>(count, dim);
    std::vector<uint8_t> flips = fixtures::GenerateVectors<uint8_t>(count, flip_size);

    BENCHMARK_SIMD_FLIP_SIGN(generic, FlipSign);
    if (SimdStatus::SupportAVX512()) {
        BENCHMARK_SIMD_FLIP_SIGN(avx512, FlipSign);
    }
    if (SimdStatus::SupportNEON()) {
        BENCHMARK_SIMD_FLIP_SIGN(neon, FlipSign);
    }
    if (SimdStatus::SupportSVE()) {
        BENCHMARK_SIMD_FLIP_SIGN(sve, FlipSign);
    }
}

TEST_CASE("SIMD FlipSign Correctness with Patterns", "[ut][simd]") {
    struct TestCase {
        std::vector<uint8_t> flip_pattern;
        std::string description;
    };

    std::vector<TestCase> test_cases = {{{0xFF, 0xFF, 0xFF, 0xFF}, "All bits set - flip all signs"},
                                        {{0x00, 0x00, 0x00, 0x00}, "No bits set - no flips"},
                                        {{0xAA, 0xAA, 0xAA, 0xAA}, "Alternating pattern 10101010"},
                                        {{0x55, 0x55, 0x55, 0x55}, "Alternating pattern 01010101"},
                                        {{0x0F, 0x0F, 0x0F, 0x0F}, "Lower nibble set"},
                                        {{0xF0, 0xF0, 0xF0, 0xF0}, "Upper nibble set"}};

    int64_t dim = 32;
    const float delta = 1e-6;

    for (const auto& test : test_cases) {
        SECTION(test.description) {
            std::vector<float> original_data(dim);
            for (int i = 0; i < dim; i++) {
                original_data[i] = static_cast<float>(i + 1);
            }

            std::vector<float> gt_data(original_data);
            std::vector<float> sse_data(original_data);
            std::vector<float> avx_data(original_data);
            std::vector<float> avx2_data(original_data);
            std::vector<float> avx512_data(original_data);
            std::vector<float> neon_data(original_data);
            std::vector<float> sve_data(original_data);

            generic::FlipSign(test.flip_pattern.data(), gt_data.data(), dim);

            if (SimdStatus::SupportAVX512()) {
                avx512::FlipSign(test.flip_pattern.data(), avx512_data.data(), dim);
                for (int i = 0; i < dim; i++) {
                    REQUIRE(std::abs(gt_data[i] - avx512_data[i]) < delta);
                }
            }
            if (SimdStatus::SupportNEON()) {
                neon::FlipSign(test.flip_pattern.data(), neon_data.data(), dim);
                for (int i = 0; i < dim; i++) {
                    REQUIRE(std::abs(gt_data[i] - neon_data[i]) < delta);
                }
            }
            if (SimdStatus::SupportSVE()) {
                sve::FlipSign(test.flip_pattern.data(), sve_data.data(), dim);
                for (int i = 0; i < dim; i++) {
                    REQUIRE(std::abs(gt_data[i] - sve_data[i]) < delta);
                }
            }

            if (test.flip_pattern[0] == 0xFF) {
                for (int i = 0; i < dim; i++) {
                    REQUIRE(gt_data[i] == -original_data[i]);
                }
            } else if (test.flip_pattern[0] == 0x00) {
                for (int i = 0; i < dim; i++) {
                    REQUIRE(gt_data[i] == original_data[i]);
                }
            }
        }
    }
}
