
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

#include <algorithm>
#include <array>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>

#include "simd_status.h"
#include "unittest.h"

using namespace vsag;

namespace {

bool
has_nan_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0U;
}

std::array<int32_t, 32>
high_acc_scalar_overwrite(const uint8_t* low_lookup_table,
                          const uint8_t* high_lookup_table,
                          const uint8_t* codes,
                          uint64_t pq_dim) {
    std::array<int32_t, 32> result{};
    for (uint64_t group = 0; group < pq_dim; ++group) {
        const auto* low_dict = low_lookup_table + group * 16;
        const auto* high_dict = high_lookup_table + group * 16;
        const auto* group_codes = codes + group * 16;
        for (uint64_t byte_index = 0; byte_index < 16; ++byte_index) {
            const uint8_t low_code = group_codes[byte_index] & 0x0FU;
            const uint8_t high_code = group_codes[byte_index] >> 4U;
            const int32_t low_value = static_cast<int32_t>(low_dict[low_code]) |
                                      (static_cast<int32_t>(high_dict[low_code]) << 8U);
            const int32_t high_value = static_cast<int32_t>(low_dict[high_code]) |
                                       (static_cast<int32_t>(high_dict[high_code]) << 8U);
            const uint64_t lane = byte_index / 2 + (byte_index % 2) * 8;
            result[lane] += low_value;
            result[16 + lane] += high_value;
        }
    }
    return result;
}

}  // namespace

template <class T>
bool
compare_vector(std::vector<T>& v1, std::vector<T>& v2) {
    if (v1.size() != v2.size()) {
        return false;
    }
    for (uint64_t i = 0; i < v1.size(); ++i) {
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

TEST_CASE("PQFastScan SIMD Compute High Dim", "[ut][simd]") {
    // pq_dim beyond ~258 overflows the int16 accumulators in the SIMD kernels;
    // generic accumulates in int32 and stays exact, so it is the reference.
    const std::vector<int64_t> dims = {2048, 4096, 8192};
    int64_t count = 10;
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

TEST_CASE("PQFastScan HighAcc SIMD parity and tail groups", "[ut][simd]") {
    const std::vector<uint64_t> group_counts = {0, 1, 2, 3, 4, 5, 240, 241, 2048};
    for (const uint64_t pq_dim : group_counts) {
        const uint64_t data_size = pq_dim * 16;
        auto low_lut = fixtures::generate_uint8_codes(1, data_size, fixtures::RandomValue(0, 1234));
        auto high_lut =
            fixtures::generate_uint8_codes(1, data_size, fixtures::RandomValue(0, 5678));
        auto codes = fixtures::generate_uint8_codes(1, data_size, fixtures::RandomValue(0, 9012));
        std::vector<int32_t> initial(32);
        for (uint64_t lane = 0; lane < initial.size(); ++lane) {
            initial[lane] = static_cast<int32_t>(lane * 17);
        }
        auto expected = initial;
        generic::PQFastScanLookUp32HighAcc(
            low_lut.data(), high_lut.data(), codes.data(), pq_dim, expected.data());

        auto verify = [&](auto lookup) {
            auto actual = initial;
            lookup(low_lut.data(), high_lut.data(), codes.data(), pq_dim, actual.data());
            INFO("pq_dim=" << pq_dim);
            REQUIRE(compare_vector(expected, actual));
        };

        verify(PQFastScanLookUp32HighAcc);
        if (SimdStatus::SupportSSE()) {
            verify(sse::PQFastScanLookUp32HighAcc);
        }
        if (SimdStatus::SupportAVX()) {
            verify(avx::PQFastScanLookUp32HighAcc);
        }
        if (SimdStatus::SupportAVX2()) {
            verify(avx2::PQFastScanLookUp32HighAcc);
        }
        if (SimdStatus::SupportAVX512()) {
            verify(avx512::PQFastScanLookUp32HighAcc);
        }
        if (SimdStatus::SupportNEON()) {
            verify(neon::PQFastScanLookUp32HighAcc);
        }
        if (SimdStatus::SupportSVE()) {
            verify(sve::PQFastScanLookUp32HighAcc);
        }
    }
}

TEST_CASE("PQFastScan HighAcc overwrite SIMD parity and safe chunk tails", "[ut][simd]") {
    const std::vector<uint64_t> group_counts = {
        0, 1, 2, 3, 5, 240, 241, 255, 256, 257, 259, 2048, 8192};
    for (const uint64_t pq_dim : group_counts) {
        const uint64_t data_size = pq_dim * 16;
        std::vector<uint8_t> low_lut(data_size);
        std::vector<uint8_t> high_lut(data_size);
        std::vector<uint8_t> codes(data_size);
        for (uint64_t i = 0; i < data_size; ++i) {
            low_lut[i] = static_cast<uint8_t>((i * 17U + 3U) & 0xFFU);
            high_lut[i] = static_cast<uint8_t>((i * 29U + 11U) & 0xFFU);
            codes[i] = static_cast<uint8_t>((i * 43U + 7U) & 0xFFU);
        }
        if (pq_dim >= 2048) {
            std::fill(low_lut.begin(), low_lut.end(), 0xFFU);
            std::fill(high_lut.begin(), high_lut.end(), 0xFFU);
        }
        const auto expected =
            high_acc_scalar_overwrite(low_lut.data(), high_lut.data(), codes.data(), pq_dim);

        auto verify = [&](auto lookup) {
            std::array<int32_t, 32> actual{};
            for (uint64_t lane = 0; lane < actual.size(); ++lane) {
                actual[lane] = -1234567 + static_cast<int32_t>(lane * 101U);
            }
            lookup(low_lut.data(), high_lut.data(), codes.data(), pq_dim, actual.data());
            INFO("pq_dim=" << pq_dim);
            REQUIRE(actual == expected);
        };

        verify(generic::PQFastScanLookUp32HighAccOverwrite);
        verify(PQFastScanLookUp32HighAccOverwrite);
        if (SimdStatus::SupportSSE()) {
            verify(sse::PQFastScanLookUp32HighAccOverwrite);
        }
        if (SimdStatus::SupportAVX()) {
            verify(avx::PQFastScanLookUp32HighAccOverwrite);
        }
        if (SimdStatus::SupportAVX2()) {
            verify(avx2::PQFastScanLookUp32HighAccOverwrite);
        }
        if (SimdStatus::SupportAVX512()) {
            verify(avx512::PQFastScanLookUp32HighAccOverwrite);
        }
        if (SimdStatus::SupportNEON()) {
            verify(neon::PQFastScanLookUp32HighAccOverwrite);
        }
        if (SimdStatus::SupportSVE()) {
            verify(sve::PQFastScanLookUp32HighAccOverwrite);
        }
    }
}

TEST_CASE("PQFastScan HighAcc LUT byte order and lane layout", "[ut][simd]") {
    std::array<uint8_t, 16> low_lut{};
    std::array<uint8_t, 16> high_lut{};
    std::array<uint8_t, 16> codes{};
    for (uint64_t value = 0; value < 16; ++value) {
        low_lut[value] = static_cast<uint8_t>(0x30U + value);
        high_lut[value] = static_cast<uint8_t>(0x10U + value);
        codes[value] = static_cast<uint8_t>(((15U - value) << 4U) | value);
    }

    auto verify = [&](auto lookup) {
        std::array<int32_t, 32> actual{};
        lookup(low_lut.data(), high_lut.data(), codes.data(), 1, actual.data());
        for (uint64_t lane = 0; lane < actual.size(); ++lane) {
            const uint64_t byte_index = 2 * (lane & 7U) + ((lane >> 3U) & 1U);
            const uint64_t code = lane < 16 ? byte_index : 15U - byte_index;
            const int32_t expected =
                static_cast<int32_t>(low_lut[code]) | (static_cast<int32_t>(high_lut[code]) << 8U);
            INFO("lane=" << lane);
            REQUIRE(actual[lane] == expected);
        }
    };

    verify(generic::PQFastScanLookUp32HighAcc);
    verify(PQFastScanLookUp32HighAcc);
    if (SimdStatus::SupportAVX2()) {
        verify(avx2::PQFastScanLookUp32HighAcc);
    }
    if (SimdStatus::SupportAVX512()) {
        verify(avx512::PQFastScanLookUp32HighAcc);
    }
}

TEST_CASE("PQFastScan HighAcc SIMD does not overflow narrow lanes", "[ut][simd]") {
    constexpr uint64_t pq_dim = 8192;
    constexpr int32_t expected_value = static_cast<int32_t>(pq_dim * 65535ULL);
    const uint64_t data_size = pq_dim * 16;
    std::vector<uint8_t> low_lut(data_size, 0xFFU);
    std::vector<uint8_t> high_lut(data_size, 0xFFU);
    std::vector<uint8_t> codes(data_size, 0U);
    std::vector<int32_t> expected(32, 0);
    generic::PQFastScanLookUp32HighAcc(
        low_lut.data(), high_lut.data(), codes.data(), pq_dim, expected.data());
    for (const int32_t value : expected) {
        REQUIRE(value == expected_value);
    }

    auto verify = [&](auto lookup) {
        std::vector<int32_t> actual(32, 0);
        lookup(low_lut.data(), high_lut.data(), codes.data(), pq_dim, actual.data());
        REQUIRE(compare_vector(expected, actual));
    };

    verify(PQFastScanLookUp32HighAcc);
    if (SimdStatus::SupportSSE()) {
        verify(sse::PQFastScanLookUp32HighAcc);
    }
    if (SimdStatus::SupportAVX()) {
        verify(avx::PQFastScanLookUp32HighAcc);
    }
    if (SimdStatus::SupportAVX2()) {
        verify(avx2::PQFastScanLookUp32HighAcc);
    }
    if (SimdStatus::SupportAVX512()) {
        verify(avx512::PQFastScanLookUp32HighAcc);
    }
    if (SimdStatus::SupportNEON()) {
        verify(neon::PQFastScanLookUp32HighAcc);
    }
    if (SimdStatus::SupportSVE()) {
        verify(sve::PQFastScanLookUp32HighAcc);
    }
}

TEST_CASE("FastScan FP32 less-than mask", "[ut][simd]") {
    std::array<float, 32> values{};
    for (uint64_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i) - 8.5F;
    }
    values[3] = std::numeric_limits<float>::quiet_NaN();
    values[19] = std::numeric_limits<float>::infinity();
    constexpr float limit = 7.25F;
    const uint32_t expected = generic::FP32LessThan32Mask(values.data(), limit);

    if (SimdStatus::SupportSSE()) {
        REQUIRE(sse::FP32LessThan32Mask(values.data(), limit) == expected);
    }
    if (SimdStatus::SupportAVX()) {
        REQUIRE(avx::FP32LessThan32Mask(values.data(), limit) == expected);
    }
    if (SimdStatus::SupportAVX2()) {
        REQUIRE(avx2::FP32LessThan32Mask(values.data(), limit) == expected);
    }
    if (SimdStatus::SupportAVX512()) {
        REQUIRE(avx512::FP32LessThan32Mask(values.data(), limit) == expected);
    }
    if (SimdStatus::SupportNEON()) {
        REQUIRE(neon::FP32LessThan32Mask(values.data(), limit) == expected);
    }
    if (SimdStatus::SupportSVE()) {
        REQUIRE(sve::FP32LessThan32Mask(values.data(), limit) == expected);
    }
}

TEST_CASE("RaBitQ FastScan32 residual postprocess SIMD parity", "[ut][simd]") {
    std::array<int32_t, 3 * 32> accumulators{};
    std::array<float, 32> f_add{};
    std::array<float, 32> f_scale{};
    std::array<float, 32> norm_codes{};
    for (uint64_t i = 0; i < 32; ++i) {
        f_add[i] = 2.0F + static_cast<float>(i) * 0.125F;
        f_scale[i] = -0.75F - static_cast<float>(i) * 0.01F;
        norm_codes[i] = 1.0F + static_cast<float>(i % 7) * 0.2F;
        for (uint64_t plane = 0; plane < 3; ++plane) {
            accumulators[plane * 32 + i] = static_cast<int32_t>(17 * i + 31 * plane + 3);
        }
    }
    f_add[5] = std::numeric_limits<float>::infinity();
    norm_codes[7] = 0.0F;
    f_scale[9] = 0.0F;
    f_scale[10] = std::numeric_limits<float>::quiet_NaN();
    f_add[11] = 0.0F;
    f_scale[11] = 0.0F;
    f_scale[12] = std::numeric_limits<float>::infinity();

    auto verify = [&](auto postprocess, uint32_t filter_bits, uint32_t valid_size) {
        std::array<float, 32> expected{};
        std::array<float, 32> actual{};
        std::array<float, 32> expected_filter_inner_products{};
        std::array<float, 32> actual_filter_inner_products{};
        const float* norm_code_ptr = filter_bits == 1 ? nullptr : norm_codes.data();
        const auto expected_mask =
            generic::RaBitQFastScan32ResidualPostprocess(accumulators.data(),
                                                         filter_bits,
                                                         0.003F,
                                                         -0.2F,
                                                         0.35F,
                                                         1.4F,
                                                         8.0F,
                                                         0.125F,
                                                         f_add.data(),
                                                         f_scale.data(),
                                                         norm_code_ptr,
                                                         valid_size,
                                                         expected.data(),
                                                         expected_filter_inner_products.data());
        for (const uint64_t invalid_lane : {5U, 9U, 10U, 11U, 12U}) {
            INFO("invalid lane=" << invalid_lane);
            REQUIRE((expected_mask & (1U << invalid_lane)) == 0);
            REQUIRE(expected[invalid_lane] == std::numeric_limits<float>::max());
            REQUIRE(has_nan_bits(expected_filter_inner_products[invalid_lane]));
        }
        if (filter_bits == 1) {
            REQUIRE((expected_mask & (1U << 7U)) != 0U);
            REQUIRE(std::isfinite(expected_filter_inner_products[7]));
        } else {
            REQUIRE((expected_mask & (1U << 7U)) == 0U);
            REQUIRE(expected[7] == std::numeric_limits<float>::max());
            REQUIRE(has_nan_bits(expected_filter_inner_products[7]));
        }
        const auto actual_mask = postprocess(accumulators.data(),
                                             filter_bits,
                                             0.003F,
                                             -0.2F,
                                             0.35F,
                                             1.4F,
                                             8.0F,
                                             0.125F,
                                             f_add.data(),
                                             f_scale.data(),
                                             norm_code_ptr,
                                             valid_size,
                                             actual.data(),
                                             actual_filter_inner_products.data());
        REQUIRE(actual_mask == expected_mask);
        for (uint64_t i = 0; i < 32; ++i) {
            INFO("filter_bits=" << filter_bits << ", lane=" << i);
            REQUIRE(std::abs(actual[i] - expected[i]) <= 1e-5F);
            if (has_nan_bits(expected_filter_inner_products[i])) {
                REQUIRE(has_nan_bits(actual_filter_inner_products[i]));
            } else {
                REQUIRE(std::abs(actual_filter_inner_products[i] -
                                 expected_filter_inner_products[i]) <= 1e-5F);
            }
        }
    };

    for (const uint32_t filter_bits : {1U, 2U, 3U}) {
        verify(RaBitQFastScan32ResidualPostprocess, filter_bits, 23);
        if (SimdStatus::SupportAVX2()) {
            verify(avx2::RaBitQFastScan32ResidualPostprocess, filter_bits, 32);
        }
        if (SimdStatus::SupportAVX512()) {
            verify(avx512::RaBitQFastScan32ResidualPostprocess, filter_bits, 32);
        }
        if (SimdStatus::SupportSSE()) {
            verify(sse::RaBitQFastScan32ResidualPostprocess, filter_bits, 32);
        }
        if (SimdStatus::SupportAVX()) {
            verify(avx::RaBitQFastScan32ResidualPostprocess, filter_bits, 32);
        }
        if (SimdStatus::SupportNEON()) {
            verify(neon::RaBitQFastScan32ResidualPostprocess, filter_bits, 32);
        }
        if (SimdStatus::SupportSVE()) {
            verify(sve::RaBitQFastScan32ResidualPostprocess, filter_bits, 32);
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
