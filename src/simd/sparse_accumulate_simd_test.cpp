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

#include <catch2/catch_all.hpp>
#include <cmath>
#include <vector>

#include "fp16_simd.h"
#include "fp32_simd.h"
#include "simd_status.h"
#include "sq8_simd.h"

using namespace vsag;

namespace {

template <typename Func, typename ValueT>
void
CheckSparseAccumulate(Func func,
                      const std::vector<float>& expected,
                      const std::vector<uint16_t>& ids,
                      const std::vector<ValueT>& vals,
                      float query_val,
                      float tolerance) {
    std::vector<float> actual(expected.size(), 0.25F);

    func(actual.data(), ids.data(), vals.data(), query_val, static_cast<uint32_t>(ids.size()));
    for (uint64_t i = 0; i < actual.size(); ++i) {
        REQUIRE(std::abs(actual[i] - expected[i]) <= tolerance);
    }
}

}  // namespace

TEST_CASE("Sparse Accumulate SIMD", "[ut][simd]") {
    const uint32_t num = GENERATE(7U, 8U, 16U, 37U);
    const bool repeated_ids = GENERATE(false, true);
    CAPTURE(num, repeated_ids);
    constexpr uint32_t dist_count = 64;
    const float query_val = -0.375F;

    std::vector<uint16_t> ids(num);
    std::vector<float> fp32_vals(num);
    std::vector<uint8_t> sq8_vals(num);
    std::vector<uint16_t> fp16_vals(num);
    std::vector<float> fp32_expected(dist_count, 0.25F);
    std::vector<float> sq8_expected(dist_count, 0.25F);
    std::vector<float> fp16_expected(dist_count, 0.25F);

    for (uint32_t i = 0; i < num; ++i) {
        const uint32_t id_source = repeated_ids && i % 3 == 1 ? i - 1 : i;
        ids[i] = static_cast<uint16_t>((id_source * 13 + 7) % dist_count);
        fp32_vals[i] = static_cast<float>(i % 11) * 0.125F - 0.5F;
        sq8_vals[i] = static_cast<uint8_t>((i * 17 + 3) % 251);
        fp16_vals[i] = generic::FloatToFP16(fp32_vals[i]);

        fp32_expected[ids[i]] += fp32_vals[i] * query_val;
        sq8_expected[ids[i]] += static_cast<float>(sq8_vals[i]) * query_val;
        fp16_expected[ids[i]] += generic::FP16ToFloat(fp16_vals[i]) * query_val;
    }

    CheckSparseAccumulate(
        generic::FP32SparseAccumulate, fp32_expected, ids, fp32_vals, query_val, 1e-6F);
    CheckSparseAccumulate(
        generic::SQ8SparseAccumulate, sq8_expected, ids, sq8_vals, query_val, 1e-6F);
    CheckSparseAccumulate(
        generic::FP16SparseAccumulate, fp16_expected, ids, fp16_vals, query_val, 1e-6F);

    if (SimdStatus::SupportSSE()) {
        CheckSparseAccumulate(
            sse::FP32SparseAccumulate, fp32_expected, ids, fp32_vals, query_val, 1e-6F);
        CheckSparseAccumulate(
            sse::SQ8SparseAccumulate, sq8_expected, ids, sq8_vals, query_val, 1e-6F);
        CheckSparseAccumulate(
            sse::FP16SparseAccumulate, fp16_expected, ids, fp16_vals, query_val, 1e-6F);
    }
    if (SimdStatus::SupportAVX()) {
        CheckSparseAccumulate(
            avx::FP32SparseAccumulate, fp32_expected, ids, fp32_vals, query_val, 1e-6F);
        CheckSparseAccumulate(
            avx::SQ8SparseAccumulate, sq8_expected, ids, sq8_vals, query_val, 1e-6F);
        if (SimdStatus::SupportAVXF16C()) {
            CheckSparseAccumulate(
                avx::FP16SparseAccumulate, fp16_expected, ids, fp16_vals, query_val, 1e-6F);
        }
    }
    if (SimdStatus::SupportAVX2()) {
        CheckSparseAccumulate(
            avx2::FP32SparseAccumulate, fp32_expected, ids, fp32_vals, query_val, 1e-6F);
        CheckSparseAccumulate(
            avx2::SQ8SparseAccumulate, sq8_expected, ids, sq8_vals, query_val, 1e-6F);
        if (SimdStatus::SupportAVX2F16C()) {
            CheckSparseAccumulate(
                avx2::FP16SparseAccumulate, fp16_expected, ids, fp16_vals, query_val, 1e-6F);
        }
    }
    if (SimdStatus::SupportAVX512()) {
        CheckSparseAccumulate(
            avx512::FP32SparseAccumulate, fp32_expected, ids, fp32_vals, query_val, 1e-6F);
        CheckSparseAccumulate(
            avx512::SQ8SparseAccumulate, sq8_expected, ids, sq8_vals, query_val, 1e-6F);
        if (SimdStatus::SupportAVX512F16C()) {
            CheckSparseAccumulate(
                avx512::FP16SparseAccumulate, fp16_expected, ids, fp16_vals, query_val, 1e-6F);
        }
    }
    if (SimdStatus::SupportNEON()) {
        CheckSparseAccumulate(
            neon::FP32SparseAccumulate, fp32_expected, ids, fp32_vals, query_val, 1e-6F);
        CheckSparseAccumulate(
            neon::SQ8SparseAccumulate, sq8_expected, ids, sq8_vals, query_val, 1e-6F);
        CheckSparseAccumulate(
            neon::FP16SparseAccumulate, fp16_expected, ids, fp16_vals, query_val, 1e-6F);
    }
    if (SimdStatus::SupportSVE()) {
        CheckSparseAccumulate(
            sve::FP32SparseAccumulate, fp32_expected, ids, fp32_vals, query_val, 1e-6F);
        CheckSparseAccumulate(
            sve::SQ8SparseAccumulate, sq8_expected, ids, sq8_vals, query_val, 1e-6F);
        CheckSparseAccumulate(
            sve::FP16SparseAccumulate, fp16_expected, ids, fp16_vals, query_val, 1e-6F);
    }

    CheckSparseAccumulate(FP32SparseAccumulate, fp32_expected, ids, fp32_vals, query_val, 1e-6F);
    CheckSparseAccumulate(SQ8SparseAccumulate, sq8_expected, ids, sq8_vals, query_val, 1e-6F);
    CheckSparseAccumulate(FP16SparseAccumulate, fp16_expected, ids, fp16_vals, query_val, 1e-6F);
}

TEST_CASE("Sparse SIMD preserves posting order and legacy half decode", "[ut][simd]") {
    // Offsets exercise naturally aligned but vector-unaligned input. IDs repeat
    // within a SIMD block and across blocks; 65535 is a valid posting ID.
    const std::vector<uint16_t> ids = {42, 65535, 1, 65535, 1, 7, 7, 1, 65535, 2, 7};
    const std::vector<float> fp32 = {0, 4, 3, -4, 6, 7, -2, 1, 3, -5, 12};
    const std::vector<uint16_t> fp16 = {
        0, 0x0000, 0x0001, 0x8000, 0x03ff, 0x0400, 0x7bff, 0x7c00, 0x7fff, 0xfc00, 0xffff};
    const std::vector<uint8_t> sq8 = {0, 255, 1, 128, 3, 127, 64, 3, 0, 10, 214};
    constexpr float query_val = -0.375F;

    auto check = [&](auto reference, auto kernel, const auto& values) {
        for (uint32_t num = 0; num < ids.size(); ++num) {
            CAPTURE(num);
            std::vector<float> expected(65536, 0.0F);
            std::vector<float> actual = expected;
            reference(expected.data(), ids.data() + 1, values.data() + 1, query_val, num);
            kernel(actual.data(), ids.data() + 1, values.data() + 1, query_val, num);
            for (uint16_t id : ids) {
                REQUIRE(actual[id] == expected[id]);
            }
        }
    };
    if (SimdStatus::SupportSSE()) {
        check(generic::FP16SparseAccumulate, sse::FP16SparseAccumulate, fp16);
    }
    if (SimdStatus::SupportNEON()) {
        check(generic::FP32SparseAccumulate, neon::FP32SparseAccumulate, fp32);
        check(generic::FP16SparseAccumulate, neon::FP16SparseAccumulate, fp16);
        check(generic::SQ8SparseAccumulate, neon::SQ8SparseAccumulate, sq8);
    }
    if (SimdStatus::SupportSVE()) {
        check(generic::FP32SparseAccumulate, sve::FP32SparseAccumulate, fp32);
        check(generic::FP16SparseAccumulate, sve::FP16SparseAccumulate, fp16);
        check(generic::SQ8SparseAccumulate, sve::SQ8SparseAccumulate, sq8);
    }
}
