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
    constexpr uint32_t num = 37;
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
        ids[i] = static_cast<uint16_t>((i * 13 + 7) % dist_count);
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
        CheckSparseAccumulate(
            avx::FP16SparseAccumulate, fp16_expected, ids, fp16_vals, query_val, 1e-6F);
    }
    if (SimdStatus::SupportAVX2()) {
        CheckSparseAccumulate(
            avx2::FP32SparseAccumulate, fp32_expected, ids, fp32_vals, query_val, 1e-6F);
        CheckSparseAccumulate(
            avx2::SQ8SparseAccumulate, sq8_expected, ids, sq8_vals, query_val, 1e-6F);
        CheckSparseAccumulate(
            avx2::FP16SparseAccumulate, fp16_expected, ids, fp16_vals, query_val, 1e-6F);
    }
    if (SimdStatus::SupportAVX512()) {
        CheckSparseAccumulate(
            avx512::FP32SparseAccumulate, fp32_expected, ids, fp32_vals, query_val, 1e-6F);
        CheckSparseAccumulate(
            avx512::SQ8SparseAccumulate, sq8_expected, ids, sq8_vals, query_val, 1e-6F);
        CheckSparseAccumulate(
            avx512::FP16SparseAccumulate, fp16_expected, ids, fp16_vals, query_val, 1e-6F);
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
