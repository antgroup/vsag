
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

#include "fp32_simd.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <cstring>
#include <random>

#include "pipnn_filter.h"
#include "pipnn_ip_tile.h"
#include "simd_status.h"
#include "unittest.h"

using namespace vsag;

namespace {

void
CheckFP32ComputeAccuracy(const std::vector<float>& vec1,
                         const std::vector<float>& vec2,
                         uint64_t dim,
                         uint64_t i,
                         FP32ComputeType generic_func,
                         FP32ComputeType sse_func,
                         FP32ComputeType avx_func,
                         FP32ComputeType avx2_func,
                         FP32ComputeType avx512_func,
                         FP32ComputeType neon_func,
                         FP32ComputeType sve_func) {
    const auto* lhs = vec1.data() + i * dim;
    const auto* rhs = vec2.data() + i * dim;
    const auto expected = generic_func(lhs, rhs, dim);

    auto check = [&](bool supported, FP32ComputeType func) {
        if (not supported) {
            return;
        }
        const auto actual = func(lhs, rhs, dim);
        REQUIRE(fixtures::dist_t(expected) == fixtures::dist_t(actual));
    };

    check(SimdStatus::SupportSSE(), sse_func);
    check(SimdStatus::SupportAVX(), avx_func);
    check(SimdStatus::SupportAVX2(), avx2_func);
    check(SimdStatus::SupportAVX512(), avx512_func);
    check(SimdStatus::SupportNEON(), neon_func);
    check(SimdStatus::SupportSVE(), sve_func);
}

void
CheckFP32ArithmeticAccuracy(const std::vector<float>& vec1,
                            const std::vector<float>& vec2,
                            uint64_t dim,
                            uint64_t i,
                            FP32ArithmeticType generic_func,
                            FP32ArithmeticType sse_func,
                            FP32ArithmeticType avx_func,
                            FP32ArithmeticType avx2_func,
                            FP32ArithmeticType avx512_func,
                            FP32ArithmeticType neon_func,
                            FP32ArithmeticType sve_func) {
    const auto* lhs = vec1.data() + i * dim;
    const auto* rhs = vec2.data() + i * dim;
    std::vector<float> expected(dim, 0.0F);
    generic_func(lhs, rhs, expected.data(), dim);

    auto check = [&](bool supported, FP32ArithmeticType func) {
        if (not supported) {
            return;
        }
        std::vector<float> actual(dim, 0.0F);
        func(lhs, rhs, actual.data(), dim);
        for (uint64_t j = 0; j < dim; ++j) {
            REQUIRE(fixtures::dist_t(expected[j]) == fixtures::dist_t(actual[j]));
        }
    };

    check(SimdStatus::SupportSSE(), sse_func);
    check(SimdStatus::SupportAVX(), avx_func);
    check(SimdStatus::SupportAVX2(), avx2_func);
    check(SimdStatus::SupportAVX512(), avx512_func);
    check(SimdStatus::SupportNEON(), neon_func);
    check(SimdStatus::SupportSVE(), sve_func);
}

void
CheckFP32ComputeBatch4Accuracy(const std::vector<float>& vec1,
                               const std::vector<float>& vec2,
                               uint64_t dim,
                               uint64_t i,
                               FP32ComputeType generic_func,
                               FP32ComputeBatch4Type generic_batch_func,
                               FP32ComputeBatch4Type sse_batch_func,
                               FP32ComputeBatch4Type avx_batch_func,
                               FP32ComputeBatch4Type avx2_batch_func,
                               FP32ComputeBatch4Type avx512_batch_func,
                               FP32ComputeBatch4Type neon_batch_func,
                               FP32ComputeBatch4Type sve_batch_func) {
    const auto* query = vec1.data() + i * dim;
    std::vector<float> expected(4, 0.0F);
    for (uint64_t j = 0; j < expected.size(); ++j) {
        expected[j] = generic_func(query, vec2.data() + (i + j) * dim, dim);
    }

    auto check = [&](bool supported, FP32ComputeBatch4Type func) {
        if (not supported) {
            return;
        }
        std::vector<float> actual(4, 0.0F);
        func(query,
             dim,
             vec2.data() + i * dim,
             vec2.data() + (i + 1) * dim,
             vec2.data() + (i + 2) * dim,
             vec2.data() + (i + 3) * dim,
             actual[0],
             actual[1],
             actual[2],
             actual[3]);
        for (uint64_t j = 0; j < expected.size(); ++j) {
            REQUIRE(fixtures::dist_t(expected[j]) == fixtures::dist_t(actual[j]));
        }
    };

    check(true, generic_batch_func);
    check(SimdStatus::SupportSSE(), sse_batch_func);
    check(SimdStatus::SupportAVX(), avx_batch_func);
    check(SimdStatus::SupportAVX2(), avx2_batch_func);
    check(SimdStatus::SupportAVX512(), avx512_batch_func);
    check(SimdStatus::SupportNEON(), neon_batch_func);
    check(SimdStatus::SupportSVE(), sve_batch_func);
}

}  // namespace

TEST_CASE("FP32 SIMD Compute", "[ut][simd]") {
    const std::vector<uint64_t> dims = {8, 16, 32, 256};
    const uint64_t count = 100;
    for (const auto dim : dims) {
        auto vec1 = fixtures::generate_vectors(count * 2, dim);
        std::vector<float> vec2(vec1.begin() + count, vec1.end());
        for (uint64_t i = 0; i < count; ++i) {
            CheckFP32ComputeAccuracy(vec1,
                                     vec2,
                                     dim,
                                     i,
                                     generic::FP32ComputeIP,
                                     sse::FP32ComputeIP,
                                     avx::FP32ComputeIP,
                                     avx2::FP32ComputeIP,
                                     avx512::FP32ComputeIP,
                                     neon::FP32ComputeIP,
                                     sve::FP32ComputeIP);
            CheckFP32ComputeAccuracy(vec1,
                                     vec2,
                                     dim,
                                     i,
                                     generic::FP32ComputeL2Sqr,
                                     sse::FP32ComputeL2Sqr,
                                     avx::FP32ComputeL2Sqr,
                                     avx2::FP32ComputeL2Sqr,
                                     avx512::FP32ComputeL2Sqr,
                                     neon::FP32ComputeL2Sqr,
                                     sve::FP32ComputeL2Sqr);
            CheckFP32ArithmeticAccuracy(vec1,
                                        vec2,
                                        dim,
                                        i,
                                        generic::FP32Sub,
                                        sse::FP32Sub,
                                        avx::FP32Sub,
                                        avx2::FP32Sub,
                                        avx512::FP32Sub,
                                        neon::FP32Sub,
                                        sve::FP32Sub);
            CheckFP32ArithmeticAccuracy(vec1,
                                        vec2,
                                        dim,
                                        i,
                                        generic::FP32Add,
                                        sse::FP32Add,
                                        avx::FP32Add,
                                        avx2::FP32Add,
                                        avx512::FP32Add,
                                        neon::FP32Add,
                                        sve::FP32Add);
            CheckFP32ArithmeticAccuracy(vec1,
                                        vec2,
                                        dim,
                                        i,
                                        generic::FP32Mul,
                                        sse::FP32Mul,
                                        avx::FP32Mul,
                                        avx2::FP32Mul,
                                        avx512::FP32Mul,
                                        neon::FP32Mul,
                                        sve::FP32Mul);
            CheckFP32ArithmeticAccuracy(vec1,
                                        vec2,
                                        dim,
                                        i,
                                        generic::FP32Div,
                                        sse::FP32Div,
                                        avx::FP32Div,
                                        avx2::FP32Div,
                                        avx512::FP32Div,
                                        neon::FP32Div,
                                        sve::FP32Div);
        }
        for (uint64_t i = 0; i < count; i += 4) {
            CheckFP32ComputeBatch4Accuracy(vec1,
                                           vec2,
                                           dim,
                                           i,
                                           generic::FP32ComputeIP,
                                           generic::FP32ComputeIPBatch4,
                                           sse::FP32ComputeIPBatch4,
                                           avx::FP32ComputeIPBatch4,
                                           avx2::FP32ComputeIPBatch4,
                                           avx512::FP32ComputeIPBatch4,
                                           neon::FP32ComputeIPBatch4,
                                           sve::FP32ComputeIPBatch4);
            CheckFP32ComputeBatch4Accuracy(vec1,
                                           vec2,
                                           dim,
                                           i,
                                           generic::FP32ComputeL2Sqr,
                                           generic::FP32ComputeL2SqrBatch4,
                                           sse::FP32ComputeL2SqrBatch4,
                                           avx::FP32ComputeL2SqrBatch4,
                                           avx2::FP32ComputeL2SqrBatch4,
                                           avx512::FP32ComputeL2SqrBatch4,
                                           neon::FP32ComputeL2SqrBatch4,
                                           sve::FP32ComputeL2SqrBatch4);
        }
    }
}

#define BENCHMARK_SIMD_COMPUTE(Simd, Comp)                                 \
    BENCHMARK_ADVANCED(#Simd #Comp) {                                      \
        for (int i = 0; i < count; ++i) {                                  \
            Simd::Comp(vec1.data() + i * dim, vec2.data() + i * dim, dim); \
        }                                                                  \
        return;                                                            \
    }

TEST_CASE("FP32 Benchmark", "[ut][simd][!benchmark]") {
    int64_t count = 500;
    int64_t dim = 128;
    auto vec1 = fixtures::generate_vectors(count * 2, dim);
    std::vector<float> vec2(vec1.begin() + count, vec1.end());
    BENCHMARK_SIMD_COMPUTE(generic, FP32ComputeIP);
    if (SimdStatus::SupportSSE()) {
        BENCHMARK_SIMD_COMPUTE(sse, FP32ComputeIP);
    }
    if (SimdStatus::SupportAVX()) {
        BENCHMARK_SIMD_COMPUTE(avx, FP32ComputeIP);
    }
    if (SimdStatus::SupportAVX2()) {
        BENCHMARK_SIMD_COMPUTE(avx2, FP32ComputeIP);
    }
    if (SimdStatus::SupportAVX512()) {
        BENCHMARK_SIMD_COMPUTE(avx512, FP32ComputeIP);
    }
    if (SimdStatus::SupportNEON()) {
        BENCHMARK_SIMD_COMPUTE(neon, FP32ComputeIP);
    }
    if (SimdStatus::SupportSVE()) {
        BENCHMARK_SIMD_COMPUTE(sve, FP32ComputeIP);
    }

    BENCHMARK_SIMD_COMPUTE(generic, FP32ComputeL2Sqr);
    if (SimdStatus::SupportSSE()) {
        BENCHMARK_SIMD_COMPUTE(sse, FP32ComputeL2Sqr);
    }
    if (SimdStatus::SupportAVX()) {
        BENCHMARK_SIMD_COMPUTE(avx, FP32ComputeL2Sqr);
    }
    if (SimdStatus::SupportAVX2()) {
        BENCHMARK_SIMD_COMPUTE(avx2, FP32ComputeL2Sqr);
    }
    if (SimdStatus::SupportAVX512()) {
        BENCHMARK_SIMD_COMPUTE(avx512, FP32ComputeL2Sqr);
    }
    if (SimdStatus::SupportNEON()) {
        BENCHMARK_SIMD_COMPUTE(neon, FP32ComputeL2Sqr);
    }
    if (SimdStatus::SupportSVE()) {
        BENCHMARK_SIMD_COMPUTE(sve, FP32ComputeL2Sqr);
    }
}

#define BENCHMARK_SIMD_COMPUTE_BATCH4(Simd, Comp)        \
    BENCHMARK_ADVANCED(#Simd #Comp) {                    \
        std::vector<float> result(4);                    \
        for (int i = 0; i < count; i += 4) {             \
            memset(result.data(), 0, 4 * sizeof(float)); \
            Simd::Comp(vec1.data() + i * dim,            \
                       dim,                              \
                       vec2.data() + i * dim,            \
                       vec2.data() + (i + 1) * dim,      \
                       vec2.data() + (i + 2) * dim,      \
                       vec2.data() + (i + 3) * dim,      \
                       result[0],                        \
                       result[1],                        \
                       result[2],                        \
                       result[3]);                       \
        }                                                \
        return;                                          \
    }

TEST_CASE("FP32 Benchmark Batch4", "[ut][simd][!benchmark]") {
    int64_t count = 500;
    int64_t dim = 256;
    auto vec1 = fixtures::generate_vectors(count * 2, dim);
    std::vector<float> vec2(vec1.begin() + count, vec1.end());
    BENCHMARK_SIMD_COMPUTE_BATCH4(generic, FP32ComputeIPBatch4);
    if (SimdStatus::SupportSSE()) {
        BENCHMARK_SIMD_COMPUTE_BATCH4(sse, FP32ComputeIPBatch4);
    }
    if (SimdStatus::SupportAVX()) {
        BENCHMARK_SIMD_COMPUTE_BATCH4(avx, FP32ComputeIPBatch4);
    }
    if (SimdStatus::SupportAVX2()) {
        BENCHMARK_SIMD_COMPUTE_BATCH4(avx2, FP32ComputeIPBatch4);
    }
    if (SimdStatus::SupportAVX512()) {
        BENCHMARK_SIMD_COMPUTE_BATCH4(avx512, FP32ComputeIPBatch4);
    }
    if (SimdStatus::SupportNEON()) {
        BENCHMARK_SIMD_COMPUTE_BATCH4(neon, FP32ComputeIPBatch4);
    }
    if (SimdStatus::SupportSVE()) {
        BENCHMARK_SIMD_COMPUTE_BATCH4(sve, FP32ComputeIPBatch4);
    }

    BENCHMARK_SIMD_COMPUTE_BATCH4(generic, FP32ComputeL2SqrBatch4);
    if (SimdStatus::SupportSSE()) {
        BENCHMARK_SIMD_COMPUTE_BATCH4(sse, FP32ComputeL2SqrBatch4);
    }
    if (SimdStatus::SupportAVX()) {
        BENCHMARK_SIMD_COMPUTE_BATCH4(avx, FP32ComputeL2SqrBatch4);
    }
    if (SimdStatus::SupportAVX2()) {
        BENCHMARK_SIMD_COMPUTE_BATCH4(avx2, FP32ComputeL2SqrBatch4);
    }
    if (SimdStatus::SupportAVX512()) {
        BENCHMARK_SIMD_COMPUTE_BATCH4(avx512, FP32ComputeL2SqrBatch4);
    }
    if (SimdStatus::SupportNEON()) {
        BENCHMARK_SIMD_COMPUTE_BATCH4(neon, FP32ComputeL2SqrBatch4);
    }
    if (SimdStatus::SupportSVE()) {
        BENCHMARK_SIMD_COMPUTE_BATCH4(sve, FP32ComputeL2SqrBatch4);
    }
}

TEST_CASE("PiPNN old row tiles match the dispatched FP32 kernels bit for bit",
          "[ut][simd][pipnn][add]") {
    const uint64_t dim = GENERATE(959U, 960U, 961U);
    const uint64_t m = GENERATE(3U, 6U, 8U);
    const uint64_t b = GENERATE(1U, 3U, 4U, 7U, 8U);
    const auto n = m + b;
    std::mt19937 generator(42);
    std::vector<float> values(n * dim);
    for (auto& value : values) {
        value = std::generate_canonical<float, 24>(generator) - 0.5F;
    }
    std::vector<const float*> rows(n);
    std::vector<uint32_t> members(n);
    for (uint64_t i = 0; i < n; ++i) {
        rows[i] = values.data() + i * dim;
        members[i] = static_cast<uint32_t>(n - 1 - i);
    }
    constexpr float sentinel = -123456.0F;
    std::vector<float> dots(n * b, sentinel), expected(n * b, sentinel);
    const auto processed =
        vsag::ComputePiPNNOldRowTiles(rows.data(), members.data(), m, b, dim, dots.data());
    const bool use_tiles = dim == 960 && vsag::FP32ComputeIP == vsag::avx512::FP32ComputeIP &&
                           vsag::FP32ComputeIPBatch4 == vsag::avx512::FP32ComputeIPBatch4;
    REQUIRE(processed == (use_tiles ? m - m % 4 : 0));
    for (uint64_t i = 0; i < processed; ++i) {
        auto* old = rows[members[i]];
        uint64_t j = 0;
        for (; j + 4 <= b; j += 4) {
            float scores[4]{};
            vsag::FP32ComputeIPBatch4(old,
                                      dim,
                                      rows[members[m + j]],
                                      rows[members[m + j + 1]],
                                      rows[members[m + j + 2]],
                                      rows[members[m + j + 3]],
                                      scores[0],
                                      scores[1],
                                      scores[2],
                                      scores[3]);
            for (int t = 0; t < 4; ++t) expected[(j + t) * n + i] = scores[t];
        }
        for (; j < b; ++j)
            expected[j * n + i] = vsag::FP32ComputeIP(old, rows[members[m + j]], dim);
    }
    REQUIRE(std::memcmp(dots.data(), expected.data(), dots.size() * sizeof(float)) == 0);
}

TEST_CASE("PiPNN dispatched relative hashes preserve both directions and coincidence",
          "[ut][simd][pipnn]") {
    const auto hash =
        GENERATE(&vsag::pipnn_filter::ScalarHash, vsag::pipnn_filter::GetKernels().hashes);
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (uint64_t planes = 1; planes <= 15; ++planes) {
        const uint16_t active = static_cast<uint16_t>((1U << planes) - 1);
        std::vector<float> a(planes, -0.0F), b(planes, +0.0F);
        a.front() = b.front() = inf;
        const auto equal = hash(a.data(), b.data(), planes);
        REQUIRE(equal.forward == (active | vsag::pipnn_filter::COINCIDENT_HASH_FLAG));
        REQUIRE(equal.reverse == equal.forward);
        for (uint64_t i = 0; i < planes; ++i) {
            CAPTURE(planes, i);
            const auto bit = static_cast<uint16_t>(1U << i);
            a[i] = -inf;
            b[i] = inf;
            auto different = hash(a.data(), b.data(), planes);
            REQUIRE(different.forward == active);
            REQUIRE(different.reverse == (active ^ bit));
            b[i] = nan;
            auto unordered = hash(a.data(), b.data(), planes);
            REQUIRE(unordered.forward == (active ^ bit));
            REQUIRE(unordered.reverse == (active ^ bit));
            b[i] = a[i];
        }
    }
}

TEST_CASE("PiPNN vector kth filter includes ties and masks the last partial group",
          "[ut][simd][pipnn]") {
    const auto select = vsag::pipnn_filter::GetKernels().kth;
    if (!select) {
        SUCCEED("The generic Add path uses row-wise comparisons");
        return;
    }
    for (uint64_t n : {0U, 1U, 15U, 16U, 17U, 31U, 32U, 33U}) {
        std::vector<float> d(n), limit(n);
        std::vector<uint32_t> rows(n + 1, UINT32_MAX), expected;
        for (uint32_t i = 0; i < n; ++i) {
            limit[i] = i % 5 == 0 ? std::numeric_limits<float>::infinity() : -0.0F;
            d[i] = i % 5 == 0 ? limit[i] : static_cast<float>(i % 3) - 1.0F;
            if (d[i] <= limit[i])
                expected.push_back(i);
        }
        const auto count = select(d.data(), limit.data(), n, rows.data());
        REQUIRE(count == expected.size());
        REQUIRE(std::equal(expected.begin(), expected.end(), rows.begin()));
        REQUIRE(rows[count] == UINT32_MAX);
    }
}
