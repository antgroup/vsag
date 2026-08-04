
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
