
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
#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "simd_status.h"

using namespace vsag;

#define TEST_FP32_COMPUTE_ACCURACY(Func)                                              \
    {                                                                                 \
        float gt, sse, avx, avx2, avx512, neon, sve;                                  \
        gt = generic::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim);        \
        if (SimdStatus::SupportSSE()) {                                               \
            sse = sse::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim);       \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sse));                   \
        }                                                                             \
        if (SimdStatus::SupportAVX()) {                                               \
            avx = avx::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim);       \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx));                   \
        }                                                                             \
        if (SimdStatus::SupportAVX2()) {                                              \
            avx2 = avx2::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim);     \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx2));                  \
        }                                                                             \
        if (SimdStatus::SupportAVX512()) {                                            \
            avx512 = avx512::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim); \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx512));                \
        }                                                                             \
        if (SimdStatus::SupportNEON()) {                                              \
            neon = neon::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim);     \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(neon));                  \
        }                                                                             \
        if (SimdStatus::SupportSVE()) {                                               \
            sve = sve::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim);       \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sve));                   \
        }                                                                             \
    };
#define TEST_FP32_ARTHIMETIC_ACCURACY(Func)                                                    \
    {                                                                                          \
        std::vector<float> gt(dim, 0.0F);                                                      \
        generic::Func(vec1.data() + i * dim, vec2.data() + i * dim, gt.data(), dim);           \
        std::vector<float> sse_gt(dim, 0.0F);                                                  \
        if (SimdStatus::SupportSSE()) {                                                        \
            sse::Func(vec1.data() + i * dim, vec2.data() + i * dim, sse_gt.data(), dim);       \
            for (uint64_t j = 0; j < dim; ++j) {                                               \
                REQUIRE(fixtures::dist_t(gt[j]) == fixtures::dist_t(sse_gt[j]));               \
            }                                                                                  \
        }                                                                                      \
        std::vector<float> avx_gt(dim, 0.0F);                                                  \
        if (SimdStatus::SupportAVX()) {                                                        \
            avx::Func(vec1.data() + i * dim, vec2.data() + i * dim, avx_gt.data(), dim);       \
            for (uint64_t j = 0; j < dim; ++j) {                                               \
                REQUIRE(fixtures::dist_t(gt[j]) == fixtures::dist_t(avx_gt[j]));               \
            }                                                                                  \
        }                                                                                      \
        std::vector<float> avx2_gt(dim, 0.0F);                                                 \
        if (SimdStatus::SupportAVX2()) {                                                       \
            avx2::Func(vec1.data() + i * dim, vec2.data() + i * dim, avx2_gt.data(), dim);     \
            for (uint64_t j = 0; j < dim; ++j) {                                               \
                REQUIRE(fixtures::dist_t(gt[j]) == fixtures::dist_t(avx2_gt[j]));              \
            }                                                                                  \
        }                                                                                      \
        std::vector<float> avx512_gt(dim, 0.0F);                                               \
        if (SimdStatus::SupportAVX512()) {                                                     \
            avx512::Func(vec1.data() + i * dim, vec2.data() + i * dim, avx512_gt.data(), dim); \
            for (uint64_t j = 0; j < dim; ++j) {                                               \
                REQUIRE(fixtures::dist_t(gt[j]) == fixtures::dist_t(avx512_gt[j]));            \
            }                                                                                  \
        }                                                                                      \
        std::vector<float> neon(dim, 0.0F);                                                    \
        if (SimdStatus::SupportNEON()) {                                                       \
            neon::Func(vec1.data() + i * dim, vec2.data() + i * dim, neon.data(), dim);        \
            for (uint64_t j = 0; j < dim; ++j) {                                               \
                REQUIRE(fixtures::dist_t(gt[j]) == fixtures::dist_t(neon[j]));                 \
            }                                                                                  \
        }                                                                                      \
        std::vector<float> sve(dim, 0.0F);                                                     \
        if (SimdStatus::SupportSVE()) {                                                        \
            sve::Func(vec1.data() + i * dim, vec2.data() + i * dim, sve.data(), dim);          \
            for (uint64_t j = 0; j < dim; ++j) {                                               \
                REQUIRE(fixtures::dist_t(gt[j]) == fixtures::dist_t(sve[j]));                  \
            }                                                                                  \
        }                                                                                      \
    };

#define TEST_FP32_COMPUTE_ACCURACY_BATCH4(Func, FuncBatch4)                              \
    {                                                                                    \
        std::vector<float> gts(4);                                                       \
        gts[0] = generic::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim);       \
        gts[1] = generic::Func(vec1.data() + i * dim, vec2.data() + (i + 1) * dim, dim); \
        gts[2] = generic::Func(vec1.data() + i * dim, vec2.data() + (i + 2) * dim, dim); \
        gts[3] = generic::Func(vec1.data() + i * dim, vec2.data() + (i + 3) * dim, dim); \
        std::vector<float> result(4, 0.0F);                                              \
        memset(result.data(), 0, 4 * sizeof(float));                                     \
        generic::FuncBatch4(vec1.data() + i * dim,                                       \
                            dim,                                                         \
                            vec2.data() + i * dim,                                       \
                            vec2.data() + (i + 1) * dim,                                 \
                            vec2.data() + (i + 2) * dim,                                 \
                            vec2.data() + (i + 3) * dim,                                 \
                            result[0],                                                   \
                            result[1],                                                   \
                            result[2],                                                   \
                            result[3]);                                                  \
        for (uint64_t j = 0; j < 4; ++j) {                                               \
            REQUIRE(fixtures::dist_t(gts[j]) == fixtures::dist_t(result[j]));            \
        }                                                                                \
        if (SimdStatus::SupportSSE()) {                                                  \
            memset(result.data(), 0, 4 * sizeof(float));                                 \
            sse::FuncBatch4(vec1.data() + i * dim,                                       \
                            dim,                                                         \
                            vec2.data() + i * dim,                                       \
                            vec2.data() + (i + 1) * dim,                                 \
                            vec2.data() + (i + 2) * dim,                                 \
                            vec2.data() + (i + 3) * dim,                                 \
                            result[0],                                                   \
                            result[1],                                                   \
                            result[2],                                                   \
                            result[3]);                                                  \
            for (uint64_t j = 0; j < 4; ++j) {                                           \
                REQUIRE(fixtures::dist_t(gts[j]) == fixtures::dist_t(result[j]));        \
            }                                                                            \
        }                                                                                \
        if (SimdStatus::SupportAVX()) {                                                  \
            memset(result.data(), 0, 4 * sizeof(float));                                 \
            avx::FuncBatch4(vec1.data() + i * dim,                                       \
                            dim,                                                         \
                            vec2.data() + i * dim,                                       \
                            vec2.data() + (i + 1) * dim,                                 \
                            vec2.data() + (i + 2) * dim,                                 \
                            vec2.data() + (i + 3) * dim,                                 \
                            result[0],                                                   \
                            result[1],                                                   \
                            result[2],                                                   \
                            result[3]);                                                  \
            for (uint64_t j = 0; j < 4; ++j) {                                           \
                REQUIRE(fixtures::dist_t(gts[j]) == fixtures::dist_t(result[j]));        \
            }                                                                            \
        }                                                                                \
        if (SimdStatus::SupportAVX2()) {                                                 \
            memset(result.data(), 0, 4 * sizeof(float));                                 \
            avx2::FuncBatch4(vec1.data() + i * dim,                                      \
                             dim,                                                        \
                             vec2.data() + i * dim,                                      \
                             vec2.data() + (i + 1) * dim,                                \
                             vec2.data() + (i + 2) * dim,                                \
                             vec2.data() + (i + 3) * dim,                                \
                             result[0],                                                  \
                             result[1],                                                  \
                             result[2],                                                  \
                             result[3]);                                                 \
            for (uint64_t j = 0; j < 4; ++j) {                                           \
                REQUIRE(fixtures::dist_t(gts[j]) == fixtures::dist_t(result[j]));        \
            }                                                                            \
        }                                                                                \
        if (SimdStatus::SupportAVX512()) {                                               \
            memset(result.data(), 0, 4 * sizeof(float));                                 \
            avx512::FuncBatch4(vec1.data() + i * dim,                                    \
                               dim,                                                      \
                               vec2.data() + i * dim,                                    \
                               vec2.data() + (i + 1) * dim,                              \
                               vec2.data() + (i + 2) * dim,                              \
                               vec2.data() + (i + 3) * dim,                              \
                               result[0],                                                \
                               result[1],                                                \
                               result[2],                                                \
                               result[3]);                                               \
            for (uint64_t j = 0; j < 4; ++j) {                                           \
                REQUIRE(fixtures::dist_t(gts[j]) == fixtures::dist_t(result[j]));        \
            }                                                                            \
        }                                                                                \
        if (SimdStatus::SupportNEON()) {                                                 \
            memset(result.data(), 0, 4 * sizeof(float));                                 \
            neon::FuncBatch4(vec1.data() + i * dim,                                      \
                             dim,                                                        \
                             vec2.data() + i * dim,                                      \
                             vec2.data() + (i + 1) * dim,                                \
                             vec2.data() + (i + 2) * dim,                                \
                             vec2.data() + (i + 3) * dim,                                \
                             result[0],                                                  \
                             result[1],                                                  \
                             result[2],                                                  \
                             result[3]);                                                 \
            for (uint64_t j = 0; j < 4; ++j) {                                           \
                REQUIRE(fixtures::dist_t(gts[j]) == fixtures::dist_t(result[j]));        \
            }                                                                            \
        }                                                                                \
        if (SimdStatus::SupportSVE()) {                                                  \
            memset(result.data(), 0, 4 * sizeof(float));                                 \
            sve::FuncBatch4(vec1.data() + i * dim,                                       \
                            dim,                                                         \
                            vec2.data() + i * dim,                                       \
                            vec2.data() + (i + 1) * dim,                                 \
                            vec2.data() + (i + 2) * dim,                                 \
                            vec2.data() + (i + 3) * dim,                                 \
                            result[0],                                                   \
                            result[1],                                                   \
                            result[2],                                                   \
                            result[3]);                                                  \
            for (uint64_t j = 0; j < 4; ++j) {                                           \
                REQUIRE(fixtures::dist_t(gts[j]) == fixtures::dist_t(result[j]));        \
            }                                                                            \
        }                                                                                \
    };

TEST_CASE("FP32 SIMD Compute", "[ut][simd]") {
    const std::vector<int64_t> dims = {8, 16, 32, 256};
    int64_t count = 100;
    for (const auto& dim : dims) {
        auto vec1 = fixtures::generate_vectors(count * 2, dim);
        std::vector<float> vec2(vec1.begin() + count, vec1.end());
        for (uint64_t i = 0; i < count; ++i) {
            TEST_FP32_COMPUTE_ACCURACY(FP32ComputeIP);
            TEST_FP32_COMPUTE_ACCURACY(FP32ComputeL2Sqr);
            TEST_FP32_ARTHIMETIC_ACCURACY(FP32Sub);
            TEST_FP32_ARTHIMETIC_ACCURACY(FP32Add);
            TEST_FP32_ARTHIMETIC_ACCURACY(FP32Mul);
            TEST_FP32_ARTHIMETIC_ACCURACY(FP32Div);
        }
        for (uint64_t i = 0; i < count; i += 4) {
            TEST_FP32_COMPUTE_ACCURACY_BATCH4(FP32ComputeIP, FP32ComputeIPBatch4);
            TEST_FP32_COMPUTE_ACCURACY_BATCH4(FP32ComputeL2Sqr, FP32ComputeL2SqrBatch4);
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
    int64_t dim = 128;
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
