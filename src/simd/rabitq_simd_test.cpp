
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

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "fp32_simd.h"

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
    };

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
    };

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
        for (int i = 0; i < count; i++) {
            auto* gt_data = gt.data() + i * dim;
            auto* avx512_data = avx512_datas.data() + i * dim;
            auto* avx2_data = avx2_datas.data() + i * dim;
            auto* avx_data = avx_datas.data() + i * dim;
            auto* sse_data = sse_datas.data() + i * dim;

            const float delta = 1e-5;
            generic::VecRescale(gt_data, dim, 0.5);
            if (SimdStatus::SupportAVX512()) {
                avx512::VecRescale(avx512_data, dim, 0.5);
                for (int i = 0; i < dim; i++) {
                    REQUIRE(gt_data[i] - avx512_data[i] < delta);
                }
            }
            if (SimdStatus::SupportAVX2()) {
                avx2::VecRescale(avx2_data, dim, 0.5);
                for (int i = 0; i < dim; i++) {
                    REQUIRE(gt_data[i] - avx2_data[i] < delta);
                }
            }
            if (SimdStatus::SupportAVX()) {
                avx::VecRescale(avx_data, dim, 0.5);
                for (int i = 0; i < dim; i++) {
                    REQUIRE(gt_data[i] - avx_data[i] < delta);
                }
            }
            if (SimdStatus::SupportSSE()) {
                sse::VecRescale(sse_data, dim, 0.5);
                for (int i = 0; i < dim; i++) {
                    REQUIRE(gt_data[i] - sse_data[i] < delta);
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

        const float delta = 1e-5;
        for (int i = 0; i < count; i++) {
            auto* gt_data = gt.data() + i * dim;
            generic::KacsWalk(gt_data, dim);
            if (SimdStatus::SupportAVX512()) {
                auto* avx512_data = avx512_datas.data() + i * dim;
                avx512::KacsWalk(avx512_data, dim);
                for (int i = 0; i < dim; i++) {
                    REQUIRE(gt_data[i] - avx512_data[i] < delta);
                }
            }
            if (SimdStatus::SupportAVX2()) {
                auto* avx2_data = avx2_datas.data() + i * dim;
                avx2::KacsWalk(avx2_data, dim);
                for (int i = 0; i < dim; i++) {
                    REQUIRE(gt_data[i] - avx2_data[i] < delta);
                }
            }
            if (SimdStatus::SupportAVX()) {
                auto* avx_data = avx_datas.data() + i * dim;
                avx::KacsWalk(avx_data, dim);
                for (int i = 0; i < dim; i++) {
                    REQUIRE(gt_data[i] - avx_data[i] < delta);
                }
            }
            if (SimdStatus::SupportSSE()) {
                auto* sse_data = sse_datas.data() + i * dim;
                sse::KacsWalk(sse_data, dim);
                for (int i = 0; i < dim; i++) {
                    REQUIRE(gt_data[i] - sse_data[i] < delta);
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

        const float delta = 1e-5;
        for (int i = 0; i < count; i++) {
            auto* gt_data = gt.data() + i * dim;
            size_t tmp_dim = dim;
            size_t ret = 0;
            while (tmp_dim > 1) {
                ret++;
                tmp_dim >>= 1;
            }
            size_t trunc_dim = 1 << ret;
            int start = dim - trunc_dim;
            generic::FHTRotate(gt_data, trunc_dim);
            generic::FHTRotate(gt_data + start, trunc_dim);

            if (SimdStatus::SupportAVX512()) {
                auto* avx512_data = avx512_datas.data() + i * dim;
                avx512::FHTRotate(avx512_data, trunc_dim);
                avx512::FHTRotate(avx512_data + start, trunc_dim);
                for (int i = 0; i < dim; i++) {
                    REQUIRE(gt_data[i] - avx512_data[i] < delta);
                }
            }
            if (SimdStatus::SupportAVX2()) {
                auto* avx2_data = avx2_datas.data() + i * dim;
                avx2::FHTRotate(avx2_data, trunc_dim);
                avx2::FHTRotate(avx2_data + start, trunc_dim);
                for (int i = 0; i < dim; i++) {
                    REQUIRE(gt_data[i] - avx2_data[i] < delta);
                }
            }
            if (SimdStatus::SupportAVX()) {
                auto* avx_data = avx_datas.data() + i * dim;
                avx::FHTRotate(avx_data, trunc_dim);
                avx::FHTRotate(avx_data + start, trunc_dim);
                for (int i = 0; i < dim; i++) {
                    REQUIRE(gt_data[i] - avx_data[i] < delta);
                }
            }
            if (SimdStatus::SupportSSE()) {
                auto* sse_data = sse_datas.data() + i * dim;
                sse::FHTRotate(sse_data, trunc_dim);
                sse::FHTRotate(sse_data + start, trunc_dim);
                for (int i = 0; i < dim; i++) {
                    REQUIRE(gt_data[i] - sse_data[i] < delta);
                }
            }
        }
    }
}
