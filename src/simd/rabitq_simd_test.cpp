
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

    if (SimdStatus::SupportNEON) {
        BENCHMARK_SIMD_COMPUTE_SQ4(neon, RaBitQSQ4UBinaryIP);
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

        std::vector<float> neon_datas(gt.size());
        neon_datas.assign(gt.begin(), gt.end());

        for (int i = 0; i < count; i++) {
            auto* gt_data = gt.data() + i * dim;
            auto* avx512_data = avx512_datas.data() + i * dim;
            auto* avx2_data = avx2_datas.data() + i * dim;
            auto* avx_data = avx_datas.data() + i * dim;
            auto* sse_data = sse_datas.data() + i * dim;
            auto* neon_data = neon_datas.data() + i * dim;

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