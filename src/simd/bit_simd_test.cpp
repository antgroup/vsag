
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

#include "bit_simd.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_all.hpp>

#include "fixtures.h"
#include "simd_status.h"

using namespace vsag;

#define TEST_BIT_OPERATOR_ACCURACY(Func)                                                     \
    {                                                                                        \
        std::vector<uint8_t> gt(num_bytes, 0);                                               \
        generic::Func(                                                                       \
            vec1.data() + i * num_bytes, vec2.data() + i * num_bytes, num_bytes, gt.data()); \
        std::vector<uint8_t> sse_gt(num_bytes, 0.0F);                                        \
        if (SimdStatus::SupportSSE()) {                                                      \
            sse::Func(vec1.data() + i * num_bytes,                                           \
                      vec2.data() + i * num_bytes,                                           \
                      num_bytes,                                                             \
                      sse_gt.data());                                                        \
            for (uint64_t j = 0; j < num_bytes; ++j) {                                       \
                REQUIRE(fixtures::dist_t(gt[j]) == fixtures::dist_t(sse_gt[j]));             \
            }                                                                                \
        }                                                                                    \
        std::vector<uint8_t> avx_gt(num_bytes, 0.0F);                                        \
        if (SimdStatus::SupportAVX()) {                                                      \
            avx::Func(vec1.data() + i * num_bytes,                                           \
                      vec2.data() + i * num_bytes,                                           \
                      num_bytes,                                                             \
                      avx_gt.data());                                                        \
            for (uint64_t j = 0; j < num_bytes; ++j) {                                       \
                REQUIRE(fixtures::dist_t(gt[j]) == fixtures::dist_t(avx_gt[j]));             \
            }                                                                                \
        }                                                                                    \
        std::vector<uint8_t> avx2_gt(num_bytes, 0);                                          \
        if (SimdStatus::SupportAVX2()) {                                                     \
            avx2::Func(vec1.data() + i * num_bytes,                                          \
                       vec2.data() + i * num_bytes,                                          \
                       num_bytes,                                                            \
                       avx2_gt.data());                                                      \
            for (uint64_t j = 0; j < num_bytes; ++j) {                                       \
                REQUIRE(fixtures::dist_t(gt[j]) == fixtures::dist_t(avx2_gt[j]));            \
            }                                                                                \
        }                                                                                    \
        std::vector<uint8_t> avx512_gt(num_bytes, 0);                                        \
        if (SimdStatus::SupportAVX512()) {                                                   \
            avx512::Func(vec1.data() + i * num_bytes,                                        \
                         vec2.data() + i * num_bytes,                                        \
                         num_bytes,                                                          \
                         avx512_gt.data());                                                  \
            for (uint64_t j = 0; j < num_bytes; ++j) {                                       \
                REQUIRE(fixtures::dist_t(gt[j]) == fixtures::dist_t(avx512_gt[j]));          \
            }                                                                                \
        }                                                                                    \
        std::vector<uint8_t> neon_gt(num_bytes, 0);                                          \
        if (SimdStatus::SupportNEON()) {                                                     \
            neon::Func(vec1.data() + i * num_bytes,                                          \
                       vec2.data() + i * num_bytes,                                          \
                       num_bytes,                                                            \
                       neon_gt.data());                                                      \
            for (uint64_t j = 0; j < num_bytes; ++j) {                                       \
                REQUIRE(fixtures::dist_t(gt[j]) == fixtures::dist_t(neon_gt[j]));            \
            }                                                                                \
        }                                                                                    \
    };

#define TEST_BIT_NOT_ACCURACY(Func)                                                 \
    {                                                                               \
        std::vector<uint8_t> gt(num_bytes, 0);                                      \
        generic::Func(vec1.data() + i * num_bytes, num_bytes, gt.data());           \
        std::vector<uint8_t> sse_gt(num_bytes, 0);                                  \
        if (SimdStatus::SupportSSE()) {                                             \
            sse::Func(vec1.data() + i * num_bytes, num_bytes, sse_gt.data());       \
            for (uint64_t j = 0; j < num_bytes; ++j) {                              \
                REQUIRE(fixtures::dist_t(gt[j]) == fixtures::dist_t(sse_gt[j]));    \
            }                                                                       \
        }                                                                           \
        std::vector<uint8_t> avx_gt(num_bytes, 0);                                  \
        if (SimdStatus::SupportAVX()) {                                             \
            avx::Func(vec1.data() + i * num_bytes, num_bytes, avx_gt.data());       \
            for (uint64_t j = 0; j < num_bytes; ++j) {                              \
                REQUIRE(fixtures::dist_t(gt[j]) == fixtures::dist_t(avx_gt[j]));    \
            }                                                                       \
        }                                                                           \
        std::vector<uint8_t> avx2_gt(num_bytes, 0);                                 \
        if (SimdStatus::SupportAVX2()) {                                            \
            avx2::Func(vec1.data() + i * num_bytes, num_bytes, avx2_gt.data());     \
            for (uint64_t j = 0; j < num_bytes; ++j) {                              \
                REQUIRE(fixtures::dist_t(gt[j]) == fixtures::dist_t(avx2_gt[j]));   \
            }                                                                       \
        }                                                                           \
        std::vector<uint8_t> avx512_gt(num_bytes, 0);                               \
        if (SimdStatus::SupportAVX512()) {                                          \
            avx512::Func(vec1.data() + i * num_bytes, num_bytes, avx512_gt.data()); \
            for (uint64_t j = 0; j < num_bytes; ++j) {                              \
                REQUIRE(fixtures::dist_t(gt[j]) == fixtures::dist_t(avx512_gt[j])); \
            }                                                                       \
        }                                                                           \
        std::vector<uint8_t> neon_gt(num_bytes, 0);                                 \
        if (SimdStatus::SupportNEON()) {                                            \
            neon::Func(vec1.data() + i * num_bytes, num_bytes, neon_gt.data());     \
            for (uint64_t j = 0; j < num_bytes; ++j) {                              \
                REQUIRE(fixtures::dist_t(gt[j]) == fixtures::dist_t(neon_gt[j]));   \
            }                                                                       \
        }                                                                           \
    };

TEST_CASE("Bit Operator (NOT)", "[ut][simd]") {
    const auto dims = fixtures::get_common_used_dims();
    int64_t count = 100;
    for (const auto& num_bytes : dims) {
        auto vec1 = fixtures::GenerateVectors<uint8_t>(count * 2, num_bytes);
        std::vector<uint8_t> vec2(vec1.begin() + count, vec1.end());
        for (uint64_t i = 0; i < count; ++i) {
            TEST_BIT_NOT_ACCURACY(BitNot);
        }
    }
}

TEST_CASE("Bit Operator (AND, OR, XOR)", "[ut][simd]") {
    const auto dims = fixtures::get_common_used_dims();
    int64_t count = 100;
    for (const auto& num_bytes : dims) {
        auto vec1 = fixtures::GenerateVectors<uint8_t>(count * 2, num_bytes);
        std::vector<uint8_t> vec2(vec1.begin() + count, vec1.end());
        for (uint64_t i = 0; i < count; ++i) {
            TEST_BIT_OPERATOR_ACCURACY(BitAnd);
            TEST_BIT_OPERATOR_ACCURACY(BitOr);
            TEST_BIT_OPERATOR_ACCURACY(BitXor);
        }
    }
}

#define BENCHMARK_BIT_OPERATOR_COMPUTE(Simd, Comp)                                      \
    BENCHMARK_ADVANCED(#Simd #Comp) {                                                   \
        for (int i = 0; i < count; ++i) {                                               \
            Simd::Comp(vec1.data() + i * dim, vec2.data() + i * dim, dim, vec3.data()); \
        }                                                                               \
        return;                                                                         \
    }

#define BENCHMARK_BIT_NOT_COMPUTE(Simd, Comp)                    \
    BENCHMARK_ADVANCED(#Simd #Comp) {                            \
        for (int i = 0; i < count; ++i) {                        \
            Simd::Comp(vec1.data() + i * dim, dim, vec3.data()); \
        }                                                        \
        return;                                                  \
    }

TEST_CASE("Bit Operator (AND, OR, XOR, NOT)", "[!benchmark][simd]") {
    const auto dim = 4096;
    int64_t count = 500;
    auto vec1 = fixtures::GenerateVectors<uint8_t>(count * 3, dim);
    std::vector<uint8_t> vec2(vec1.begin() + count, vec1.end());
    std::vector<uint8_t> vec3(vec1.begin() + count * 2, vec1.end());

    SECTION("Bit Operator And") {
        BENCHMARK_BIT_OPERATOR_COMPUTE(generic, BitAnd);
        if (SimdStatus::SupportSSE()) {
            BENCHMARK_BIT_OPERATOR_COMPUTE(sse, BitAnd);
        }
        if (SimdStatus::SupportAVX()) {
            BENCHMARK_BIT_OPERATOR_COMPUTE(avx, BitAnd);
        }
        if (SimdStatus::SupportAVX2()) {
            BENCHMARK_BIT_OPERATOR_COMPUTE(avx2, BitAnd);
        }
        if (SimdStatus::SupportAVX512()) {
            BENCHMARK_BIT_OPERATOR_COMPUTE(avx512, BitAnd);
        }
        if (SimdStatus::SupportNEON()) {
            BENCHMARK_BIT_OPERATOR_COMPUTE(neon, BitAnd);
        }
    }

    SECTION("Bit Operator Or") {
        BENCHMARK_BIT_OPERATOR_COMPUTE(generic, BitOr);
        if (SimdStatus::SupportSSE()) {
            BENCHMARK_BIT_OPERATOR_COMPUTE(sse, BitOr);
        }
        if (SimdStatus::SupportAVX()) {
            BENCHMARK_BIT_OPERATOR_COMPUTE(avx, BitOr);
        }
        if (SimdStatus::SupportAVX2()) {
            BENCHMARK_BIT_OPERATOR_COMPUTE(avx2, BitOr);
        }
        if (SimdStatus::SupportAVX512()) {
            BENCHMARK_BIT_OPERATOR_COMPUTE(avx512, BitOr);
        }
        if (SimdStatus::SupportNEON()) {
            BENCHMARK_BIT_OPERATOR_COMPUTE(neon, BitOr);
        }
    }

    SECTION("Bit Operator Xor") {
        BENCHMARK_BIT_OPERATOR_COMPUTE(generic, BitXor);
        if (SimdStatus::SupportSSE()) {
            BENCHMARK_BIT_OPERATOR_COMPUTE(sse, BitXor);
        }
        if (SimdStatus::SupportAVX()) {
            BENCHMARK_BIT_OPERATOR_COMPUTE(avx, BitXor);
        }
        if (SimdStatus::SupportAVX2()) {
            BENCHMARK_BIT_OPERATOR_COMPUTE(avx2, BitXor);
        }
        if (SimdStatus::SupportAVX512()) {
            BENCHMARK_BIT_OPERATOR_COMPUTE(avx512, BitXor);
        }
        if (SimdStatus::SupportNEON()) {
            BENCHMARK_BIT_OPERATOR_COMPUTE(neon, BitXor);
        }
    }

    SECTION("Bit Operator Not") {
        BENCHMARK_BIT_NOT_COMPUTE(generic, BitNot);
        if (SimdStatus::SupportSSE()) {
            BENCHMARK_BIT_NOT_COMPUTE(sse, BitNot);
        }
        if (SimdStatus::SupportAVX()) {
            BENCHMARK_BIT_NOT_COMPUTE(avx, BitNot);
        }
        if (SimdStatus::SupportAVX2()) {
            BENCHMARK_BIT_NOT_COMPUTE(avx2, BitNot);
        }
        if (SimdStatus::SupportAVX512()) {
            BENCHMARK_BIT_NOT_COMPUTE(avx512, BitNot);
        }
        if (SimdStatus::SupportNEON()) {
            BENCHMARK_BIT_NOT_COMPUTE(neon, BitNot);
        }
    }
}
