
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

#include "sq8_uniform_simd.h"

#include <catch2/benchmark/catch_benchmark.hpp>

#include "simd_status.h"
#include "unittest.h"

using namespace vsag;

#define TEST_ACCURACY(Func)                                                                      \
    {                                                                                            \
        auto gt =                                                                                \
            generic::Func(codes1.data() + i * code_size, codes2.data() + i * code_size, dim);    \
        if (SimdStatus::SupportSSE()) {                                                          \
            auto sse =                                                                           \
                sse::Func(codes1.data() + i * code_size, codes2.data() + i * code_size, dim);    \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sse));                              \
        }                                                                                        \
        if (SimdStatus::SupportAVX()) {                                                          \
            auto avx =                                                                           \
                avx::Func(codes1.data() + i * code_size, codes2.data() + i * code_size, dim);    \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx));                              \
        }                                                                                        \
        if (SimdStatus::SupportAVX2()) {                                                         \
            auto avx2 =                                                                          \
                avx2::Func(codes1.data() + i * code_size, codes2.data() + i * code_size, dim);   \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx2));                             \
        }                                                                                        \
        if (SimdStatus::SupportAVX512()) {                                                       \
            auto avx512 =                                                                        \
                avx512::Func(codes1.data() + i * code_size, codes2.data() + i * code_size, dim); \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx512));                           \
        }                                                                                        \
        if (SimdStatus::SupportNEON()) {                                                         \
            auto neon =                                                                          \
                neon::Func(codes1.data() + i * code_size, codes2.data() + i * code_size, dim);   \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(neon));                             \
        }                                                                                        \
        if (SimdStatus::SupportSVE()) {                                                          \
            auto sve =                                                                           \
                sve::Func(codes1.data() + i * code_size, codes2.data() + i * code_size, dim);    \
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sve));                              \
        }                                                                                        \
    }

TEST_CASE("SQ8 Uniform SIMD Compute Codes", "[ut][simd]") {
    auto dims = fixtures::get_common_used_dims();
    int64_t count = 100;
    for (const auto& dim : dims) {
        uint32_t code_size = dim;
        auto codes1 = fixtures::generate_uint8_codes(count, dim, 114);
        auto codes2 = fixtures::generate_uint8_codes(count, dim, 514);
        for (uint64_t i = 0; i < count; ++i) {
            TEST_ACCURACY(SQ8UniformComputeCodesIP);
        }
    }
}

#define BENCHMARK_SIMD_COMPUTE(Simd, Comp)                                                 \
    BENCHMARK_ADVANCED(#Simd #Comp) {                                                      \
        for (int i = 0; i < count; ++i) {                                                  \
            Simd::Comp(codes1.data() + i * code_size, codes2.data() + i * code_size, dim); \
        }                                                                                  \
        return;                                                                            \
    }

TEST_CASE("SQ8 Uniform SIMD Compute Benchmark", "[ut][simd][!benchmark]") {
    int64_t count = 100;
    int64_t dim = 256;
    uint32_t code_size = dim;

    auto codes1 = fixtures::generate_uint8_codes(count, dim, 114);
    auto codes2 = fixtures::generate_uint8_codes(count, dim, 514);
    BENCHMARK_SIMD_COMPUTE(generic, SQ8UniformComputeCodesIP);
    if (SimdStatus::SupportSSE()) {
        BENCHMARK_SIMD_COMPUTE(sse, SQ8UniformComputeCodesIP);
    }
    if (SimdStatus::SupportAVX()) {
        BENCHMARK_SIMD_COMPUTE(avx, SQ8UniformComputeCodesIP);
    }
    if (SimdStatus::SupportAVX2()) {
        BENCHMARK_SIMD_COMPUTE(avx2, SQ8UniformComputeCodesIP);
    }
    if (SimdStatus::SupportAVX512()) {
        BENCHMARK_SIMD_COMPUTE(avx512, SQ8UniformComputeCodesIP);
    }
    if (SimdStatus::SupportNEON()) {
        BENCHMARK_SIMD_COMPUTE(neon, SQ8UniformComputeCodesIP);
    }
    if (SimdStatus::SupportSVE()) {
        BENCHMARK_SIMD_COMPUTE(sve, SQ8UniformComputeCodesIP);
    }
}

// ---------------- batch IP correctness + benchmark ----------------

namespace {

// Compute the ground-truth batch result using generic per-pair IP.
inline void
GenericBatch(const uint8_t* query,
             const uint8_t* codes,
             uint64_t dim,
             uint64_t n_codes,
             uint64_t code_stride,
             float* out) {
    for (uint64_t i = 0; i < n_codes; ++i) {
        out[i] = generic::SQ8UniformComputeCodesIP(query, codes + i * code_stride, dim);
    }
}

}  // namespace

TEST_CASE("SQ8 Uniform SIMD Compute Codes Batch", "[ut][simd]") {
    // Cover dims that are AMX-friendly (multiples of 64), AVX-512-aligned
    // (multiples of 32), and odd ones to exercise the tail handling on
    // the AMX path.
    const std::vector<int64_t> dims = {32, 48, 64, 96, 128, 192, 200, 256, 384, 512};
    // n_codes mix: smaller than the AMX 16-block, exactly one block,
    // multiple blocks plus a small remainder.
    const std::vector<int64_t> ns = {1, 7, 15, 16, 17, 33, 100};
    // Exercise tightly-packed (stride=dim) and strided (stride=dim+pad).
    const std::vector<int64_t> pads = {0, 8};

    for (auto dim : dims) {
        for (auto n : ns) {
            for (auto pad : pads) {
                const uint64_t code_stride = dim + pad;
                auto query = fixtures::generate_uint8_codes(1, dim, 31);
                auto codes = fixtures::generate_uint8_codes(n, code_stride, 17);
                std::vector<float> gt(n), got(n);
                GenericBatch(query.data(), codes.data(), dim, n, code_stride, gt.data());

                // Always exercise the dispatch entry point.
                std::fill(got.begin(), got.end(), 0.F);
                SQ8UniformComputeCodesIPBatch(
                    query.data(), codes.data(), dim, n, code_stride, got.data());
                for (int64_t i = 0; i < n; ++i) {
                    REQUIRE(fixtures::dist_t(gt[i]) == fixtures::dist_t(got[i]));
                }

                // And every available ISA's per-namespace implementation.
#define TEST_BATCH_NS(NS, Cond)                                           \
    if (Cond) {                                                           \
        std::fill(got.begin(), got.end(), 0.F);                           \
        NS::SQ8UniformComputeCodesIPBatch(                                \
            query.data(), codes.data(), dim, n, code_stride, got.data()); \
        for (int64_t i = 0; i < n; ++i) {                                 \
            REQUIRE(fixtures::dist_t(gt[i]) == fixtures::dist_t(got[i])); \
        }                                                                 \
    }
                TEST_BATCH_NS(generic, true);
                TEST_BATCH_NS(sse, SimdStatus::SupportSSE());
                TEST_BATCH_NS(avx, SimdStatus::SupportAVX());
                TEST_BATCH_NS(avx2, SimdStatus::SupportAVX2());
                TEST_BATCH_NS(avx512, SimdStatus::SupportAVX512());
#if defined(ENABLE_AMX)
                TEST_BATCH_NS(amx, SimdStatus::SupportAMX());
#endif
                TEST_BATCH_NS(neon, SimdStatus::SupportNEON());
                TEST_BATCH_NS(sve, SimdStatus::SupportSVE());
#undef TEST_BATCH_NS
            }
        }
    }
}

#define BENCHMARK_SIMD_BATCH(Simd)                                                      \
    BENCHMARK_ADVANCED(#Simd "SQ8UniformComputeCodesIPBatch") {                         \
        for (int q = 0; q < n_queries; ++q) {                                           \
            Simd::SQ8UniformComputeCodesIPBatch(                                        \
                queries.data() + q * dim, codes.data(), dim, n_codes, dim, out.data()); \
        }                                                                               \
        return;                                                                         \
    }

TEST_CASE("SQ8 Uniform SIMD Compute Codes Batch Benchmark", "[ut][simd][!benchmark]") {
    const int64_t dim = 256;
    const int64_t n_queries = 100;
    const int64_t n_codes = 1000;

    auto queries = fixtures::generate_uint8_codes(n_queries, dim, 91);
    auto codes = fixtures::generate_uint8_codes(n_codes, dim, 92);
    std::vector<float> out(n_codes);

    BENCHMARK_SIMD_BATCH(generic);
    if (SimdStatus::SupportSSE())
        BENCHMARK_SIMD_BATCH(sse);
    if (SimdStatus::SupportAVX())
        BENCHMARK_SIMD_BATCH(avx);
    if (SimdStatus::SupportAVX2())
        BENCHMARK_SIMD_BATCH(avx2);
    if (SimdStatus::SupportAVX512())
        BENCHMARK_SIMD_BATCH(avx512);
#if defined(ENABLE_AMX)
    if (SimdStatus::SupportAMX())
        BENCHMARK_SIMD_BATCH(amx);
#endif
    if (SimdStatus::SupportNEON())
        BENCHMARK_SIMD_BATCH(neon);
    if (SimdStatus::SupportSVE())
        BENCHMARK_SIMD_BATCH(sve);
}
