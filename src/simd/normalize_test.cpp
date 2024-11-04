
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

#include "normalize.h"

#include "catch2/benchmark/catch_benchmark.hpp"
#include "catch2/catch_test_macros.hpp"
#include "fixtures.h"

using namespace vsag;

#ifndef ENABLE_SSE
namespace sse = generic;
#endif

#ifndef ENABLE_AVX2
namespace avx2 = sse;
#endif

#ifndef ENABLE_AVX512
namespace avx512 = avx2;
#endif

TEST_CASE("Normalize SIMD Compute", "[simd]") {
    auto dims = fixtures::get_common_used_dims();
    int64_t count = 100;
    for (auto& dim : dims) {
        auto vec1 = fixtures::generate_vectors(count, dim);
        std::vector<float> tmp_value(dim * 4);
        for (uint64_t i = 0; i < count; ++i) {
            auto gt = generic::Normalize(vec1.data() + i * dim, tmp_value.data(), dim);
            auto sse = sse::Normalize(vec1.data() + i * dim, tmp_value.data() + dim, dim);
            auto avx2 = avx2::Normalize(vec1.data() + i * dim, tmp_value.data() + dim * 2, dim);
            auto avx512 = avx512::Normalize(vec1.data() + i * dim, tmp_value.data() + dim * 3, dim);
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sse));
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx2));
            REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx512));
            for (int j = 0; j < dim; ++j) {
                REQUIRE(fixtures::dist_t(tmp_value[j]) == fixtures::dist_t(tmp_value[j + dim]));
                REQUIRE(fixtures::dist_t(tmp_value[j]) == fixtures::dist_t(tmp_value[j + dim * 2]));
                REQUIRE(fixtures::dist_t(tmp_value[j]) == fixtures::dist_t(tmp_value[j + dim * 3]));
            }
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

TEST_CASE("Normalize benchmark", "[simd][!benchmark]") {
    int64_t count = 500;
    int64_t dim = 128;
    auto vec1 = fixtures::generate_vectors(count * 2, dim);
    std::vector<float> vec2(vec1.begin() + count, vec1.end());
    BENCHMARK_SIMD_COMPUTE(generic, Normalize);
    BENCHMARK_SIMD_COMPUTE(sse, Normalize);
    BENCHMARK_SIMD_COMPUTE(avx2, Normalize);
    BENCHMARK_SIMD_COMPUTE(avx512, Normalize);
}
