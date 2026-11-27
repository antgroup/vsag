
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

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

#include "simd/rabitq_simd.h"
#include "simd/simd_status.h"

namespace vsag {
namespace {

void
CheckBinaryIP(RaBitQFloatBinaryType single, RaBitQFloatBinaryBatch4Type batch) {
    for (const uint64_t dim : {0, 1, 4, 7, 8, 9, 15, 16, 17, 31, 63, 64, 65, 128, 1025}) {
        const uint64_t bytes = (dim + 7) / 8;
        std::vector<float> query(dim + 1);
        std::array<std::vector<uint8_t>, 4> bases;
        for (uint64_t d = 0; d < dim; ++d) {
            query[d + 1] = static_cast<float>((d * 17) % 37) / 37.0F;
        }
        for (uint32_t i = 0; i < bases.size(); ++i) {
            auto& base = bases[i];
            base.resize(bytes + 1);
            for (uint64_t j = 0; j < bytes; ++j) {
                base[j + 1] = static_cast<uint8_t>((j * 31 + i * 117) ^ (j >> 2));
            }
        }
        for (const float scale : {0.0F, 0.0005F, 1e-3F, 0.125F}) {
            std::array<float, 4> out{99.0F, 99.0F, 99.0F, 99.0F};
            batch(query.data() + 1,
                  bases[0].data() + 1,
                  bases[1].data() + 1,
                  bases[2].data() + 1,
                  bases[3].data() + 1,
                  dim,
                  scale,
                  out.data());
            for (uint32_t i = 0; i < bases.size(); ++i) {
                const float expected =
                    generic::RaBitQFloatBinaryIP(query.data() + 1, bases[i].data() + 1, dim, scale);
                const float tolerance = 2e-6F * static_cast<float>(dim + 1);
                REQUIRE(std::abs(out[i] - expected) <= tolerance);
                if (single) {
                    REQUIRE(std::abs(single(query.data() + 1, bases[i].data() + 1, dim, scale) -
                                     expected) <= tolerance);
                }
            }
        }
    }
}

void
CheckSQ4UBinary(RaBitQSQ4UBinaryType inner,
                RaBitQSQ4UBinaryWithBaseSumType single,
                RaBitQSQ4UBinaryWithBaseSumBatch4Type batch) {
    for (const uint64_t dim : {0, 1, 7, 8, 9, 119, 127, 128, 129, 255, 256, 257, 8193, 65537}) {
        const uint64_t bytes = (dim + 7) / 8;
        std::vector<uint8_t> codes(4 * bytes + 1);
        std::array<std::vector<uint8_t>, 4> bases;
        for (uint64_t j = 0; j < 4 * bytes; ++j) {
            codes[j + 1] = static_cast<uint8_t>((j * 31) ^ (j >> 2) ^ 253U);
        }
        for (uint32_t i = 0; i < bases.size(); ++i) {
            auto& base = bases[i];
            base.resize(bytes + 1);
            for (uint64_t j = 0; j < bytes; ++j) {
                base[j + 1] = static_cast<uint8_t>((j * 31 + i * 117) ^ (j >> 2));
            }
            if (dim % 8 != 0) {
                base.back() &= (1U << (dim % 8)) - 1;
            }
        }
        std::array<uint64_t, 4> out{99, 99, 99, 99};
        batch(codes.data() + 1,
              bases[0].data() + 1,
              bases[1].data() + 1,
              bases[2].data() + 1,
              bases[3].data() + 1,
              dim,
              out.data());
        for (uint32_t i = 0; i < bases.size(); ++i) {
            const auto expected =
                generic::RaBitQSQ4UBinaryIPWithBaseSum(codes.data() + 1, bases[i].data() + 1, dim);
            REQUIRE(inner(codes.data() + 1, bases[i].data() + 1, dim) ==
                    static_cast<uint32_t>(expected));
            REQUIRE(single(codes.data() + 1, bases[i].data() + 1, dim) == expected);
            REQUIRE(out[i] == expected);
        }
    }
}

}  // namespace

TEST_CASE("RaBitQ binary additional ISA kernels match generic",
          "[ut][simd][rabitq-binary-parity]") {
    if (SimdStatus::SupportSSE()) {
        CheckBinaryIP(sse::RaBitQFloatBinaryIP, sse::RaBitQFloatBinaryIPBatch4);
        CheckSQ4UBinary(sse::RaBitQSQ4UBinaryIP,
                        sse::RaBitQSQ4UBinaryIPWithBaseSum,
                        sse::RaBitQSQ4UBinaryIPWithBaseSumBatch4);
    }
    if (SimdStatus::SupportAVX() && SimdStatus::SupportSSE()) {
        CheckSQ4UBinary(avx::RaBitQSQ4UBinaryIP,
                        avx::RaBitQSQ4UBinaryIPWithBaseSum,
                        avx::RaBitQSQ4UBinaryIPWithBaseSumBatch4);
    }
    if (SimdStatus::SupportAVX2()) {
        CheckSQ4UBinary(avx2::RaBitQSQ4UBinaryIP,
                        avx2::RaBitQSQ4UBinaryIPWithBaseSum,
                        avx2::RaBitQSQ4UBinaryIPWithBaseSumBatch4);
    }
    if (SimdStatus::SupportNEON()) {
        CheckBinaryIP(nullptr, neon::RaBitQFloatBinaryIPBatch4);
        CheckSQ4UBinary(neon::RaBitQSQ4UBinaryIP,
                        neon::RaBitQSQ4UBinaryIPWithBaseSum,
                        neon::RaBitQSQ4UBinaryIPWithBaseSumBatch4);
    }
    CheckSQ4UBinary(
        RaBitQSQ4UBinaryIP, RaBitQSQ4UBinaryIPWithBaseSum, RaBitQSQ4UBinaryIPWithBaseSumBatch4);
}

}  // namespace vsag
