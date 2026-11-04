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

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "simd/rabitq_simd.h"
#include "simd/simd_status.h"

TEST_CASE("RaBitQ AVX512 single three-bit and five-bit supplement handle short tails",
          "[ut][simd]") {
    if (not vsag::SimdStatus::SupportAVX512()) {
        return;
    }
    for (uint64_t dim :
         {0UL, 1UL, 7UL, 8UL, 15UL, 16UL, 17UL, 31UL, 33UL, 127UL, 128UL, 129UL, 960UL, 1024UL}) {
        const uint64_t plane_bytes = (dim + 7) / 8;
        std::vector<float> values(dim + 1);
        auto* query = values.data() + 1;
        for (uint64_t d = 0; d < dim; ++d) {
            query[d] = static_cast<float>(static_cast<int64_t>(d % 31) - 15) / 64.0F;
        }
        for (uint32_t pattern = 0; pattern < 32; ++pattern) {
            std::vector<uint8_t> filter(3 * plane_bytes + 1, 0);
            std::vector<uint8_t> supplement(5 * plane_bytes + 1, 0);
            double expected_filter = 0.0;
            double expected_supplement = 0.0;
            for (uint64_t d = 0; d < dim; ++d) {
                const uint32_t value = (pattern + d) & 31U;
                expected_filter += query[d] * (static_cast<double>(value & 7U) - 3.5);
                expected_supplement += query[d] * static_cast<double>(value);
                for (uint32_t bit = 0; bit < 3; ++bit) {
                    filter[1 + bit * plane_bytes + (d >> 3)] |= ((value >> (2 - bit)) & 1U)
                                                                << (d & 7);
                }
                for (uint32_t bit = 0; bit < 5; ++bit) {
                    supplement[1 + bit * plane_bytes + (d >> 3)] |= ((value >> bit) & 1U)
                                                                    << (d & 7);
                }
            }
            CHECK(vsag::avx512::RaBitQFloatThreeBitCenteredIP(query, filter.data() + 1, dim) ==
                  static_cast<float>(expected_filter));
            CHECK(vsag::avx512::RaBitQFloatSupplementCodeIP(query, supplement.data() + 1, dim, 5) ==
                  static_cast<float>(expected_supplement));
        }
    }
}

TEST_CASE("RaBitQ AVX512 in-register FHT matches scalar butterflies", "[ut][simd]") {
    if (not vsag::SimdStatus::SupportAVX512()) {
        return;
    }
    for (uint64_t dim : {1UL, 2UL, 4UL, 8UL, 16UL, 32UL, 64UL, 128UL, 512UL, 1024UL}) {
        std::vector<float> actual(dim + 1);
        auto* values = actual.data() + 1;
        for (uint64_t i = 0; i < dim; ++i) {
            values[i] = static_cast<float>(static_cast<int64_t>((i * 17) % 101) - 50) / 64.0F;
        }
        const std::vector<float> original(values, values + dim);
        auto expected = original;
        vsag::generic::FHTRotate(expected.data(), dim);
        vsag::avx512::FHTRotate(values, dim);
        CHECK(std::memcmp(values, expected.data(), dim * sizeof(float)) == 0);
        vsag::avx512::FHTRotate(values, dim);
        for (uint64_t i = 0; i < dim; ++i) {
            CHECK(values[i] / static_cast<float>(dim) == original[i]);
        }
    }
}
