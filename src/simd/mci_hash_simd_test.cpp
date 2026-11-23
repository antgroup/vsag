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

#include "mci_hash_simd.h"

#include <cstdint>
#include <limits>

#include "unittest.h"

TEST_CASE("MCI SIMD membership checks both unaligned buckets", "[ut][simd]") {
    alignas(16) const int32_t first[] = {0, 3, -3, 42, std::numeric_limits<int32_t>::min(), 99};
    alignas(16) const int32_t second[] = {0, 1, 7, -15, std::numeric_limits<int32_t>::max(), 99};
    for (int lane = 1; lane <= 4; ++lane) {
        CAPTURE(lane);
        REQUIRE(vsag::simd::mci_hash_contains_two_int32x4(first + 1, second + 1, first[lane]));
        REQUIRE(vsag::simd::mci_hash_contains_two_int32x4(first + 1, second + 1, second[lane]));
    }
    REQUIRE_FALSE(vsag::simd::mci_hash_contains_two_int32x4(first + 1, second + 1, 0));
    REQUIRE_FALSE(vsag::simd::mci_hash_contains_two_int32x4(first + 1, second + 1, 99));
}
