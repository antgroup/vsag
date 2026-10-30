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
#include <cstdint>

#include "simd/basic_func.h"
#include "unittest.h"

namespace vsag {

TEST_CASE("Generic prefetch is a no-op", "[ut][simd]") {
    std::array<uint8_t, 4> data{1, 2, 3, 4};
    const auto original = data;
    REQUIRE_NOTHROW(generic::Prefetch(data.data()));
    REQUIRE(data == original);
    REQUIRE_NOTHROW(generic::Prefetch(nullptr));
}

}  // namespace vsag
