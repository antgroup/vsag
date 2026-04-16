
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

#include "linear_congruential_generator.h"

#include "unittest.h"

using namespace vsag;

TEST_CASE("LinearCongruentialGenerator Basic", "[ut][LinearCongruentialGenerator]") {
    LinearCongruentialGenerator gen;

    SECTION("value range") {
        for (int i = 0; i < 2000; ++i) {
            float v = gen.NextFloat();
            REQUIRE(v >= 0.0F);
            REQUIRE(v <= 1.0F);
        }
    }

    SECTION("sequence changes") {
        float first = gen.NextFloat();
        bool has_different = false;
        for (int i = 0; i < 100; ++i) {
            if (gen.NextFloat() != first) {
                has_different = true;
                break;
            }
        }
        REQUIRE(has_different);
    }
}
