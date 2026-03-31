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

#include "label_table.h"

#include <catch2/catch_test_macros.hpp>

#include "default_allocator.h"

using namespace vsag;

TEST_CASE("LabelTable TryGetIdByLabel", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();

    SECTION("reverse map lookup") {
        LabelTable label_table(allocator.get());
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);

        auto [ok1, id1] = label_table.TryGetIdByLabel(100);
        auto [ok2, id2] = label_table.TryGetIdByLabel(999);

        REQUIRE(ok1);
        REQUIRE(id1 == 0);
        REQUIRE(ok2 == false);
        REQUIRE(id2 == 0);
    }

    SECTION("linear lookup without reverse map") {
        LabelTable label_table(allocator.get(), false);
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);

        auto [ok1, id1] = label_table.TryGetIdByLabel(200);
        auto [ok2, id2] = label_table.TryGetIdByLabel(999);

        REQUIRE(ok1);
        REQUIRE(id1 == 1);
        REQUIRE(ok2 == false);
        REQUIRE(id2 == 0);
    }
}
