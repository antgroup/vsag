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
#include <iostream>
#include <memory>

#include "impl/allocator/default_allocator.h"

using namespace vsag;

TEST_CASE("LabelTable Basic Operations", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    LabelTable label_table(allocator.get());

    SECTION("Insert and GetLabelById") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);
        label_table.Insert(2, 300);

        REQUIRE(label_table.GetLabelById(0) == 100);
        REQUIRE(label_table.GetLabelById(1) == 200);
        REQUIRE(label_table.GetLabelById(2) == 300);
    }

    SECTION("GetIdByLabel with reverse map") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);
        label_table.Insert(2, 300);

        REQUIRE(label_table.GetIdByLabel(100) == 0);
        REQUIRE(label_table.GetIdByLabel(200) == 1);
        REQUIRE(label_table.GetIdByLabel(300) == 2);
    }

    SECTION("CheckLabel") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);

        REQUIRE(label_table.CheckLabel(100) == true);
        REQUIRE(label_table.CheckLabel(200) == true);
        REQUIRE(label_table.CheckLabel(300) == false);
    }

    SECTION("MarkRemove and IsRemoved") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);
        label_table.Insert(2, 300);

        REQUIRE(label_table.CheckLabel(100) == true);
        label_table.MarkRemove(100);
        REQUIRE(label_table.IsRemoved(0) == true);
        REQUIRE(label_table.CheckLabel(100) == false);
    }

    SECTION("GetIdByLabel with removed label") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);
        label_table.MarkRemove(100);

        REQUIRE_THROWS_AS(label_table.GetIdByLabel(100), VsagException);
        REQUIRE(label_table.GetIdByLabel(100, true) == 0);  // return even removed
    }

    SECTION("SetImmutable disables reverse map") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);

        REQUIRE(label_table.use_reverse_map_ == true);
        label_table.SetImmutable();
        REQUIRE(label_table.use_reverse_map_ == false);

        // Should still work but using linear search
        REQUIRE(label_table.GetIdByLabel(100) == 0);
        REQUIRE(label_table.GetIdByLabel(200) == 1);
    }
}

TEST_CASE("LabelTable Without Reverse Map", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    LabelTable label_table(allocator.get(), false);  // disable reverse map

    SECTION("Insert and GetIdByLabel without reverse map") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);
        label_table.Insert(2, 300);

        REQUIRE(label_table.GetIdByLabel(100) == 0);
        REQUIRE(label_table.GetIdByLabel(200) == 1);
        REQUIRE(label_table.GetIdByLabel(300) == 2);
    }
}

TEST_CASE("LabelTable Memory Management", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    LabelTable label_table(allocator.get());

    SECTION("GetTotalCount") {
        REQUIRE(label_table.GetTotalCount() == 0);

        label_table.Insert(0, 100);
        REQUIRE(label_table.GetTotalCount() == 1);

        label_table.Insert(1, 200);
        REQUIRE(label_table.GetTotalCount() == 2);
    }

    SECTION("Resize") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);

        label_table.Resize(10);
        REQUIRE(label_table.GetTotalCount() == 2);

        // Should be able to insert at new positions
        label_table.Insert(9, 900);
        REQUIRE(label_table.GetLabelById(9) == 900);
    }

    SECTION("GetMemoryUsage") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);

        auto usage = label_table.GetMemoryUsage();
        REQUIRE(usage > 0);
    }
}

TEST_CASE("LabelTable Filter Operations", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    LabelTable label_table(allocator.get());

    SECTION("GetDeletedIdsFilter with no deletions") {
        auto filter = label_table.GetDeletedIdsFilter();
        REQUIRE(filter == nullptr);
    }

    SECTION("GetDeletedIdsFilter with deletions") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);
        label_table.MarkRemove(100);

        auto filter = label_table.GetDeletedIdsFilter();
        REQUIRE(filter != nullptr);
    }
}

TEST_CASE("LabelTable Edge Cases", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    LabelTable label_table(allocator.get());

    SECTION("GetLabelById with invalid id") {
        label_table.Insert(0, 100);

        REQUIRE_THROWS_AS(label_table.GetLabelById(1), VsagException);
        REQUIRE_THROWS_AS(label_table.GetLabelById(1000), VsagException);
    }

    SECTION("GetIdByLabel with non-existent label") {
        REQUIRE_THROWS_AS(label_table.GetIdByLabel(999), VsagException);
    }

    SECTION("Insert at large id") {
        label_table.Insert(1000, 5000);
        REQUIRE(label_table.GetLabelById(1000) == 5000);
        REQUIRE(label_table.GetIdByLabel(5000) == 1000);
    }
}
