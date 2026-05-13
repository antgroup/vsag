#include "label_table.h"

#include <catch2/catch_test_macros.hpp>

#include "impl/allocator/default_allocator.h"

using namespace vsag;

TEST_CASE("LabelTable Supports Configurable Remap Implementation", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();

    SECTION("robin remap works") {
        LabelTable label_table(allocator.get(), true, false, LabelRemapType::ROBIN);
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);

        REQUIRE(label_table.GetRemapSize() == 2);
        REQUIRE(label_table.GetIdByLabel(100) == 0);
        REQUIRE(label_table.GetIdByLabel(200) == 1);
    }

    SECTION("pg remap remains default") {
        LabelTable label_table(allocator.get(), true, false, LabelRemapType::PG);
        label_table.Insert(0, 100);

        REQUIRE(label_table.GetRemapSize() == 1);
        REQUIRE(label_table.GetIdByLabel(100) == 0);
    }
}
