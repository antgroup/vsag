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

#include "region_filter_executor.h"

#include "attr/argparse.h"
#include "datacell/attribute_inverted_interface.h"
#include "impl/allocator/safe_allocator.h"
#include "unittest.h"

using namespace vsag;

namespace {

template <typename Type, typename List>
void
InsertTypedAttributes(const AttrInvertedInterfacePtr& index,
                      Type type_value,
                      std::vector<List> region_values,
                      std::vector<List> residence_values) {
    AttributeValue<Type> type;
    type.name_ = "kind";
    type.GetValue() = {type_value};
    AttributeValue<List> region;
    region.name_ = "geo";
    region.GetValue() = std::move(region_values);
    AttributeValue<List> residence;
    residence.name_ = "home";
    residence.GetValue() = std::move(residence_values);
    AttributeSet set{{&type, &region, &residence}};
    index->Insert(set, 0, 0);
}

struct Attributes {
    AttributeValue<int16_t> type;
    AttributeValue<int64_t> region;
    AttributeValue<int64_t> residence;
    AttributeValue<int16_t> decoy_type;
    AttributeValue<int64_t> decoy_region;
    AttributeValue<int64_t> decoy_residence;
    AttributeSet set;

    Attributes(int16_t type_value,
               std::vector<int64_t> region_values,
               std::vector<int64_t> residence_values)
        : set{{&type, &region, &residence, &decoy_type, &decoy_region, &decoy_residence}} {
        type.name_ = "kind";
        type.GetValue() = {type_value};
        region.name_ = "geo";
        region.GetValue() = std::move(region_values);
        residence.name_ = "home";
        residence.GetValue() = std::move(residence_values);
        decoy_type.name_ = "region_type";
        decoy_type.GetValue() = {-1};
        decoy_region.name_ = "region_list";
        decoy_region.GetValue() = {10};
        decoy_residence.name_ = "residence_tag_list";
        decoy_residence.GetValue() = {20, 30};
    }
};

}  // namespace

TEST_CASE("RegionFilterExecutor truth table and expression fields", "[ut][RegionFilterExecutor]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto attr_index = AttributeInvertedInterface::MakeInstance(allocator.get(), true);
    std::vector<Attributes> rows;
    std::vector<bool> expected;
    rows.reserve(32);
    expected.reserve(32);
    auto add_row = [&](int16_t type, bool region, bool residence, bool trigger) {
        std::vector<int64_t> regions;
        std::vector<int64_t> residences;
        if (region) {
            regions.emplace_back(10);
        }
        if (residence) {
            residences.emplace_back(20);
        }
        if (trigger) {
            residences.emplace_back(30);
        }
        rows.emplace_back(type, std::move(regions), std::move(residences));
        switch (type) {
            case -1:
                expected.emplace_back(true);
                break;
            case 0:
                expected.emplace_back(trigger || (region && residence));
                break;
            case 1:
                expected.emplace_back(residence);
                break;
            case 2:
                expected.emplace_back(region);
                break;
            case 3:
                expected.emplace_back(region || residence);
                break;
            case 4:
                expected.emplace_back(region && residence);
                break;
            case 5:
                expected.emplace_back(region && !residence);
                break;
        }
    };
    for (int16_t type = -1; type <= 5; ++type) {
        if (type == 0) {
            for (const bool region : {false, true}) {
                for (const bool residence : {false, true}) {
                    for (const bool trigger : {false, true}) {
                        add_row(type, region, residence, trigger);
                    }
                }
            }
        } else {
            for (const bool region : {false, true}) {
                for (const bool residence : {false, true}) {
                    add_row(type, region, residence, false);
                }
            }
        }
    }
    for (uint64_t id = 0; id < rows.size(); ++id) {
        attr_index->Insert(rows[id].set, id, 3);
    }

    auto expression = AstParse(R"(region_filter(kind,geo,home,"10|404","20|405","30|406"))");
    auto executor = Executor::MakeInstance(allocator.get(), expression, attr_index);
    REQUIRE(std::dynamic_pointer_cast<RegionFilterExecutor>(executor) != nullptr);
    executor->Init();
    auto* filter = executor->Run(3);
    for (uint64_t id = 0; id < expected.size(); ++id) {
        REQUIRE(filter->CheckValid(id) == expected[id]);
    }
    REQUIRE(executor->only_bitset_);

    REQUIRE(executor->Run(7)->CheckValid(int64_t{0}) == false);
    executor->Clear();
    filter = executor->Run(3);
    REQUIRE(filter->CheckValid(int64_t{0}));
    REQUIRE_NOTHROW(executor->Clear());
    REQUIRE_NOTHROW(executor->Clear());
}

TEST_CASE("RegionFilterExecutor validates inputs and lifecycle", "[ut][RegionFilterExecutor]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto attr_index = AttributeInvertedInterface::MakeInstance(allocator.get(), true);
    Attributes row(1, {10}, {20});
    attr_index->Insert(row.set, 0, 0);

    auto expression = AstParse(R"(region_filter(kind,geo,home,"10","20","30"))");
    auto executor = Executor::MakeInstance(allocator.get(), expression, attr_index);
    REQUIRE_THROWS_AS(executor->Run(), VsagException);

    try {
        Executor::MakeInstance(allocator.get(), expression, nullptr);
        FAIL("null attribute index must be rejected");
    } catch (const VsagException& error) {
        REQUIRE(error.error_.type == ErrorType::INVALID_ARGUMENT);
        REQUIRE(std::string(error.what()) == "attribute executor requires an attribute index");
    }

    auto function = AstParse(R"(FUNCTION(time_filter,"1|abc","int32|string"))");
    try {
        Executor::MakeInstance(allocator.get(), function, attr_index);
        FAIL("FUNCTION expression must not be executable");
    } catch (const VsagException& error) {
        REQUIRE(error.error_.type == ErrorType::UNSUPPORTED_INDEX_OPERATION);
        REQUIRE(std::string(error.what()) ==
                "FUNCTION expressions are parsed for compatibility only and are not executable");
    }

    REQUIRE_THROWS_AS(
        std::make_shared<RegionFilterExecutor>(allocator.get(), AstParse("kind = 1"), attr_index),
        VsagException);

    auto malformed = std::make_shared<RegionFilterExpression>(
        std::make_shared<FieldExpression>("kind"),
        std::make_shared<FieldExpression>("geo"),
        std::make_shared<FieldExpression>("home"),
        std::make_shared<StrListConstant>(StrList{"10"}),
        std::make_shared<IntListConstant<int64_t>>(std::vector<int64_t>{20}),
        std::make_shared<IntListConstant<int64_t>>(std::vector<int64_t>{30}));
    REQUIRE_THROWS_AS(
        std::make_shared<RegionFilterExecutor>(allocator.get(), malformed, attr_index),
        VsagException);

    auto sparse_index = AttributeInvertedInterface::MakeInstance(allocator.get(), false);
    sparse_index->Insert(row.set, 0);
    REQUIRE_THROWS_AS(
        std::make_shared<RegionFilterExecutor>(allocator.get(), expression, sparse_index),
        VsagException);

    REQUIRE_THROWS_AS(AstParse(R"(region_filter(kind,geo,home,"10","20",))"), VsagException);
    REQUIRE_THROWS_AS(AstParse(R"(region_filter(kind,geo,home,"10",,"30"))"), VsagException);
    REQUIRE_THROWS_AS(AstParse(R"(region_filter(kind,geo,home,,"20","30"))"), VsagException);
}

TEST_CASE("RegionFilterExecutor checks integer field boundaries", "[ut][RegionFilterExecutor]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    SECTION("signed boundaries are accepted") {
        auto index = AttributeInvertedInterface::MakeInstance(allocator.get(), true);
        InsertTypedAttributes<int8_t, int8_t>(index, -1, {-128, 127}, {-128, 127});
        auto expression =
            AstParse(R"(region_filter(kind,geo,home,"-128|127","-128|127","-128|127"))");
        auto executor = std::make_shared<RegionFilterExecutor>(allocator.get(), expression, index);
        REQUIRE_NOTHROW(executor->Init());
        REQUIRE(executor->Run(0)->CheckValid(int64_t{0}));
    }

    SECTION("signed overflow is rejected") {
        auto index = AttributeInvertedInterface::MakeInstance(allocator.get(), true);
        InsertTypedAttributes<int8_t, int8_t>(index, 1, {0}, {0});
        auto expression = AstParse(R"(region_filter(kind,geo,home,"128","0","0"))");
        REQUIRE_THROWS_AS(
            std::make_shared<RegionFilterExecutor>(allocator.get(), expression, index),
            VsagException);
        expression = AstParse(R"(region_filter(kind,geo,home,"-129","0","0"))");
        REQUIRE_THROWS_AS(
            std::make_shared<RegionFilterExecutor>(allocator.get(), expression, index),
            VsagException);
    }

    SECTION("unsigned list boundaries are checked") {
        auto index = AttributeInvertedInterface::MakeInstance(allocator.get(), true);
        InsertTypedAttributes<int16_t, uint8_t>(index, 2, {0, 255}, {0, 255});
        auto expression = AstParse(R"(region_filter(kind,geo,home,"0|255","0|255","255"))");
        auto executor = std::make_shared<RegionFilterExecutor>(allocator.get(), expression, index);
        executor->Init();
        REQUIRE(executor->Run(0)->CheckValid(int64_t{0}));

        expression = AstParse(R"(region_filter(kind,geo,home,"256","0","0"))");
        REQUIRE_THROWS_AS(
            std::make_shared<RegionFilterExecutor>(allocator.get(), expression, index),
            VsagException);
        expression = AstParse(R"(region_filter(kind,geo,home,"-1","0","0"))");
        REQUIRE_THROWS_AS(
            std::make_shared<RegionFilterExecutor>(allocator.get(), expression, index),
            VsagException);
    }

    SECTION("unsigned type field is rejected because it cannot represent minus one") {
        auto index = AttributeInvertedInterface::MakeInstance(allocator.get(), true);
        InsertTypedAttributes<uint8_t, int16_t>(index, 1, {10}, {20});
        auto expression = AstParse(R"(region_filter(kind,geo,home,"10","20","30"))");
        REQUIRE_THROWS_AS(
            std::make_shared<RegionFilterExecutor>(allocator.get(), expression, index),
            VsagException);
    }
}
