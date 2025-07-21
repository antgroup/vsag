
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

#include "gno_imi_parameter.h"

#include <catch2/catch_test_macros.hpp>

#include "parameter_test.h"

TEST_CASE("GNO-IMI Parameters Test", "[ut][GNOIMIParameter]") {
    auto param_str = R"({
        "first_order_buckets_count": 200,
        "second_order_buckets_count": 50
    })";
    vsag::JsonType param_json = vsag::JsonType::parse(param_str);
    auto param = std::make_shared<vsag::GNOIMIParameter>();
    param->FromJson(param_json);
    REQUIRE(param->first_order_buckets_count == 200);
    REQUIRE(param->second_order_buckets_count == 50);
    vsag::ParameterTest::TestToJson(param);
}

TEST_CASE("GNO-IMI CheckCompatibility", "[ut][GNOIMIParameter]") {
    auto param = std::make_shared<vsag::GNOIMIParameter>();
    param->first_order_buckets_count = 200;
    param->second_order_buckets_count = 50;

    // Check compatibility with itself
    REQUIRE(param->CheckCompatibility(param));

    // Check compatibility with a different GNO-IMI parameter
    auto other_param1 = std::make_shared<vsag::GNOIMIParameter>();
    other_param1->first_order_buckets_count = 100;
    other_param1->second_order_buckets_count = 50;
    REQUIRE_FALSE(param->CheckCompatibility(other_param1));

    // Check compatibility with a different GNO-IMI parameter
    auto other_param2 = std::make_shared<vsag::GNOIMIParameter>();
    other_param2->first_order_buckets_count = 200;
    other_param2->second_order_buckets_count = 100;
    REQUIRE_FALSE(param->CheckCompatibility(other_param2));

    // Check compatibility with a different parameter type
    auto other_type_param = std::make_shared<vsag::EmptyParameter>();
    REQUIRE_FALSE(param->CheckCompatibility(other_type_param));
}
