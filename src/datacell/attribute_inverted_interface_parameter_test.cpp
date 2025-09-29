
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

#include "attribute_inverted_interface_parameter.h"

#include <catch2/catch_test_macros.hpp>

#include "parameter_test.h"

using namespace vsag;

TEST_CASE("AttributeInvertedInterfaceParameter ToJson Test",
          "[ut][AttributeInvertedInterfaceParameter]") {
    std::string param_str = R"(
    {
        "has_buckets": true
    })";
    auto param = std::make_shared<AttributeInvertedInterfaceParameter>();
    auto json = JsonType::Parse(param_str);
    param->FromJson(json);
    REQUIRE(param->has_buckets_ == true);
    ParameterTest::TestToJson(param);

    param_str = R"(
    {
        "has_buckets": false
    })";
    json = JsonType::Parse(param_str);
    param->FromJson(json);
    REQUIRE(param->has_buckets_ == false);
    ParameterTest::TestToJson(param);
}

TEST_CASE("AttributeInvertedInterfaceParameter CheckCompatibility Test",
          "[ut][AttributeInvertedInterfaceParameter]") {
    std::string param_str = R"(
    {
        "has_buckets": true
    })";
    auto param = std::make_shared<AttributeInvertedInterfaceParameter>();
    auto json = JsonType::Parse(param_str);
    param->FromJson(json);

    std::string other_param_str = R"(
    {
        "has_buckets": false
    })";
    auto other_json = JsonType::Parse(other_param_str);
    auto other_param = std::make_shared<AttributeInvertedInterfaceParameter>();
    other_param->FromJson(other_json);
    REQUIRE(param->CheckCompatibility(other_param) == false);
}
