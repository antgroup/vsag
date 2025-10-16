
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

#include "scalar_quantizer_parameter.h"

#include <catch2/catch_test_macros.hpp>

#include "parameter_test.h"

using namespace vsag;

TEST_CASE("SQ8 Quantizer Parameter ToJson Test", "[ut][SQ8QuantizerParameter]") {
    std::string param_str = "{}";
    auto param = std::make_shared<ScalarQuantizerParameter<8>>();
    REQUIRE(param->GetTypeName() == "sq8");

    JsonType param_json = JsonType::Parse(param_str);
    param->FromJson(param_json);
    ParameterTest::TestToJson(param);
}

TEST_CASE("SQ4 Quantizer Parameter ToJson Test", "[ut][SQ4QuantizerParameter]") {
    std::string param_str = "{}";
    auto param = std::make_shared<ScalarQuantizerParameter<4>>();
    REQUIRE(param->GetTypeName() == "sq4");

    JsonType param_json = JsonType::Parse(param_str);
    param->FromJson(param_json);
    ParameterTest::TestToJson(param);
}
