
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

#include "sq4_uniform_quantizer_parameter.h"

#include <catch2/catch_test_macros.hpp>

#include "parameter_test.h"

using namespace vsag;

TEST_CASE("SQ4 Uniform Quantizer Parameter ToJson Test", "[ut][SQ4UniformQuantizerParameter]") {
    std::string param_str = R"(
        {
            "sq4_uniform_trunc_rate": 0.06
        }
    )";
    auto param = std::make_shared<SQ4UniformQuantizerParameter>();
    param->FromJson(JsonType::parse(param_str));
    REQUIRE(std::abs(param->trunc_rate_ - 0.06) < 1e-5F);
    ParameterTest::TestToJson(param);

    TestParamCheckCompatibility<SQ4UniformQuantizerParameter>(param_str);
}
