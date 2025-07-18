
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

#include "product_quantizer_parameter.h"

#include <catch2/catch_test_macros.hpp>

#include "parameter_test.h"

using namespace vsag;

TEST_CASE("Product Quantizer Parameter ToJson Test", "[ut][ProductQuantizerParameter]") {
    std::string param_str = R"(
        {
            "pq_dim": 64,
            "pq_bits": 8
        }
    )";
    auto param = std::make_shared<ProductQuantizerParameter>();
    param->FromJson(JsonType::parse(param_str));
    ParameterTest::TestToJson(param);
    REQUIRE(param->pq_bits_ == 8);
    REQUIRE(param->pq_dim_ == 64);

    TestParamCheckCompatibility<ProductQuantizerParameter>(param_str);
}
