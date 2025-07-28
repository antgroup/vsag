
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

#include "vector_transformer_parameter.h"

#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>

#include "inner_string_params.h"
#include "parameter_test.h"

using namespace vsag;

#define TEST_COMPATIBILITY_CASE(section_name, param_str1, param_str2, expect_compatible) \
    SECTION(section_name) {                                                              \
        auto tq_param1 = std::make_shared<vsag::VectorTransformerParameter>();           \
        auto tq_param2 = std::make_shared<vsag::VectorTransformerParameter>();           \
        tq_param1->FromString(param_str1);                                               \
        tq_param2->FromString(param_str2);                                               \
        if (expect_compatible) {                                                         \
            REQUIRE(tq_param1->CheckCompatibility(tq_param2));                           \
        } else {                                                                         \
            REQUIRE_FALSE(tq_param1->CheckCompatibility(tq_param2));                     \
        }                                                                                \
    }

TEST_CASE("Transformer Parameter CheckCompatibility", "[ut][VectorTransformerParameter]") {
    auto param_template = R"(
        {{
            "input_dim": {},
            "pca_dim": {}
        }}
    )";
    auto param_960_480 = fmt::format(param_template, 960, 480);
    auto param_959_480 = fmt::format(param_template, 959, 480);
    auto param_960_959 = fmt::format(param_template, 960, 959);

    SECTION("wrong parameter type") {
        auto param = std::make_shared<vsag::VectorTransformerParameter>();
        param->FromString(param_960_480);
        REQUIRE(param->CheckCompatibility(param));
        REQUIRE_FALSE(param->CheckCompatibility(std::make_shared<vsag::EmptyParameter>()));
    }

    TEST_COMPATIBILITY_CASE("different pca_dim", param_960_480, param_960_959, false);
    TEST_COMPATIBILITY_CASE("different input_dim", param_960_480, param_959_480, false);
    TEST_COMPATIBILITY_CASE("same", param_960_480, param_960_480, true);
}

TEST_CASE("Transformer Parameter ToJson Test", "[ut][VectorTransformerParameter]") {
    std::string param_str = R"(
        {
            "input_dim": 960,
            "pca_dim": 480
        }
    )";
    auto param = std::make_shared<VectorTransformerParameter>();
    param->FromJson(JsonType::parse(param_str));
    REQUIRE(param->input_dim_ == 960);
    REQUIRE(param->pca_dim_ == 480);
    ParameterTest::TestToJson(param);
}