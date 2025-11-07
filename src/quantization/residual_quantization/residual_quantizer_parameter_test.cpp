
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

#include "residual_quantizer_parameter.h"

#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "inner_string_params.h"
#include "parameter_test.h"

using namespace vsag;

#define TEST_COMPATIBILITY_CASE(section_name, param_str1, param_str2, expect_compatible) \
    SECTION(section_name) {                                                              \
        auto rq_param1 = std::make_shared<vsag::ResidualQuantizerParameter>();           \
        auto rq_param2 = std::make_shared<vsag::ResidualQuantizerParameter>();           \
        rq_param1->FromString(param_str1);                                               \
        rq_param2->FromString(param_str2);                                               \
        if (expect_compatible) {                                                         \
            REQUIRE(rq_param1->CheckCompatibility(rq_param2));                           \
        } else {                                                                         \
            REQUIRE_FALSE(rq_param1->CheckCompatibility(rq_param2));                     \
        }                                                                                \
    }

TEST_CASE("Residual Quantizer Parameter CheckCompatibility", "[ut][ResidualQuantizerParameter]") {
    constexpr static const char* param_template = R"(
        {{
            "rq_base_quantization_type": "{}",
            "rq_centroids_count": {}
        }}
    )";

    auto param_rabitq_10 = fmt::format(param_template, "rabitq", 10);
    auto param_fp32_10 = fmt::format(param_template, "fp32", 10);
    auto param_fp32_1 = fmt::format(param_template, "fp32", 1);

    SECTION("wrong parameter type") {
        auto param = std::make_shared<vsag::ResidualQuantizerParameter>();
        param->FromString(param_rabitq_10);
        REQUIRE(param->CheckCompatibility(param));
        REQUIRE_FALSE(param->CheckCompatibility(std::make_shared<vsag::EmptyParameter>()));
    }

    TEST_COMPATIBILITY_CASE("different centroids count", param_fp32_1, param_fp32_10, false);
    TEST_COMPATIBILITY_CASE("different base quantizer", param_rabitq_10, param_fp32_10, false);
    TEST_COMPATIBILITY_CASE("same", param_rabitq_10, param_rabitq_10, true);
}

TEST_CASE("RQ Parameter ToJson Test", "[ut][ResidualQuantizerParameter]") {
    std::string quantizer_type = GENERATE("rabitq", "sq4_uniform", "fp32");
    int centroids_count = GENERATE(1, 10, 100);
    constexpr static const char* param_template = R"(
        {{
            "rq_base_quantization_type": "{}",
            "rq_centroids_count": {}
        }}
    )";

    auto param_str = fmt::format(param_template, quantizer_type, centroids_count);
    auto param = std::make_shared<ResidualQuantizerParameter>();
    param->FromJson(JsonType::Parse(param_str));
    REQUIRE(param->base_quantizer_json_[QUANTIZATION_TYPE_KEY].GetString() == quantizer_type);
    REQUIRE(param->centroids_count_ == centroids_count);
    ParameterTest::TestToJson(param);
}

TEST_CASE("Invalid Cases Test", "[ut][ResidualQuantizerParameter]") {
    constexpr static const char* param_template = R"(
        {{
            "rq_centroids_count": {}
        }}
    )";
    auto invalid_param_str = fmt::format(param_template, 10);

    auto param = std::make_shared<ResidualQuantizerParameter>();

    REQUIRE_THROWS(param->FromString(invalid_param_str));
}
