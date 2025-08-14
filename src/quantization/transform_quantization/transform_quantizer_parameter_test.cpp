
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

#include "transform_quantizer_parameter.h"

#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>

#include "inner_string_params.h"
#include "parameter_test.h"

using namespace vsag;

#define TEST_COMPATIBILITY_CASE(section_name, param_str1, param_str2, expect_compatible) \
    SECTION(section_name) {                                                              \
        auto tq_param1 = std::make_shared<vsag::TransformQuantizerParameter>();          \
        auto tq_param2 = std::make_shared<vsag::TransformQuantizerParameter>();          \
        tq_param1->FromString(param_str1);                                               \
        tq_param2->FromString(param_str2);                                               \
        if (expect_compatible) {                                                         \
            REQUIRE(tq_param1->CheckCompatibility(tq_param2));                           \
        } else {                                                                         \
            REQUIRE_FALSE(tq_param1->CheckCompatibility(tq_param2));                     \
        }                                                                                \
    }

TEST_CASE("Transform Quantizer Parameter CheckCompatibility", "[ut][TransformQuantizerParameter]") {
    auto param_template = R"(
        {{
            "tq_chain": "{}",
            "rabitq_use_fht": true,
            "pca_dim": 960
        }}
    )";
    auto param_pca_rom_rabitq = fmt::format(param_template, "pca, rom, rabitq");
    auto param_pca_rom_fp32 = fmt::format(param_template, "pca, rom, fp32");
    auto param_pca_fht_fp32 = fmt::format(param_template, "pca, fht, fp32");
    auto param_pca_fp32 = fmt::format(param_template, "pca, fp32");
    auto param_pca_fht_fp32_no_space = fmt::format(param_template, "pca,fht,fp32");

    SECTION("wrong parameter type") {
        auto param = std::make_shared<vsag::TransformQuantizerParameter>();
        param->FromString(param_pca_rom_rabitq);
        REQUIRE(param->CheckCompatibility(param));
        REQUIRE_FALSE(param->CheckCompatibility(std::make_shared<vsag::EmptyParameter>()));
    }

    TEST_COMPATIBILITY_CASE(
        "different base quantization", param_pca_rom_rabitq, param_pca_rom_fp32, false);
    TEST_COMPATIBILITY_CASE("different pre-process", param_pca_rom_fp32, param_pca_fht_fp32, false);
    TEST_COMPATIBILITY_CASE("different length", param_pca_fp32, param_pca_fht_fp32, false);
    TEST_COMPATIBILITY_CASE(
        "different space", param_pca_fht_fp32_no_space, param_pca_fht_fp32, true);
}

TEST_CASE("TQ Parameter ToJson Test", "[ut][TransformQuantizerParameter]") {
    std::string param_str = R"(
        {
            "tq_chain": "pca, rom, rabitq",
            "rabitq_use_fht": true,
            "pca_dim": 960
        }
    )";
    auto param = std::make_shared<TransformQuantizerParameter>();
    param->FromJson(JsonType::parse(param_str));
    REQUIRE(param->base_quantizer_json_[QUANTIZATION_TYPE_KEY] == "rabitq");
    REQUIRE(param->tq_chain_.size() == 2);
    ParameterTest::TestToJson(param);
}

TEST_CASE("TQ parameter Split Merge String Test", "[ut][TransformQuantizerParameter]") {
    std::string str = "pca,rom,rabitq";
    auto chain_str = TransformQuantizerParameter::SplitString(str);
    auto recover_str = TransformQuantizerParameter::MergeStrings(chain_str);
    REQUIRE(str == recover_str);
}