
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

#include "bruteforce_parameter.h"

#include "bruteforce.h"
#include "parameter_test.h"
#include "unittest.h"

TEST_CASE("BruteForce Parameters CheckCompatibility",
          "[ut][BruteForceParameter][CheckCompatibility]") {
    const auto* param_str = R"({
        "base_codes": {
            "codes_type": "flatten_codes",
            "io_params": {
                "type": "block_memory_io"
            },
            "quantization_params": {
                "type": "fp32"
            }
        },
        "type": "brute_force",
        "use_attribute_filter": true
    })";

    SECTION("wrong parameter type") {
        auto param = std::make_shared<vsag::BruteForceParameter>();
        param->FromString(param_str);
        REQUIRE(param->CheckCompatibility(param));
        REQUIRE(param->use_attribute_filter == true);
        REQUIRE_FALSE(param->CheckCompatibility(std::make_shared<vsag::EmptyParameter>()));
    }
}

TEST_CASE("BruteForce maps resize increase count bit", "[ut][BruteForceParameter]") {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;

    auto default_param = std::dynamic_pointer_cast<vsag::BruteForceParameter>(
        vsag::BruteForce::CheckAndMappingExternalParam(vsag::JsonType::Parse("{}"), common_param));
    REQUIRE(default_param != nullptr);
    REQUIRE(default_param->resize_increase_count_bit == vsag::DEFAULT_RESIZE_INCREASE_COUNT_BIT);

    auto configured_param = std::dynamic_pointer_cast<vsag::BruteForceParameter>(
        vsag::BruteForce::CheckAndMappingExternalParam(
            vsag::JsonType::Parse(R"({"resize_increase_count_bit": 1})"), common_param));
    REQUIRE(configured_param != nullptr);
    REQUIRE(configured_param->resize_increase_count_bit == 1);
    REQUIRE(configured_param->ToJson()[vsag::RESIZE_INCREASE_COUNT_BIT].GetUint64() == 1);

    REQUIRE_THROWS(vsag::BruteForce::CheckAndMappingExternalParam(
        vsag::JsonType::Parse(R"({"resize_increase_count_bit": 0})"), common_param));
    REQUIRE_THROWS(vsag::BruteForce::CheckAndMappingExternalParam(
        vsag::JsonType::Parse(R"({"resize_increase_count_bit": 32})"), common_param));
    REQUIRE_THROWS(vsag::BruteForce::CheckAndMappingExternalParam(
        vsag::JsonType::Parse(R"({"resize_increase_count_bit": 1.5})"), common_param));
    REQUIRE_THROWS(vsag::BruteForce::CheckAndMappingExternalParam(
        vsag::JsonType::Parse(R"({"resize_increase_count_bit": -1})"), common_param));
}

TEST_CASE("BruteForce maps store raw vector to flatten quantizers", "[ut][BruteForceParameter]") {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;

    auto parameter = std::dynamic_pointer_cast<vsag::BruteForceParameter>(
        vsag::BruteForce::CheckAndMappingExternalParam(
            vsag::JsonType::Parse(R"({"store_raw_vector": true, "use_residual": true})"),
            common_param));
    REQUIRE(parameter != nullptr);
    const auto json = parameter->ToJson();
    REQUIRE(json[vsag::BASE_CODES_KEY][vsag::QUANTIZATION_PARAMS_KEY][vsag::HOLD_MOLDS].GetBool());
    REQUIRE(
        json[vsag::PRECISE_CODES_KEY][vsag::QUANTIZATION_PARAMS_KEY][vsag::HOLD_MOLDS].GetBool());
    REQUIRE_FALSE(json.Contains(vsag::QUANTIZATION_PARAMS_KEY));
}
