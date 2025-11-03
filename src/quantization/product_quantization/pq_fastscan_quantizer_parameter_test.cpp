
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

#include "pq_fastscan_quantizer_parameter.h"

#include <catch2/catch_test_macros.hpp>

#include "parameter_test.h"

using namespace vsag;

TEST_CASE("PQFS Parameter ToJson Test", "[ut][PQFastScanQuantizerParameter]") {
    std::string param_str = R"(
        {
            "pq_dim": 64,
            "pq_train_sample_size": 65535
        }
    )";
    auto param = std::make_shared<PQFastScanQuantizerParameter>();
    param->FromJson(JsonType::Parse(param_str));
    ParameterTest::TestToJson(param);
    REQUIRE(param->pq_dim_ == 64);
    REQUIRE(param->train_sample_size_ == 65535);

    TestParamCheckCompatibility<PQFastScanQuantizerParameter>(param_str);
}

TEST_CASE("PQFS Parameter Invalid Values Test", "[ut][PQFastScanQuantizerParameter]") {
    // Test with invalid pq_dim (zero)
    std::string param_str_zero_pq_dim = R"(
        {
            "pq_dim": 0,
            "pq_train_sample_size": 65535
        }
    )";
    auto param_zero_pq_dim = std::make_shared<PQFastScanQuantizerParameter>();
    param_zero_pq_dim->FromJson(JsonType::Parse(param_str_zero_pq_dim));
    REQUIRE(param_zero_pq_dim->pq_dim_ == 0);

    // Test with invalid pq_dim (negative)
    std::string param_str_negative_pq_dim = R"(
        {
            "pq_dim": -1,
            "pq_train_sample_size": 65535
        }
    )";
    auto param_negative_pq_dim = std::make_shared<PQFastScanQuantizerParameter>();
    param_negative_pq_dim->FromJson(JsonType::Parse(param_str_negative_pq_dim));
    REQUIRE(param_negative_pq_dim->pq_dim_ == -1);

    // Test with invalid train_sample_size (zero)
    std::string param_str_zero_train_size = R"(
        {
            "pq_dim": 64,
            "pq_train_sample_size": 0
        }
    )";
    auto param_zero_train_size = std::make_shared<PQFastScanQuantizerParameter>();
    param_zero_train_size->FromJson(JsonType::Parse(param_str_zero_train_size));
    REQUIRE(param_zero_train_size->train_sample_size_ == 0);

    // Test with invalid train_sample_size (negative)
    std::string param_str_negative_train_size = R"(
        {
            "pq_dim": 64,
            "pq_train_sample_size": -1
        }
    )";
    auto param_negative_train_size = std::make_shared<PQFastScanQuantizerParameter>();
    param_negative_train_size->FromJson(JsonType::Parse(param_str_negative_train_size));
    REQUIRE(param_negative_train_size->train_sample_size_ == -1);

    // Test compatibility checks with invalid values
    auto valid_param = std::make_shared<PQFastScanQuantizerParameter>();
    std::string valid_param_str = R"(
        {
            "pq_dim": 64,
            "pq_train_sample_size": 65535
        }
    )";
    valid_param->FromJson(JsonType::Parse(valid_param_str));

    // Valid param should be compatible with itself
    REQUIRE(valid_param->CheckCompatibility(valid_param));

    // Valid param should not be compatible with invalid params
    REQUIRE_FALSE(valid_param->CheckCompatibility(param_zero_pq_dim));
    REQUIRE_FALSE(valid_param->CheckCompatibility(param_negative_pq_dim));
    REQUIRE_FALSE(valid_param->CheckCompatibility(param_zero_train_size));
    REQUIRE_FALSE(valid_param->CheckCompatibility(param_negative_train_size));
}

TEST_CASE("PQFS Parameter Default Values Test", "[ut][PQFastScanQuantizerParameter]") {
    auto param = std::make_shared<PQFastScanQuantizerParameter>();
    // Check default values
    REQUIRE(param->pq_dim_ == 1);
    REQUIRE(param->train_sample_size_ == 65536UL);

    // Test ToJson with default values
    auto json = param->ToJson();
    REQUIRE(json.Contains("pq_dim"));
    REQUIRE(json.Contains("pq_train_sample_size"));
    REQUIRE(json["pq_dim"].GetInt() == 1);
    REQUIRE(json["pq_train_sample_size"].GetInt() == 65536);
}

TEST_CASE("PQFS Parameter FromJson Partial Fields Test", "[ut][PQFastScanQuantizerParameter]") {
    // Test with only pq_dim specified
    std::string param_str1 = R"(
        {
            "pq_dim": 32
        }
    )";
    auto param1 = std::make_shared<PQFastScanQuantizerParameter>();
    param1->FromJson(JsonType::Parse(param_str1));
    REQUIRE(param1->pq_dim_ == 32);
    REQUIRE(param1->train_sample_size_ == 65536UL);  // Should keep default value

    // Test with only train_sample_size specified
    std::string param_str2 = R"(
        {
            "pq_train_sample_size": 32768
        }
    )";
    auto param2 = std::make_shared<PQFastScanQuantizerParameter>();
    param2->FromJson(JsonType::Parse(param_str2));
    REQUIRE(param2->pq_dim_ == 1);  // Should keep default value
    REQUIRE(param2->train_sample_size_ == 32768);

    // Test with empty JSON
    std::string param_str3 = R"({})";
    auto param3 = std::make_shared<PQFastScanQuantizerParameter>();
    param3->FromJson(JsonType::Parse(param_str3));
    REQUIRE(param3->pq_dim_ == 1);                   // Should keep default value
    REQUIRE(param3->train_sample_size_ == 65536UL);  // Should keep default value
}

TEST_CASE("PQFS Parameter CheckCompatibility Test", "[ut][PQFastScanQuantizerParameter]") {
    // Test compatibility with self
    std::string param_str = R"(
        {
            "pq_dim": 64,
            "pq_train_sample_size": 65535
        }
    )";
    auto param1 = std::make_shared<PQFastScanQuantizerParameter>();
    auto param2 = std::make_shared<PQFastScanQuantizerParameter>();
    param1->FromJson(JsonType::Parse(param_str));
    param2->FromJson(JsonType::Parse(param_str));

    REQUIRE(param1->CheckCompatibility(param1));  // Self compatibility
    REQUIRE(param1->CheckCompatibility(param2));  // Same values compatibility

    // Test incompatibility with different pq_dim
    std::string param_str_different_pq_dim = R"(
        {
            "pq_dim": 32,
            "pq_train_sample_size": 65535
        }
    )";
    auto param3 = std::make_shared<PQFastScanQuantizerParameter>();
    param3->FromJson(JsonType::Parse(param_str_different_pq_dim));
    REQUIRE_FALSE(param1->CheckCompatibility(param3));

    // Test incompatibility with different train_sample_size
    std::string param_str_different_train_size = R"(
        {
            "pq_dim": 64,
            "pq_train_sample_size": 32768
        }
    )";
    auto param4 = std::make_shared<PQFastScanQuantizerParameter>();
    param4->FromJson(JsonType::Parse(param_str_different_train_size));
    REQUIRE_FALSE(param1->CheckCompatibility(param4));

    // Test incompatibility with different parameter type
    TestParamCheckCompatibility<PQFastScanQuantizerParameter>(param_str);
}
