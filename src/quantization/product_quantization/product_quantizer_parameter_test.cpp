
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
            "pq_bits": 8,
            "pq_train_sample_size": 65535
        }
    )";
    auto param = std::make_shared<ProductQuantizerParameter>();
    param->FromJson(JsonType::Parse(param_str));
    ParameterTest::TestToJson(param);

    REQUIRE(param->pq_bits_ == 8);
    
    REQUIRE(param->pq_dim_ == 64);
    
    REQUIRE(param->train_sample_size_ == 65535);
    

    TestParamCheckCompatibility<ProductQuantizerParameter>(param_str);
}

TEST_CASE("Product Quantizer Parameter Default Values Test", "[ut][ProductQuantizerParameter]") {
    auto param = std::make_shared<ProductQuantizerParameter>();
    // Check default values
    REQUIRE(param->pq_dim_ == 1);
    REQUIRE(param->pq_bits_ == 8);
    REQUIRE(param->train_sample_size_ == 65536);
    
    // Test ToJson with default values
    auto json = param->ToJson();
    REQUIRE(json.Contains("pq_dim"));
    REQUIRE(json.Contains("pq_bits"));
    REQUIRE(json.Contains("pq_train_sample_size"));
    REQUIRE(json["pq_dim"].GetInt() == 1);
    REQUIRE(json["pq_bits"].GetInt() == 8);
    REQUIRE(json["pq_train_sample_size"].GetInt() == 65536);
}

TEST_CASE("Product Quantizer Parameter FromJson Partial Fields Test", "[ut][ProductQuantizerParameter]") {
    // Test with only pq_dim specified
    std::string param_str1 = R"(
        {
            "pq_dim": 32
        }
    )";
    auto param1 = std::make_shared<ProductQuantizerParameter>();
    param1->FromJson(JsonType::Parse(param_str1));
    REQUIRE(param1->pq_dim_ == 32);
    REQUIRE(param1->pq_bits_ == 8); // Should keep default value
    REQUIRE(param1->train_sample_size_ == 65536); // Should keep default value
    
    // Test with only pq_bits specified
    std::string param_str2 = R"(
        {
            "pq_bits": 4
        }
    )";
    auto param2 = std::make_shared<ProductQuantizerParameter>();
    param2->FromJson(JsonType::Parse(param_str2));
    REQUIRE(param2->pq_dim_ == 1); // Should keep default value
    REQUIRE(param2->pq_bits_ == 4);
    REQUIRE(param2->train_sample_size_ == 65536); // Should keep default value
    
    // Test with only train_sample_size specified
    std::string param_str3 = R"(
        {
            "pq_train_sample_size": 32768
        }
    )";
    auto param3 = std::make_shared<ProductQuantizerParameter>();
    param3->FromJson(JsonType::Parse(param_str3));
    REQUIRE(param3->pq_dim_ == 1); // Should keep default value
    REQUIRE(param3->pq_bits_ == 8); // Should keep default value
    REQUIRE(param3->train_sample_size_ == 32768);
    
    // Test with empty JSON
    std::string param_str4 = R"({})";
    auto param4 = std::make_shared<ProductQuantizerParameter>();
    param4->FromJson(JsonType::Parse(param_str4));
    REQUIRE(param4->pq_dim_ == 1); // Should keep default value
    REQUIRE(param4->pq_bits_ == 8); // Should keep default value
    REQUIRE(param4->train_sample_size_ == 65536); // Should keep default value
}

TEST_CASE("Product Quantizer Parameter CheckCompatibility Test", "[ut][ProductQuantizerParameter]") {
    // Test compatibility with self
    std::string param_str = R"(
        {
            "pq_dim": 64,
            "pq_bits": 8,
            "pq_train_sample_size": 65535
        }
    )";
    auto param1 = std::make_shared<ProductQuantizerParameter>();
    auto param2 = std::make_shared<ProductQuantizerParameter>();
    param1->FromJson(JsonType::Parse(param_str));
    param2->FromJson(JsonType::Parse(param_str));
    
    REQUIRE(param1->CheckCompatibility(param1)); // Self compatibility
    REQUIRE(param1->CheckCompatibility(param2)); // Same values compatibility
    
    // Test incompatibility with different pq_dim
    std::string param_str_different_pq_dim = R"(
        {
            "pq_dim": 32,
            "pq_bits": 8,
            "pq_train_sample_size": 65535
        }
    )";
    auto param3 = std::make_shared<ProductQuantizerParameter>();
    param3->FromJson(JsonType::Parse(param_str_different_pq_dim));
    REQUIRE_FALSE(param1->CheckCompatibility(param3));
    
    // Test incompatibility with different pq_bits
    std::string param_str_different_pq_bits = R"(
        {
            "pq_dim": 64,
            "pq_bits": 4,
            "pq_train_sample_size": 65535
        }
    )";
    auto param4 = std::make_shared<ProductQuantizerParameter>();
    param4->FromJson(JsonType::Parse(param_str_different_pq_bits));
    REQUIRE_FALSE(param1->CheckCompatibility(param4));
    
    // Test incompatibility with different train_sample_size
    std::string param_str_different_train_size = R"(
        {
            "pq_dim": 64,
            "pq_bits": 8,
            "pq_train_sample_size": 32768
        }
    )";
    auto param5 = std::make_shared<ProductQuantizerParameter>();
    param5->FromJson(JsonType::Parse(param_str_different_train_size));
    REQUIRE_FALSE(param1->CheckCompatibility(param5));
    
    // Test incompatibility with different parameter type
    TestParamCheckCompatibility<ProductQuantizerParameter>(param_str);
}

TEST_CASE("Product Quantizer Parameter Invalid Values Test", "[ut][ProductQuantizerParameter]") {
    // Test with invalid pq_dim (zero and negative values)
    std::string param_str_invalid_pq_dim_zero = R"(
        {
            "pq_dim": 0,
            "pq_bits": 8,
            "pq_train_sample_size": 65535
        }
    )";
    auto param_zero_dim = std::make_shared<ProductQuantizerParameter>();
    REQUIRE_THROWS_AS(param_zero_dim->FromJson(JsonType::Parse(param_str_invalid_pq_dim_zero)), VsagException);
    
    std::string param_str_invalid_pq_dim_negative = R"(
        {
            "pq_dim": -1,
            "pq_bits": 8,
            "pq_train_sample_size": 65535
        }
    )";
    auto param_negative_dim = std::make_shared<ProductQuantizerParameter>();
    REQUIRE_THROWS_AS(param_negative_dim->FromJson(JsonType::Parse(param_str_invalid_pq_dim_negative)), VsagException);
    
    // Test with invalid pq_bits (zero, negative and too large values)
    std::string param_str_invalid_pq_bits_zero = R"(
        {
            "pq_dim": 64,
            "pq_bits": 0,
            "pq_train_sample_size": 65535
        }
    )";
    auto param_zero_bits = std::make_shared<ProductQuantizerParameter>();
    param_zero_bits->FromJson(JsonType::Parse(param_str_invalid_pq_bits_zero));
    // Should keep default value
    REQUIRE(param_zero_bits->pq_bits_ == 8);
    
    std::string param_str_invalid_pq_bits_negative = R"(
        {
            "pq_dim": 64,
            "pq_bits": -1,
            "pq_train_sample_size": 65535
        }
    )";
    auto param_negative_bits = std::make_shared<ProductQuantizerParameter>();
    param_negative_bits->FromJson(JsonType::Parse(param_str_invalid_pq_bits_negative));
    // Should keep default value
    REQUIRE(param_negative_bits->pq_bits_ == 8);
    
    std::string param_str_invalid_pq_bits_large = R"(
        {
            "pq_dim": 64,
            "pq_bits": 64,
            "pq_train_sample_size": 65535
        }
    )";
    auto param_large_bits = std::make_shared<ProductQuantizerParameter>();
    param_large_bits->FromJson(JsonType::Parse(param_str_invalid_pq_bits_large));
    // Should keep default value
    REQUIRE(param_large_bits->pq_bits_ == 8);
    
    // Test with invalid train_sample_size (zero and negative values)
    std::string param_str_invalid_train_size_zero = R"(
        {
            "pq_dim": 64,
            "pq_bits": 8,
            "pq_train_sample_size": 0
        }
    )";
    auto param_zero_train_size = std::make_shared<ProductQuantizerParameter>();
    param_zero_train_size->FromJson(JsonType::Parse(param_str_invalid_train_size_zero));
    // Should keep default value
    REQUIRE(param_zero_train_size->train_sample_size_ == 65536);
    
    std::string param_str_invalid_train_size_negative = R"(
        {
            "pq_dim": 64,
            "pq_bits": 8,
            "pq_train_sample_size": -1
        }
    )";
    auto param_negative_train_size = std::make_shared<ProductQuantizerParameter>();
    param_negative_train_size->FromJson(JsonType::Parse(param_str_invalid_train_size_negative));
    // Should keep default value
    REQUIRE(param_negative_train_size->train_sample_size_ == 65536);
}
