
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

#include "bucket_datacell_parameter.h"

#include <catch2/catch_test_macros.hpp>

#include "parameter_test.h"

using namespace vsag;

TEST_CASE("BucketDataCellParameter ToJson Test", "[ut][BucketDataCellParameter]") {
    std::string param_str = R"(
    {
        "io_params": {
            "type": "memory_io"
        },
        "quantization_params": {
            "type": "sq8"
        },
        "buckets_count": 10
    })";
    auto param = std::make_shared<BucketDataCellParameter>();
    auto json = JsonType::parse(param_str);
    param->FromJson(json);
    REQUIRE(param->buckets_count == 10);
    ParameterTest::TestToJson(param);
}

TEST_CASE("BucketDataCellParameter Parse Exception", "[ut][BucketDataCellParameter]") {
    auto check_param = [](const std::string& str) -> BucketDataCellParamPtr {
        auto param = std::make_shared<BucketDataCellParameter>();
        auto json = JsonType::parse(str);
        param->FromJson(json);
        return param;
    };

    SECTION("miss io param") {
        std::string param_str = R"(
        {
            "quantization_params": {
                "type": "sq8",
            },
            "buckets_count": 10
        })";
        REQUIRE_THROWS(check_param(param_str));
    }

    SECTION("miss quantization param") {
        std::string param_str = R"(
        {
            "io_params": {
                "type": "memory_io"
            },
            "buckets_count": 10
        })";
        REQUIRE_THROWS(check_param(param_str));
    }

    SECTION("wrong io param type") {
        std::string param_str = R"(
        {
            "io_params": {
                "type": "wrong_io"
            },
            "buckets_count": 10
        })";
        REQUIRE_THROWS(check_param(param_str));
    }

    SECTION("wrong quantization param type") {
        std::string param_str = R"(
        {
            "quantization_params": {
                "type": "wrong_quantization",
            },
            "buckets_count": 10
        })";
        REQUIRE_THROWS(check_param(param_str));
    }

    SECTION("valid on missing buckets_count") {
        std::string param_str = R"(
        {
            "io_params": {
                "type": "memory_io"
            },
            "quantization_params": {
                "type": "sq8"
            }
        })";
        auto param = check_param(param_str);
    }
}

TEST_CASE("bucket CheckCompatibility", "[ut][BucketDataCellParameter]") {
    std::string param_str = R"(
    {
        "io_params": {
            "type": "memory_io"
        },
        "quantization_params": {
            "type": "sq8"
        },
        "buckets_count": 10
    })";
    auto param = std::make_shared<BucketDataCellParameter>();
    param->FromString(param_str);
    REQUIRE(param->CheckCompatibility(param));
    auto other_type_param = std::make_shared<vsag::EmptyParameter>();
    REQUIRE_FALSE(param->CheckCompatibility(other_type_param));
}
