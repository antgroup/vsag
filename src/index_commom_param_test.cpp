
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

#include <catch2/catch_test_macros.hpp>

#include "factory/resource_owner_wrapper.h"
#include "index_common_param.h"

TEST_CASE("IndexCommonParam Basic Test", "[ut][IndexCommonParam]") {
    std::shared_ptr<vsag::Resource> resource =
        std::make_shared<vsag::ResourceOwnerWrapper>(new vsag::Resource(), true);
    SECTION("wrong metric type") {
        auto build_parameter_json = R"(
        {
            "metric_type": "unknown type",
            "dtype": "float32",
            "dim": 12
        }
        )";
        auto parsed_params = vsag::JsonType::Parse(build_parameter_json);
        REQUIRE_THROWS(vsag::IndexCommonParam::CheckAndCreate(parsed_params, resource));
    }

    SECTION("wrong data type") {
        auto build_parameter_json = R"(
        {
            "metric_type": "l2",
            "dtype": "unknown type",
            "dim": 12
        }
        )";
        auto parsed_params = vsag::JsonType::Parse(build_parameter_json);
        REQUIRE_THROWS(vsag::IndexCommonParam::CheckAndCreate(parsed_params, resource));
    }

    SECTION("wrong dim") {
        auto build_parameter_json = R"(
        {
            "metric_type": "l2",
            "dtype": "float32",
            "dim": -1
        }
        )";
        auto parsed_params = vsag::JsonType::Parse(build_parameter_json);
        REQUIRE_THROWS(vsag::IndexCommonParam::CheckAndCreate(parsed_params, resource));
    }

    SECTION("success") {
        auto build_parameter_json = R"(
        {
            "metric_type": "l2",
            "dtype": "float32",
            "dim": 12,
            "extra_info_size": 38
        }
        )";
        auto parsed_params = vsag::JsonType::Parse(build_parameter_json);
        auto param = vsag::IndexCommonParam::CheckAndCreate(parsed_params, resource);
        REQUIRE(param.metric_ == vsag::MetricType::METRIC_TYPE_L2SQR);
        REQUIRE(param.dim_ == 12);
        REQUIRE(param.extra_info_size_ == 38);
        REQUIRE(param.data_type_ == vsag::DataTypes::DATA_TYPE_FLOAT);
    }
}

TEST_CASE("IndexCommonParam L1 Test", "[ut][IndexCommonParam][l1]") {
    auto resource =
        std::make_shared<vsag::ResourceOwnerWrapper>(new vsag::Resource(), true);

    SECTION("float32 l1 succeeds") {
        auto params = vsag::JsonType::Parse(R"({
            "metric_type": "l1",
            "dtype": "float32",
            "dim": 5
        })");
        auto common = vsag::IndexCommonParam::CheckAndCreate(params, resource);
        REQUIRE(common.metric_ == vsag::MetricType::METRIC_TYPE_L1);
        REQUIRE(common.data_type_ == vsag::DataTypes::DATA_TYPE_FLOAT);
        REQUIRE(common.dim_ == 5);
    }

    SECTION("l1 rejects non-float32 dtype") {
        const char* invalid_params[] = {
            R"({"metric_type":"l1","dtype":"int8","dim":8})",
            R"({"metric_type":"l1","dtype":"sparse","dim":8})",
        };
        for (const auto* param_str : invalid_params) {
            auto params = vsag::JsonType::Parse(param_str);
            REQUIRE_THROWS(vsag::IndexCommonParam::CheckAndCreate(params, resource));
        }
    }

    SECTION("aliases are rejected") {
        const char* invalid_params[] = {
            R"({"metric_type":"manhattan","dtype":"float32","dim":5})",
            R"({"metric_type":"L1","dtype":"float32","dim":5})",
        };
        for (const auto* param_str : invalid_params) {
            auto params = vsag::JsonType::Parse(param_str);
            REQUIRE_THROWS(vsag::IndexCommonParam::CheckAndCreate(params, resource));
        }
    }
}
