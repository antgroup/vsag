
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

#include <fmt/format.h>

#include <catch2/matchers/catch_matchers_string.hpp>
#include <string>

#include "factory/resource_owner_wrapper.h"
#include "index_common_param.h"
#include "unittest.h"
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

TEST_CASE("IndexCommonParam Binary Hamming Test", "[ut][IndexCommonParam]") {
    std::shared_ptr<vsag::Resource> resource =
        std::make_shared<vsag::ResourceOwnerWrapper>(new vsag::Resource(), true);
    auto create = [&](const std::string& dtype, const std::string& metric, int64_t dim) {
        auto parsed_params = vsag::JsonType::Parse(fmt::format(
            R"({{"dtype": "{}", "metric_type": "{}", "dim": {}}})", dtype, metric, dim));
        return vsag::IndexCommonParam::CheckAndCreate(parsed_params, resource);
    };

    SECTION("binary with hamming") {
        // dim counts bits, including dimensions that are not multiples of eight.
        for (int64_t dim : {1, 7, 8, 9, 63, 64, 65, 127, 128, 4096}) {
            auto param = create("binary", "hamming", dim);
            REQUIRE(param.data_type_ == vsag::DataTypes::DATA_TYPE_BINARY);
            REQUIRE(param.metric_ == vsag::MetricType::METRIC_TYPE_HAMMING);
            REQUIRE(param.repr_ == vsag::RecordRepr::DENSE);
            REQUIRE(param.dim_ == dim);
        }
        REQUIRE(create("binary", "hamming", vsag::MAX_BINARY_DIM).dim_ == vsag::MAX_BINARY_DIM);
    }

    SECTION("binary requires hamming") {
        auto metric = GENERATE("l2", "ip", "cosine");
        REQUIRE_THROWS_WITH(
            create("binary", metric, 128),
            Catch::Matchers::ContainsSubstring(
                fmt::format("parameters[dtype]=binary requires parameters[metric_type]=hamming, "
                            "got {}",
                            metric)));
    }

    SECTION("hamming requires binary") {
        auto dtype = GENERATE("float32", "float16", "bfloat16", "int8", "sparse");
        REQUIRE_THROWS_WITH(
            create(dtype, "hamming", 128),
            Catch::Matchers::ContainsSubstring(
                fmt::format("parameters[metric_type]=hamming requires parameters[dtype]=binary, "
                            "got {}",
                            dtype)));
    }

    SECTION("binary rejects other representations and oversized dims") {
        auto parsed_params = vsag::JsonType::Parse(
            R"({"dtype": "binary", "metric_type": "hamming", "dim": 64, "repr": "multi_vector"})");
        REQUIRE_THROWS_WITH(vsag::IndexCommonParam::CheckAndCreate(parsed_params, resource),
                            Catch::Matchers::ContainsSubstring("requires parameters[repr]=dense"));
        REQUIRE_THROWS_WITH(create("binary", "hamming", vsag::MAX_BINARY_DIM + 1),
                            Catch::Matchers::ContainsSubstring("counts bits"));
        REQUIRE_THROWS(create("binary", "hamming", 0));
        auto no_dim = vsag::JsonType::Parse(R"({"dtype": "binary", "metric_type": "hamming"})");
        REQUIRE_THROWS(vsag::IndexCommonParam::CheckAndCreate(no_dim, resource));
    }

    SECTION("unknown values list the new choices") {
        REQUIRE_THROWS_WITH(create("bits", "hamming", 8),
                            Catch::Matchers::ContainsSubstring("binary"));
        REQUIRE_THROWS_WITH(create("binary", "jaccard", 8),
                            Catch::Matchers::ContainsSubstring("hamming"));
    }
}
