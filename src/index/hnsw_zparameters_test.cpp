
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

#include "hnsw_zparameters.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("create hnsw with correct parameter", "[ut][hnsw]") {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    auto build_parameter_json = R"(
        {
            "max_degree": 16,
            "ef_construction": 100
        }
        )";

    auto parsed_params = vsag::JsonType::Parse(build_parameter_json);
    vsag::HnswParameters::FromJson(parsed_params, common_param);
}

TEST_CASE("create float32 hnsw with l1 parameter", "[ut][hnsw][l1]") {
    auto parsed_params = vsag::JsonType::Parse(R"(
        {
            "max_degree": 8,
            "ef_construction": 100
        }
    )");
    vsag::IndexCommonParam common_param;
    common_param.dim_ = 5;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L1;

    auto params = vsag::HnswParameters::FromJson(parsed_params, common_param);
    REQUIRE(params.type == vsag::DataTypes::DATA_TYPE_FLOAT);
    REQUIRE_FALSE(params.normalize);
    REQUIRE(params.space->get_data_size() == 5 * sizeof(float));

    const float lhs[5]{-1.0F, 2.5F, 4.0F, 0.0F, -2.0F};
    const float rhs[5]{2.0F, -0.5F, 1.0F, 0.25F, 3.0F};
    REQUIRE(params.space->get_dist_func()(lhs, rhs, params.space->get_dist_func_param()) == 14.25F);

    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_INT8;
    REQUIRE_THROWS(vsag::HnswParameters::FromJson(parsed_params, common_param));
}

TEST_CASE("create hnsw with wrong parameter", "[ut][hnsw]") {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    constexpr static const char* build_parameter_json = R"(
        {{
            "max_degree": {},
            "ef_construction": {}
        }}
        )";
    SECTION("small max_degree") {
        auto param_str = fmt::format(build_parameter_json, 3, 100);
        auto parsed_params = vsag::JsonType::Parse(param_str);
        REQUIRE_THROWS(vsag::HnswParameters::FromJson(parsed_params, common_param));
    }
    SECTION("big max_degree") {
        auto wrong_param_str = fmt::format(build_parameter_json, common_param.dim_ + 1, 200);
        auto wrong_parsed_params = vsag::JsonType::Parse(wrong_param_str);
        REQUIRE_THROWS(vsag::HnswParameters::FromJson(wrong_parsed_params, common_param));
        auto correct_param_str = fmt::format(build_parameter_json, common_param.dim_, 200);
        auto correct_parsed_params = vsag::JsonType::Parse(correct_param_str);
        vsag::HnswParameters::FromJson(correct_parsed_params, common_param);
    }
    SECTION("small ef_construction") {
        auto param_str = fmt::format(build_parameter_json, 16, 15);
        auto parsed_params = vsag::JsonType::Parse(param_str);
        REQUIRE_THROWS(vsag::HnswParameters::FromJson(parsed_params, common_param));
    }
    SECTION("big max_degree") {
        auto wrong_param_str = fmt::format(build_parameter_json, 16, 1601);
        auto wrong_parsed_params = vsag::JsonType::Parse(wrong_param_str);
        REQUIRE_THROWS(vsag::HnswParameters::FromJson(wrong_parsed_params, common_param));
        auto correct_param_str = fmt::format(build_parameter_json, 16, 1600);
        auto correct_parsed_params = vsag::JsonType::Parse(correct_param_str);
        vsag::HnswParameters::FromJson(correct_parsed_params, common_param);
    }
}

TEST_CASE("parse hnsw search skip strategy", "[ut][hnsw]") {
    SECTION("default deterministic accumulative strategy") {
        auto params = vsag::HnswSearchParameters::FromJson(R"(
        {
            "hnsw": {
                "ef_search": 100,
                "skip_ratio": 0.3
            }
        })");
        REQUIRE(params.skip_strategy_type ==
                vsag::FilterSearchSkipStrategyType::DETERMINISTIC_ACCUMULATIVE);
    }

    SECTION("explicit random strategy") {
        auto params = vsag::HnswSearchParameters::FromJson(R"(
        {
            "hnsw": {
                "ef_search": 100,
                "skip_ratio": 0.3,
                "skip_strategy": "random"
            }
        })");
        REQUIRE(params.skip_strategy_type == vsag::FilterSearchSkipStrategyType::RANDOM);
    }

    SECTION("invalid strategy") {
        REQUIRE_THROWS(vsag::HnswSearchParameters::FromJson(R"(
        {
            "hnsw": {
                "ef_search": 100,
                "skip_strategy": "invalid"
            }
        })"));
    }

    SECTION("non-string strategy") {
        REQUIRE_THROWS(vsag::HnswSearchParameters::FromJson(R"(
        {
            "hnsw": {
                "ef_search": 100,
                "skip_strategy": 1
            }
        })"));
    }
}
