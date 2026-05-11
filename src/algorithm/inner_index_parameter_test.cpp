
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

#include "inner_index_parameter.h"

#include <numeric>

#include "unittest.h"

TEST_CASE("Parameters Train Sample Count Test", "[ut][InnerIndexParameter][train_sample_count]") {
    constexpr const char* param_str = R"({})";

    // Test valid values
    auto json_obj = vsag::JsonType::Parse(param_str);
    json_obj["train_sample_count"].SetInt(32767);
    auto modified_param_str = json_obj.Dump();

    vsag::JsonType param_json = vsag::JsonType::Parse(modified_param_str);
    auto param = std::make_shared<vsag::InnerIndexParameter>();
    param->FromJson(param_json);
    REQUIRE(param->train_sample_count == 32767);

    json_obj["train_sample_count"].SetInt(512);
    modified_param_str = json_obj.Dump();

    param_json = vsag::JsonType::Parse(modified_param_str);
    param = std::make_shared<vsag::InnerIndexParameter>();
    param->FromJson(param_json);
    REQUIRE(param->train_sample_count == 512);

    param_json = vsag::JsonType::Parse(param_str);
    param = std::make_shared<vsag::InnerIndexParameter>();
    param->FromJson(param_json);
    REQUIRE(param->train_sample_count == 65536L);

    // Test invalid value less than minimum 512
    json_obj = vsag::JsonType::Parse(param_str);
    json_obj["train_sample_count"].SetInt(100);  // Invalid value, less than minimum 512
    modified_param_str = json_obj.Dump();

    param_json = vsag::JsonType::Parse(modified_param_str);
    param = std::make_shared<vsag::InnerIndexParameter>();

    REQUIRE_THROWS_AS(param->FromJson(param_json), vsag::VsagException);

    // Test invalid value exceeding maximum 65536
    json_obj = vsag::JsonType::Parse(param_str);
    json_obj["train_sample_count"].SetInt(1000000);
    modified_param_str = json_obj.Dump();

    param_json = vsag::JsonType::Parse(modified_param_str);
    param = std::make_shared<vsag::InnerIndexParameter>();

    REQUIRE_THROWS_AS(param->FromJson(param_json), vsag::VsagException);
}

TEST_CASE("Sampling Logic Test", "[ut][InnerIndexParameter][sampling]") {
    SECTION("Train sample count affects actual sampling") {
        // This test conceptually verifies that different train_sample_count values
        // would lead to different sampling behavior in the implementation
        // Note: Actual sampling behavior is tested in ivf.cpp unit tests

        constexpr const char* param_str = R"({})";

        // Test that the parameter correctly stores the configured sample count
        auto json_obj = vsag::JsonType::Parse(param_str);
        json_obj["train_sample_count"].SetInt(20000);
        auto modified_param_str = json_obj.Dump();

        vsag::JsonType param_json = vsag::JsonType::Parse(modified_param_str);
        auto param = std::make_shared<vsag::InnerIndexParameter>();
        param->FromJson(param_json);
        REQUIRE(param->train_sample_count == 20000);

        // Verify that this value is different from the default
        REQUIRE(param->train_sample_count != 65536L);
    }
}

TEST_CASE("Label remap type parameter test", "[ut][InnerIndexParameter][label_remap_type]") {
    auto param = std::make_shared<vsag::InnerIndexParameter>();
    auto json_obj = vsag::JsonType::Parse(R"({})");

    SECTION("default is pg") {
        param->FromJson(json_obj);
        REQUIRE(param->label_remap_type == vsag::LabelRemapType::PG);
        REQUIRE(param->ToJson()["label_remap_type"].GetString() == "pg");
    }

    SECTION("parse robin") {
        json_obj["label_remap_type"].SetString("robin");
        param->FromJson(json_obj);
        REQUIRE(param->label_remap_type == vsag::LabelRemapType::ROBIN);
        REQUIRE(param->ToJson()["label_remap_type"].GetString() == "robin");
    }

    SECTION("reject invalid value") {
        json_obj["label_remap_type"].SetString("invalid");
        REQUIRE_THROWS_AS(param->FromJson(json_obj), vsag::VsagException);
    }
}
