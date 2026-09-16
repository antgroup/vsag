
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

    // Explicit values may exceed the default maximum 65536
    json_obj = vsag::JsonType::Parse(param_str);
    json_obj["train_sample_count"].SetInt(1000000);
    modified_param_str = json_obj.Dump();

    param_json = vsag::JsonType::Parse(modified_param_str);
    param = std::make_shared<vsag::InnerIndexParameter>();

    REQUIRE_NOTHROW(param->FromJson(param_json));
    REQUIRE(param->train_sample_count == 1000000);
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

TEST_CASE("RaBitQ split configuration test", "[ut][InnerIndexParameter][rabitq_split]") {
    auto external_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "rabitq",
        "precise_quantization_type": "rabitq",
        "use_reorder": true,
        "rabitq_bits_per_dim_base": 3,
        "rabitq_bits_per_dim_precise": 5
    })");

    const auto config = vsag::ParseRaBitQSplitConfig(external_json);
    REQUIRE(config.enabled);
    REQUIRE(config.filter_bits == 3);
    REQUIRE(config.supplement_bits == 5);
    REQUIRE(config.TotalBits() == 8);

    auto inner_json = vsag::JsonType::Parse(R"({
        "base_codes": {
            "quantization_params": {}
        }
    })");
    vsag::ApplyRaBitQSplitConfig(config, inner_json);
    REQUIRE(inner_json["reorder_source"].GetString() == "base");
    REQUIRE(inner_json["base_codes"]["codes_type"].GetString() == "rabitq_split");
    REQUIRE(inner_json["base_codes"]["quantization_params"]["rabitq_version"].GetString() ==
            "split");
    REQUIRE(
        inner_json["base_codes"]["quantization_params"]["rabitq_bits_per_dim_filter"].GetInt() ==
        3);
    REQUIRE(inner_json["base_codes"]["quantization_params"]["rabitq_bits_per_dim_base"].GetInt() ==
            8);
}

TEST_CASE("hold_molds is applied only to supported quantizers",
          "[ut][InnerIndexParameter][hold_molds]") {
    auto fp32 = vsag::JsonType::Parse(R"({"type":"fp32"})");
    fp32 = vsag::ApplyHoldMoldsToQuantizer(fp32, true);
    REQUIRE(fp32[vsag::HOLD_MOLDS].GetBool());

    auto int8 = vsag::JsonType::Parse(R"({"type":"int8"})");
    int8 = vsag::ApplyHoldMoldsToQuantizer(int8, true);
    REQUIRE(int8[vsag::HOLD_MOLDS].GetBool());

    auto transform = vsag::JsonType::Parse(R"({"type":"tq","tq_chain":"mrle, fp32"})");
    transform = vsag::ApplyHoldMoldsToQuantizer(transform, true);
    REQUIRE(transform[vsag::HOLD_MOLDS].GetBool());

    auto rabitq = vsag::JsonType::Parse(R"({"type":"rabitq"})");
    rabitq = vsag::ApplyHoldMoldsToQuantizer(rabitq, true);
    REQUIRE_FALSE(rabitq.Contains(vsag::HOLD_MOLDS));
}
