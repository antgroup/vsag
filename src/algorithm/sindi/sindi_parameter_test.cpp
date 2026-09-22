
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

#include "sindi_parameter.h"

#include "inner_string_params.h"
#include "parameter_test.h"
#include "unittest.h"
using namespace vsag;

#define TEST_COMPATIBILITY_CASE(section_name, param_member, val1, val2, expect_compatible) \
    SECTION(section_name) {                                                                \
        SINDIDefaultParam param1;                                                          \
        SINDIDefaultParam param2;                                                          \
        param1.param_member = val1;                                                        \
        param2.param_member = val2;                                                        \
        auto param_str1 = generate_sindi_param(param1);                                    \
        auto param_str2 = generate_sindi_param(param2);                                    \
        auto sindi_param1 = std::make_shared<vsag::SINDIParameter>();                      \
        auto sindi_param2 = std::make_shared<vsag::SINDIParameter>();                      \
        sindi_param1->FromString(param_str1);                                              \
        sindi_param2->FromString(param_str2);                                              \
        if (expect_compatible) {                                                           \
            REQUIRE(sindi_param1->CheckCompatibility(sindi_param2));                       \
        } else {                                                                           \
            REQUIRE_FALSE(sindi_param1->CheckCompatibility(sindi_param2));                 \
        }                                                                                  \
    }

struct SINDIDefaultParam {
    bool use_reorder = true;
    std::string sparse_value_quant_type = QUANTIZATION_TYPE_VALUE_FP32;
    float doc_prune_ratio = 0.1F;
    int window_size = 55555;
    int term_id_limit = 10000;
    int avg_doc_term_length = 100;
    bool remap_term_ids = false;
    bool immutable = false;
};

std::string
generate_sindi_param(const SINDIDefaultParam& param) {
    vsag::JsonType json;
    json[USE_REORDER_KEY].SetBool(param.use_reorder);
    if (param.sparse_value_quant_type == QUANTIZATION_TYPE_VALUE_FP16) {
        json[USE_QUANTIZATION].SetString(QUANTIZATION_TYPE_VALUE_FP16);
    } else {
        json[USE_QUANTIZATION].SetBool(param.sparse_value_quant_type ==
                                       QUANTIZATION_TYPE_VALUE_SQ8);
    }
    json[SPARSE_DOC_PRUNE_RATIO].SetFloat(param.doc_prune_ratio);
    json[SPARSE_WINDOW_SIZE].SetInt(param.window_size);
    json[SPARSE_TERM_ID_LIMIT].SetInt(param.term_id_limit);
    json[SPARSE_AVG_DOC_TERM_LENGTH].SetInt(param.avg_doc_term_length);
    json[SPARSE_REMAP_TERM_IDS].SetBool(param.remap_term_ids);
    json[SPARSE_IMMUTABLE].SetBool(param.immutable);
    return json.Dump();
}

TEST_CASE("SINDI Index Parameters Test", "[ut][SINDIParameter]") {
    SINDIDefaultParam default_param;
    std::string param_str = generate_sindi_param(default_param);
    vsag::JsonType param_json = vsag::JsonType::Parse(param_str);
    auto param = std::make_shared<vsag::SINDIParameter>();
    param->FromJson(param_json);
    REQUIRE(param->use_reorder == default_param.use_reorder);
    REQUIRE(SparseValueQuantizationTypeToString(param->sparse_value_quant_type) ==
            default_param.sparse_value_quant_type);
    REQUIRE(std::abs(param->doc_prune_ratio - default_param.doc_prune_ratio) < 1e-3);
    REQUIRE(param->window_size == default_param.window_size);
    REQUIRE(param->term_id_limit == default_param.term_id_limit);
    REQUIRE(param->avg_doc_term_length == default_param.avg_doc_term_length);
    REQUIRE(param->remap_term_ids == default_param.remap_term_ids);
    REQUIRE(param->immutable == default_param.immutable);

    vsag::ParameterTest::TestToJson(param);
    REQUIRE_FALSE(param->ToJson().Contains(SPARSE_IMMUTABLE));

    auto search_param_str = R"({
        "sindi": {
            "query_prune_ratio": 0.2,
            "n_candidate": 20,
            "term_prune_ratio": 0.1
        }
    })";
    auto search_param = std::make_shared<vsag::SINDISearchParameter>();
    vsag::JsonType search_param_json = vsag::JsonType::Parse(search_param_str);
    search_param->FromJson(search_param_json);
    vsag::ParameterTest::TestToJson(search_param);
    REQUIRE_FALSE(search_param->ToJson()[INDEX_SINDI].Contains("use_term_lists_heap_insert"));

    auto legacy_search_param_str = R"({
        "sindi": {
            "query_prune_ratio": 0.2,
            "n_candidate": 20,
            "term_prune_ratio": 0.1,
            "use_term_lists_heap_insert": false
        }
    })";
    auto legacy_search_param = std::make_shared<vsag::SINDISearchParameter>();
    legacy_search_param->FromJson(vsag::JsonType::Parse(legacy_search_param_str));
    REQUIRE_FALSE(
        legacy_search_param->ToJson()[INDEX_SINDI].Contains("use_term_lists_heap_insert"));
}

TEST_CASE("SINDI Index Parameters Compatibility Test", "[ut][SINDIParameter]") {
    TEST_COMPATIBILITY_CASE("use_reorder compatibility", use_reorder, true, false, false);
    TEST_COMPATIBILITY_CASE("value quantization compatibility",
                            sparse_value_quant_type,
                            QUANTIZATION_TYPE_VALUE_FP32,
                            QUANTIZATION_TYPE_VALUE_FP16,
                            false);
    TEST_COMPATIBILITY_CASE("doc_prune_ratio compatibility", doc_prune_ratio, 0.2F, 0.3F, false);
    TEST_COMPATIBILITY_CASE("window_size compatibility", window_size, 33333, 55555, false);
    TEST_COMPATIBILITY_CASE("term_id_limit compatibility", term_id_limit, 10000, 10001, false);
    TEST_COMPATIBILITY_CASE(
        "avg_doc_term_length compatibility", avg_doc_term_length, 100, 200, false);
    TEST_COMPATIBILITY_CASE("remap_term_ids compatibility", remap_term_ids, false, true, false);
    TEST_COMPATIBILITY_CASE("immutable compatibility", immutable, false, true, false);
}

TEST_CASE("SINDI immutable Parameter", "[ut][SINDIParameter]") {
    SECTION("default is false when not specified") {
        auto param_str = R"({"term_id_limit": 1000, "window_size": 50000})";
        auto param = std::make_shared<vsag::SINDIParameter>();
        param->FromJson(vsag::JsonType::Parse(param_str));
        REQUIRE(param->immutable == false);
    }

    SECTION("serialized only when enabled") {
        SINDIDefaultParam dp;
        dp.immutable = true;
        auto param = std::make_shared<vsag::SINDIParameter>();
        param->FromString(generate_sindi_param(dp));
        REQUIRE(param->immutable == true);

        auto json = param->ToJson();
        REQUIRE(json.Contains(SPARSE_IMMUTABLE));
        REQUIRE(json[SPARSE_IMMUTABLE].GetBool());
    }
}

TEST_CASE("SINDI use_quantization Parameter", "[ut][SINDIParameter]") {
    SECTION("legacy false means fp32") {
        SINDIDefaultParam dp;
        dp.sparse_value_quant_type = QUANTIZATION_TYPE_VALUE_FP32;
        auto param = std::make_shared<vsag::SINDIParameter>();
        param->FromString(generate_sindi_param(dp));
        REQUIRE(param->sparse_value_quant_type == SparseValueQuantizationType::FP32);
        REQUIRE_FALSE(param->ToJson()["use_quantization"].GetBool());
    }

    SECTION("legacy true means sq8") {
        SINDIDefaultParam dp;
        dp.sparse_value_quant_type = QUANTIZATION_TYPE_VALUE_SQ8;
        auto param = std::make_shared<vsag::SINDIParameter>();
        param->FromString(generate_sindi_param(dp));
        REQUIRE(param->sparse_value_quant_type == SparseValueQuantizationType::SQ8);
        REQUIRE(param->ToJson()["use_quantization"].GetBool());
    }

    SECTION("string fp16 means fp16") {
        SINDIDefaultParam dp;
        dp.sparse_value_quant_type = QUANTIZATION_TYPE_VALUE_FP16;
        auto param = std::make_shared<vsag::SINDIParameter>();
        param->FromString(generate_sindi_param(dp));
        REQUIRE(param->sparse_value_quant_type == SparseValueQuantizationType::FP16);
        REQUIRE(param->ToJson()["use_quantization"].GetString() == QUANTIZATION_TYPE_VALUE_FP16);
    }

    SECTION("unexpected type throws clear argument error") {
        auto param = std::make_shared<vsag::SINDIParameter>();
        REQUIRE_THROWS_AS(param->FromString(R"({"term_id_limit":1000,"use_quantization":1})"),
                          vsag::VsagException);
    }
}

TEST_CASE("SINDI term_id_limit upper bound", "[ut][SINDIParameter]") {
    SINDIDefaultParam param;
    param.term_id_limit = 50'000'000;
    auto valid_param = std::make_shared<vsag::SINDIParameter>();
    valid_param->FromString(generate_sindi_param(param));
    REQUIRE(valid_param->term_id_limit == 50'000'000);

    param.term_id_limit = 50'000'001;
    auto invalid_param = std::make_shared<vsag::SINDIParameter>();
    REQUIRE_THROWS(invalid_param->FromString(generate_sindi_param(param)));
}

TEST_CASE("SINDI remap_term_ids Parameter", "[ut][SINDIParameter]") {
    SECTION("default is false when not specified") {
        auto param_str = R"({"term_id_limit": 1000, "window_size": 50000})";
        auto param = std::make_shared<vsag::SINDIParameter>();
        param->FromJson(vsag::JsonType::Parse(param_str));
        REQUIRE(param->remap_term_ids == false);
    }

    SECTION("parse true") {
        SINDIDefaultParam dp;
        dp.remap_term_ids = true;
        auto param_str = generate_sindi_param(dp);
        auto param = std::make_shared<vsag::SINDIParameter>();
        param->FromString(param_str);
        REQUIRE(param->remap_term_ids == true);
    }

    SECTION("parse false") {
        SINDIDefaultParam dp;
        dp.remap_term_ids = false;
        auto param_str = generate_sindi_param(dp);
        auto param = std::make_shared<vsag::SINDIParameter>();
        param->FromString(param_str);
        REQUIRE(param->remap_term_ids == false);
    }

    SECTION("round-trip: FromJson -> ToJson -> FromJson") {
        SINDIDefaultParam dp;
        dp.remap_term_ids = true;
        auto param_str = generate_sindi_param(dp);
        auto param1 = std::make_shared<vsag::SINDIParameter>();
        param1->FromString(param_str);

        auto json2 = param1->ToJson();
        auto param2 = std::make_shared<vsag::SINDIParameter>();
        param2->FromJson(json2);
        REQUIRE(param2->remap_term_ids == true);
    }
}
