
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

#include <catch2/catch_test_macros.hpp>

#include "parameter_test.h"

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
    float doc_prune_ratio = 0.1F;
    int window_size = 66666;
};

std::string
generate_sindi_param(const SINDIDefaultParam& param) {
    vsag::JsonType json;
    json["use_reorder"].SetBool(param.use_reorder);
    json["doc_prune_ratio"].SetFloat(param.doc_prune_ratio);
    json["window_size"].SetInt(param.window_size);
    return json.Dump();
}

TEST_CASE("SINDI Index Parameters Test", "[ut][SINDIParameter]") {
    SINDIDefaultParam default_param;
    std::string param_str = generate_sindi_param(default_param);
    vsag::JsonType param_json = vsag::JsonType::Parse(param_str);
    auto param = std::make_shared<vsag::SINDIParameter>();
    param->FromJson(param_json);
    REQUIRE(param->use_reorder == true);
    REQUIRE(std::abs(param->doc_prune_ratio - 0.1F) < 1e-3);
    REQUIRE(param->window_size == 66666);

    vsag::ParameterTest::TestToJson(param);

    auto search_param_str = R"({
        "sindi": {
            "query_prune_ratio": 0.2,
            "n_candidate": 20,
            "term_prune_ratio": 0.1
        }
    })";
    auto search_param = std::make_shared<vsag::SINDIParameter>();
    vsag::JsonType search_param_json = vsag::JsonType::Parse(search_param_str);
    search_param->FromJson(search_param_json);
    vsag::ParameterTest::TestToJson(search_param);
}

TEST_CASE("SINDI Index Parameters Compatibility Test", "[ut][SINDIParameter]") {
    TEST_COMPATIBILITY_CASE("use_reorder compatibility", use_reorder, true, false, false);
    TEST_COMPATIBILITY_CASE("doc_prune_ratio compatibility", doc_prune_ratio, 0.2F, 0.3F, false);
    TEST_COMPATIBILITY_CASE("window_size compatibility", window_size, 66666, 77777, false);
}
