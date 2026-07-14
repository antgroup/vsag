
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

#include "pyramid_zparameters.h"

#include <fmt/ranges.h>

#include <nlohmann/json.hpp>

#include "index_common_param.h"
#include "parameter_test.h"
#include "pyramid.h"
#include "unittest.h"
#include "vsag_exception.h"

struct PyramidDefaultParam {
    int max_degree = 32;
    float alpha = 1.3f;
    int ef_construction = 400;
    bool use_reorder = true;
    std::string base_quantization_type = "fp32";
    std::vector<int> no_build_levels = {0, 1, 2};
    std::string base_io_type = "memory_io";
    std::string graph_type = "odescent";
    std::string graph_storage_type = "compressed";
    int build_thread_count = 8;
    std::string precise_quantization_type = "fp32";
    int base_pq_dim = 0;
    std::string base_file_path = "base_path";
    std::string precise_io_type = "block_memory_io";
    std::string precise_file_path = "precise_path";
    uint32_t index_min_size = 1000;
    bool support_duplicate = false;
};

std::string
generate_pyramid(const PyramidDefaultParam& param) {
    static constexpr auto param_str = R"(
        {{
            "base_codes": {{
                "codes_type": "flatten",
                "io_params": {{
                    "file_path": "{}",
                    "type": "{}"
                }},
                "quantization_params": {{
                    "hold_molds": false,
                    "nbits": 8,
                    "pca_dim": 0,
                    "pq_dim": {},
                    "rabitq_bits_per_dim_query": 32,
                    "sq4_uniform_trunc_rate": 0.05,
                    "tq_chain": "",
                    "type": "{}"
                }}
            }},
            "build_thread_count": {},
            "ef_construction": {},
            "graph": {{
                "alpha": {},
                "build_block_size": 10000,
                "graph_iter_turn": 30,
                "graph_storage_type": "{}",
                "graph_type": "{}",
                "init_capacity": 100,
                "io_params": {{
                    "file_path": "./default_file_path",
                    "type": "block_memory_io"
                }},
                "max_degree": {},
                "min_in_degree": 1,
                "neighbor_sample_rate": 0.2,
                "remove_flag_bit": 8,
                "support_remove": false
            }},
            "no_build_levels": [{}],
            "precise_codes": {{
                "codes_type": "flatten",
                "io_params": {{
                    "file_path": "{}",
                    "type": "{}"
                }},
                "quantization_params": {{
                    "hold_molds": false,
                    "pca_dim": 0,
                    "pq_dim": 1,
                    "sq4_uniform_trunc_rate": 0.05,
                    "type": "{}"
                }}
            }},
            "type": "pyramid",
            "use_reorder": {},
            "index_min_size": {},
            "support_duplicate": {}
        }}
    )";
    return fmt::format(param_str,
                       param.base_file_path,
                       param.base_io_type,
                       param.base_pq_dim,
                       param.base_quantization_type,
                       param.build_thread_count,
                       param.ef_construction,
                       param.alpha,
                       param.graph_storage_type,
                       param.graph_type,
                       param.max_degree,
                       fmt::join(param.no_build_levels, ","),
                       param.precise_file_path,
                       param.precise_io_type,
                       param.precise_quantization_type,
                       param.use_reorder,
                       param.index_min_size,
                       param.support_duplicate);
}

TEST_CASE("Pyramid Parameters Test", "[ut][PyramidParameters]") {
    PyramidDefaultParam index_param;
    auto param_str = generate_pyramid(index_param);
    vsag::JsonType param_json = vsag::JsonType::Parse(param_str);
    auto param = std::make_shared<vsag::PyramidParameters>();
    param->FromJson(param_json);
    vsag::ParameterTest::TestToJson(param);
}

std::shared_ptr<vsag::PyramidParameters>
ParsePyramidWithHierarchies(const nlohmann::json& hierarchies) {
    PyramidDefaultParam index_param;
    auto param_json = vsag::JsonType::Parse(generate_pyramid(index_param));
    *param_json["hierarchies"].GetInnerJson() = hierarchies;
    auto param = std::make_shared<vsag::PyramidParameters>();
    param->FromJson(param_json);
    return param;
}

TEST_CASE("Pyramid Hierarchy Parameters Test", "[ut][PyramidParameters][hierarchy]") {
    SECTION("parse string and object hierarchy definitions") {
        auto param = ParsePyramidWithHierarchies(
            nlohmann::json::array({"site", {{"name", "taxonomy"}, {"no_build_levels", {2, 0}}}}));

        REQUIRE(param->has_hierarchies);
        REQUIRE(param->hierarchies.size() == 2);
        REQUIRE(param->hierarchies[0].name == "site");
        REQUIRE(param->hierarchies[0].no_build_levels == std::vector<int32_t>{0, 1, 2});
        REQUIRE(param->hierarchies[1].name == "taxonomy");
        REQUIRE(param->hierarchies[1].no_build_levels == std::vector<int32_t>{0, 2});

        auto output = param->ToJson();
        const auto* hierarchies_json = output["hierarchies"].GetInnerJson();
        REQUIRE(hierarchies_json->size() == 2);
        REQUIRE((*hierarchies_json)[0]["name"] == "site");
        REQUIRE((*hierarchies_json)[1]["name"] == "taxonomy");
        REQUIRE((*hierarchies_json)[1]["no_build_levels"] == nlohmann::json::array({0, 2}));
    }

    SECTION("per-hierarchy build params override top-level") {
        auto param = ParsePyramidWithHierarchies(nlohmann::json::array(
            {{{"name", "site"}, {"max_degree", 128}, {"alpha", 1.5}},
             {{"name", "cat"}, {"ef_construction", 800}, {"index_min_size", 50}}}));

        REQUIRE(param->hierarchies.size() == 2);
        // "site" overrides max_degree and alpha, inherits the rest from top-level
        REQUIRE(param->hierarchies[0].max_degree == 128);
        REQUIRE(param->hierarchies[0].alpha == 1.5F);
        REQUIRE(param->hierarchies[0].ef_construction == param->ef_construction);
        REQUIRE(param->hierarchies[0].index_min_size == param->index_min_size);
        REQUIRE(param->hierarchies[0].no_build_levels == param->no_build_levels);
        // "cat" overrides ef_construction and index_min_size, inherits the rest
        REQUIRE(param->hierarchies[1].ef_construction == 800);
        REQUIRE(param->hierarchies[1].index_min_size == 50);
        REQUIRE(param->hierarchies[1].max_degree == param->max_degree);
        REQUIRE(param->hierarchies[1].alpha == param->alpha);

        // ToJson roundtrip preserves per-hierarchy params
        auto output = param->ToJson();
        const auto* h_json = output["hierarchies"].GetInnerJson();
        REQUIRE((*h_json)[0]["max_degree"] == 128);
        REQUIRE((*h_json)[0]["alpha"] == 1.5F);
        REQUIRE((*h_json)[1]["ef_construction"] == 800);
        REQUIRE((*h_json)[1]["index_min_size"] == 50);
    }

    SECTION("string shorthand inherits all top-level build params") {
        auto param = ParsePyramidWithHierarchies(nlohmann::json::array({"onlyone"}));

        REQUIRE(param->hierarchies.size() == 1);
        REQUIRE(param->hierarchies[0].name == "onlyone");
        REQUIRE(param->hierarchies[0].no_build_levels == param->no_build_levels);
        REQUIRE(param->hierarchies[0].max_degree == param->max_degree);
        REQUIRE(param->hierarchies[0].ef_construction == param->ef_construction);
        REQUIRE(param->hierarchies[0].alpha == param->alpha);
        REQUIRE(param->hierarchies[0].index_min_size == param->index_min_size);
    }

    SECTION("invalid hierarchy definitions are rejected") {
        REQUIRE_THROWS(ParsePyramidWithHierarchies(nlohmann::json::array()));
        REQUIRE_THROWS(ParsePyramidWithHierarchies(nlohmann::json::array({""})));
        REQUIRE_THROWS(
            ParsePyramidWithHierarchies(nlohmann::json::array({"site", {{"name", "site"}}})));
        REQUIRE_THROWS(ParsePyramidWithHierarchies(nlohmann::json::array({{{"name", ""}}})));
        REQUIRE_THROWS(ParsePyramidWithHierarchies(
            nlohmann::json::array({{{"name", "site"}, {"no_build_levels", 1}}})));
        REQUIRE_THROWS(ParsePyramidWithHierarchies(
            nlohmann::json::array({{{"name", "site"}, {"no_build_levels", {-1}}}})));
        REQUIRE_THROWS(ParsePyramidWithHierarchies(
            nlohmann::json::array({{{"name", "site"}, {"max_degree", 0}}})));
        REQUIRE_THROWS(ParsePyramidWithHierarchies(
            nlohmann::json::array({{{"name", "site"}, {"ef_construction", -1}}})));
        REQUIRE_THROWS(
            ParsePyramidWithHierarchies(nlohmann::json::array({{{"name", "site"}, {"alpha", 0}}})));
        REQUIRE_THROWS(ParsePyramidWithHierarchies(
            nlohmann::json::array({{{"name", "site"}, {"index_min_size", -1}}})));
        REQUIRE_THROWS(ParsePyramidWithHierarchies(nlohmann::json::array({{{"foo", "site"}}})));
        REQUIRE_THROWS_AS(ParsePyramidWithHierarchies(nlohmann::json::array({1})),
                          vsag::VsagException);
        REQUIRE_THROWS_AS(ParsePyramidWithHierarchies(nlohmann::json::array({true})),
                          vsag::VsagException);
        REQUIRE_THROWS_AS(ParsePyramidWithHierarchies(nlohmann::json::array({nullptr})),
                          vsag::VsagException);
    }
}

#define TEST_COMPATIBILITY_CASE(section_name, param_member, val1, val2, expect_compatible) \
    SECTION(section_name) {                                                                \
        PyramidDefaultParam param1;                                                        \
        PyramidDefaultParam param2;                                                        \
        param1.param_member = val1;                                                        \
        param2.param_member = val2;                                                        \
        auto param_str1 = generate_pyramid(param1);                                        \
        auto param_str2 = generate_pyramid(param2);                                        \
        auto pyramid_param1 = std::make_shared<vsag::PyramidParameters>();                 \
        auto pyramid_param2 = std::make_shared<vsag::PyramidParameters>();                 \
        pyramid_param1->FromString(param_str1);                                            \
        pyramid_param2->FromString(param_str2);                                            \
        if (expect_compatible) {                                                           \
            REQUIRE(pyramid_param1->CheckCompatibility(pyramid_param2));                   \
        } else {                                                                           \
            REQUIRE_FALSE(pyramid_param1->CheckCompatibility(pyramid_param2));             \
        }                                                                                  \
    }

TEST_CASE("Pyramid Parameters CheckCompatibility", "[ut][PyramidParameter][CheckCompatibility]") {
    SECTION("wrong parameter type") {
        PyramidDefaultParam default_param;
        auto param_str = generate_pyramid(default_param);
        auto param = std::make_shared<vsag::PyramidParameters>();
        param->FromString(param_str);
        REQUIRE(param->CheckCompatibility(param));
        REQUIRE_FALSE(param->CheckCompatibility(std::make_shared<vsag::EmptyParameter>()));
    }
    TEST_COMPATIBILITY_CASE("different graph max_degree", max_degree, 18, 24, false);
    TEST_COMPATIBILITY_CASE("different graph alpha", alpha, 1.0f, 1.5f, true);
    TEST_COMPATIBILITY_CASE("different ef_construction", ef_construction, 150, 200, true)
    TEST_COMPATIBILITY_CASE(
        "different base codes quantization type", base_quantization_type, "fp32", "fp16", false);
    std::vector<int> build_levels1 = {0, 1, 2};
    std::vector<int> build_levels2 = {0, 1, 4};
    std::vector<int> build_levels3 = {1, 2, 0};
    TEST_COMPATIBILITY_CASE(
        "different not build levels", no_build_levels, build_levels1, build_levels2, false);
    TEST_COMPATIBILITY_CASE(
        "same no build levels", no_build_levels, build_levels1, build_levels3, true);
    TEST_COMPATIBILITY_CASE(
        "different base io type", base_io_type, "memory_io", "block_memory_io", true);

    TEST_COMPATIBILITY_CASE("different graph type", graph_type, "odescent", "nsw", true);
    TEST_COMPATIBILITY_CASE("different build thread count", build_thread_count, 4, 8, true);
    TEST_COMPATIBILITY_CASE(
        "different precise quantization type", precise_quantization_type, "fp32", "fp16", false);
    TEST_COMPATIBILITY_CASE("different index min size", index_min_size, 500, 1500, false);
    TEST_COMPATIBILITY_CASE("different support duplicate", support_duplicate, false, true, false);

    SECTION("same hierarchies in different order") {
        auto param1 = ParsePyramidWithHierarchies(
            nlohmann::json::array({{{"name", "site"}, {"no_build_levels", {0, 1}}},
                                   {{"name", "taxonomy"}, {"no_build_levels", {2}}}}));
        auto param2 = ParsePyramidWithHierarchies(
            nlohmann::json::array({{{"name", "taxonomy"}, {"no_build_levels", {2}}},
                                   {{"name", "site"}, {"no_build_levels", {1, 0}}}}));

        REQUIRE(param1->CheckCompatibility(param2));
    }

    SECTION("different hierarchy no_build_levels") {
        auto param1 = ParsePyramidWithHierarchies(
            nlohmann::json::array({{{"name", "site"}, {"no_build_levels", {0}}}}));
        auto param2 = ParsePyramidWithHierarchies(
            nlohmann::json::array({{{"name", "site"}, {"no_build_levels", {1}}}}));

        REQUIRE_FALSE(param1->CheckCompatibility(param2));
    }

    SECTION("different hierarchy names") {
        auto param1 = ParsePyramidWithHierarchies(nlohmann::json::array({"site"}));
        auto param2 = ParsePyramidWithHierarchies(nlohmann::json::array({"taxonomy"}));

        REQUIRE_FALSE(param1->CheckCompatibility(param2));
    }

    SECTION("legacy and explicit hierarchy modes are incompatible") {
        PyramidDefaultParam default_param;
        auto legacy_param = std::make_shared<vsag::PyramidParameters>();
        legacy_param->FromString(generate_pyramid(default_param));
        auto hierarchy_param = ParsePyramidWithHierarchies(nlohmann::json::array({"site"}));

        REQUIRE_FALSE(legacy_param->CheckCompatibility(hierarchy_param));
        REQUIRE_FALSE(hierarchy_param->CheckCompatibility(legacy_param));
    }
}

TEST_CASE("Pyramid Search Hierarchy Parameters Test",
          "[ut][PyramidParameters][hierarchy][search]") {
    SECTION("single hierarchy selector") {
        auto search_param = vsag::PyramidSearchParameters::FromJson(
            R"({"pyramid":{"ef_search":100,"hierarchies":["taxonomy"]}})");

        REQUIRE(search_param.HasHierarchySelector());
        REQUIRE(search_param.hierarchies == std::vector<std::string>{"taxonomy"});
        REQUIRE(search_param.hierarchy_op == vsag::PyramidSearchParameters::HierarchyOp::SINGLE);
    }

    SECTION("multi hierarchy intersection selector") {
        auto search_param = vsag::PyramidSearchParameters::FromJson(
            R"({"pyramid":{"ef_search":100,"hierarchies":["site","taxonomy"],)"
            R"("hierarchy_op":"intersection"}})");

        REQUIRE(search_param.HasHierarchySelector());
        REQUIRE(search_param.hierarchies == std::vector<std::string>{"site", "taxonomy"});
        REQUIRE(search_param.hierarchy_op != vsag::PyramidSearchParameters::HierarchyOp::SINGLE);
        REQUIRE(search_param.hierarchy_op ==
                vsag::PyramidSearchParameters::HierarchyOp::INTERSECTION);
    }

    SECTION("multi hierarchy union selector") {
        auto search_param = vsag::PyramidSearchParameters::FromJson(
            R"({"pyramid":{"ef_search":100,"hierarchies":["site","date"],)"
            R"("hierarchy_op":"union"}})");

        REQUIRE(search_param.hierarchies == std::vector<std::string>{"site", "date"});
        REQUIRE(search_param.hierarchy_op != vsag::PyramidSearchParameters::HierarchyOp::SINGLE);
        REQUIRE(search_param.hierarchy_op == vsag::PyramidSearchParameters::HierarchyOp::UNION);
    }

    SECTION("no hierarchies field (legacy mode)") {
        auto search_param =
            vsag::PyramidSearchParameters::FromJson(R"({"pyramid":{"ef_search":100}})");

        REQUIRE_FALSE(search_param.HasHierarchySelector());
        REQUIRE(search_param.hierarchies.empty());
        REQUIRE(search_param.hierarchy_op == vsag::PyramidSearchParameters::HierarchyOp::SINGLE);
    }

    SECTION("invalid hierarchy search parameters are rejected") {
        // empty hierarchy name
        REQUIRE_THROWS(vsag::PyramidSearchParameters::FromJson(
            R"({"pyramid":{"ef_search":100,"hierarchies":[""]}})"));
        // empty array
        REQUIRE_THROWS(vsag::PyramidSearchParameters::FromJson(
            R"({"pyramid":{"ef_search":100,"hierarchies":[]}})"));
        // multiple hierarchies without hierarchy_op
        REQUIRE_THROWS(vsag::PyramidSearchParameters::FromJson(
            R"({"pyramid":{"ef_search":100,"hierarchies":["site","taxonomy"]}})"));
        // duplicate names
        REQUIRE_THROWS(vsag::PyramidSearchParameters::FromJson(
            R"({"pyramid":{"ef_search":100,"hierarchies":["site","site"],)"
            R"("hierarchy_op":"union"}})"));
        // invalid hierarchy_op value
        REQUIRE_THROWS(vsag::PyramidSearchParameters::FromJson(
            R"({"pyramid":{"ef_search":100,"hierarchies":["site","taxonomy"],)"
            R"("hierarchy_op":"minus"}})"));
        // hierarchy_op with single hierarchy
        REQUIRE_THROWS(vsag::PyramidSearchParameters::FromJson(
            R"({"pyramid":{"ef_search":100,"hierarchies":["site"],)"
            R"("hierarchy_op":"union"}})"));
    }
}

TEST_CASE("Pyramid maps support_duplicate to graph parameter", "[ut][PyramidParameters]") {
    auto param = vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "base_io_type": "memory_io",
        "precise_quantization_type": "fp32",
        "precise_io_type": "block_memory_io",
        "graph_storage_type": "compressed",
        "graph_type": "odescent",
        "max_degree": 32,
        "alpha": 1.2,
        "build_thread_count": 1,
        "ef_construction": 100,
        "index_min_size": 0,
        "support_duplicate": true,
        "hierarchies": [
            "site",
            {"name": "taxonomy", "no_build_levels": [0, 2]}
        ],
        "use_reorder": true
    })");

    vsag::IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    auto pyramid_param = vsag::Pyramid::CheckAndMappingExternalParam(param, common_param);
    auto typed_param = std::dynamic_pointer_cast<vsag::PyramidParameters>(pyramid_param);

    REQUIRE(typed_param != nullptr);
    REQUIRE(typed_param->support_duplicate);
    REQUIRE(typed_param->graph_param->support_duplicate_);
    REQUIRE(typed_param->has_hierarchies);
    REQUIRE(typed_param->hierarchies.size() == 2);
    REQUIRE(typed_param->hierarchies[0].name == "site");
    REQUIRE(typed_param->hierarchies[1].name == "taxonomy");
    REQUIRE(typed_param->hierarchies[1].no_build_levels == std::vector<int32_t>{0, 2});
}

TEST_CASE("Pyramid maps fast RaBitQ to base and precise quantizers", "[ut][PyramidParameters]") {
    auto param = vsag::JsonType::Parse(R"({
        "base_quantization_type": "rabitq",
        "precise_quantization_type": "rabitq",
        "rabitq_bits_per_dim_base": 4,
        "fast_encode_rabitq": false,
        "fast_encode_rabitq_rounds": 11,
        "hierarchies": ["site"],
        "use_reorder": true
    })");

    vsag::IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    auto mapped = vsag::Pyramid::CheckAndMappingExternalParam(param, common_param);
    auto typed_param = std::dynamic_pointer_cast<vsag::PyramidParameters>(mapped);

    REQUIRE(typed_param != nullptr);
    REQUIRE(typed_param->base_codes_param != nullptr);
    REQUIRE(typed_param->precise_codes_param != nullptr);
    const auto base_json = typed_param->base_codes_param->ToJson();
    const auto precise_json = typed_param->precise_codes_param->ToJson();
    REQUIRE_FALSE(base_json["quantization_params"]["fast_encode_rabitq"].GetBool());
    REQUIRE(base_json["quantization_params"]["fast_encode_rabitq_rounds"].GetInt() == 11);
    REQUIRE_FALSE(precise_json["quantization_params"]["fast_encode_rabitq"].GetBool());
    REQUIRE(precise_json["quantization_params"]["fast_encode_rabitq_rounds"].GetInt() == 11);
}
