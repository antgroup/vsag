
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

#include "algorithm/pyramid_zparameters.h"

#include <catch2/catch_test_macros.hpp>

#include "parameter_test.h"

struct PyramidDefaultParam {
    int max_degree = 16;
    float alpha = 1.2f;
    int ef_construction = 200;
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
    std::string precise_io_type = "memory_io";
    std::string precise_file_path = "precise_path";
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
            "use_reorder": {}
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
                       param.use_reorder);
}

TEST_CASE("Pyramid Parameters Test", "[ut][PyramidParameters]") {
    PyramidDefaultParam index_param;
    auto param_str = generate_pyramid(index_param);
    vsag::JsonType param_json = vsag::JsonType::Parse(param_str);
    auto param = std::make_shared<vsag::PyramidParameters>();
    param->FromJson(param_json);
    vsag::ParameterTest::TestToJson(param);
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
}
