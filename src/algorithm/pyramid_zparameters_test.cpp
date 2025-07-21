
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
    std::string odescent_io_type = "memory_io";
    int odescent_max_degree = 16;
    bool odescent_support_remove = false;
    int odescent_remove_flag_bit = 8;
    std::string base_codes_io_type = "memory_io";
    std::string base_codes_quantization_type = "fp32";
    std::vector<int> no_build_levels = {0, 1, 4};
};

std::string
generate_pyramid(const PyramidDefaultParam& param) {
    static constexpr auto param_str = R"(
        {{
            "odescent": {{
                "io_params": {{
                    "type": "{}"
                }},
                "max_degree": {},
                "alpha": 1.5,
                "graph_iter_turn": 10,
                "neighbor_sample_rate": 0.5,
                "support_remove": {},
                "remove_flag_bit": {}
            }},
            "base_codes": {{
                "io_params": {{
                    "type": "{}"
                }},
                "quantization_params": {{
                    "type": "{}"
                }}
            }},
            "no_build_levels": [{}],
            "ef_construction": 700
        }}
    )";
    return fmt::format(param_str,
                       param.odescent_io_type,
                       param.odescent_max_degree,
                       param.odescent_support_remove,
                       param.odescent_remove_flag_bit,
                       param.base_codes_io_type,
                       param.base_codes_quantization_type,
                       fmt::join(param.no_build_levels, ","));
}

TEST_CASE("Pyramid Parameters Test", "[ut][PyramidParameters]") {
    PyramidDefaultParam index_param;
    auto param_str = generate_pyramid(index_param);
    vsag::JsonType param_json = vsag::JsonType::parse(param_str);
    auto param = std::make_shared<vsag::PyramidParameters>();
    param->FromJson(param_json);
    vsag::ParameterTest::TestToJson(param);

    SECTION("invalid build_levels") {
        auto invalid_param_str1 = R"(
        {
            "odescent": {
                "io_params": {
                    "type": "memory_io"
                },
                "max_degree": 16
            },
            "no_build_levels": 2
        }
        )";
        REQUIRE_THROWS(param->FromJson(invalid_param_str1));
        auto invalid_param_str2 = R"(
        {
            "odescent": {
                "io_params": {
                    "type": "memory_io"
                },
                "max_degree": 16
            },
            "no_build_levels": [1,2, "hehehe"]
        }
        )";
        REQUIRE_THROWS(param->FromJson(invalid_param_str2));
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
    TEST_COMPATIBILITY_CASE(
        "different graph io type", odescent_io_type, "memory_io", "block_memory_io", true);
    TEST_COMPATIBILITY_CASE("different graph max_degree", odescent_max_degree, 18, 24, false);
    TEST_COMPATIBILITY_CASE(
        "different graph support remove", odescent_support_remove, true, false, false);
    TEST_COMPATIBILITY_CASE(
        "different graph remove flag bit", odescent_remove_flag_bit, 8, 4, false);
    TEST_COMPATIBILITY_CASE(
        "different base codes io type", base_codes_io_type, "memory_io", "block_memory_io", true);
    TEST_COMPATIBILITY_CASE("different base codes quantization type",
                            base_codes_quantization_type,
                            "fp32",
                            "fp16",
                            false);
    std::vector<int> build_levels1 = {0, 1, 2};
    std::vector<int> build_levels2 = {0, 1, 4};
    std::vector<int> build_levels3 = {1, 2, 0};
    TEST_COMPATIBILITY_CASE(
        "different not build levels", no_build_levels, build_levels1, build_levels2, false);
    TEST_COMPATIBILITY_CASE(
        "same no build levels", no_build_levels, build_levels1, build_levels3, true);
}
