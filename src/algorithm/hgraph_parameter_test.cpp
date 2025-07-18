
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

#include "hgraph_parameter.h"

#include <catch2/catch_test_macros.hpp>

#include "parameter_test.h"

#define TEST_COMPATIBILITY_CASE(section_name, param_member, val1, val2, expect_compatible) \
    SECTION(section_name) {                                                                \
        HGraphDefaultParam param1;                                                         \
        HGraphDefaultParam param2;                                                         \
        param1.param_member = val1;                                                        \
        param2.param_member = val2;                                                        \
        auto param_str1 = generate_hgraph_param(param1);                                   \
        auto param_str2 = generate_hgraph_param(param2);                                   \
        auto hgraph_param1 = std::make_shared<vsag::HGraphParameter>();                    \
        auto hgraph_param2 = std::make_shared<vsag::HGraphParameter>();                    \
        hgraph_param1->FromString(param_str1);                                             \
        hgraph_param2->FromString(param_str2);                                             \
        if (expect_compatible) {                                                           \
            REQUIRE(hgraph_param1->CheckCompatibility(hgraph_param2));                     \
        } else {                                                                           \
            REQUIRE_FALSE(hgraph_param1->CheckCompatibility(hgraph_param2));               \
        }                                                                                  \
    }

struct HGraphDefaultParam {
    std::string base_codes_io_type = "block_memory_io";
    std::string base_codes_quantization_type = "pq";
    int base_codes_pq_dim = 8;
    std::string precise_codes_io_type = "block_memory_io";
    std::string graph_io_type = "block_memory_io";
    std::string graph_storage_type = "flat";
    std::string precise_codes_quantization_type = "fp32";
    int max_degree = 26;
    bool support_remove = true;
    int remove_flag_bit = 8;
    bool use_attribute_filter = false;
    bool use_reorder = true;
    bool have_extro_info = false;
};

std::string
generate_hgraph_param(const HGraphDefaultParam& param) {
    static constexpr auto param_str = R"({{
        "base_codes": {{
            "codes_type": "flatten_codes",
            "io_params": {{
                "file_path": "./default_file_path",
                "type": "{}"
            }},
            "quantization_params": {{
                "pca_dim": 0,
                "pq_dim": {},
                "sq4_uniform_trunc_rate": 0.05,
                "type": "{}"
            }}
        }},
        "build_by_base": false,
        "extra_info": {{
            "io_params": {{
                "file_path": "./default_file_path",
                "type": "block_memory_io"
            }}
        }},
        "graph": {{
            "graph_storage_type": "{}",
            "init_capacity": 100,
            "io_params": {{
                "file_path": "./default_file_path",
                "type": "block_memory_io"
            }},
            "max_degree": {},
            "support_remove": {},
            "remove_flag_bit": {}
        }},
        "ignore_reorder": false,
        "precise_codes": {{
            "codes_type": "flatten_codes",
            "io_params": {{
                "file_path": "./default_file_path",
                "type": "{}"
            }},
            "quantization_params": {{
                "pca_dim": 0,
                "pq_dim": 1,
                "sq4_uniform_trunc_rate": 0.05,
                "type": "{}"
            }}
        }},
        "type": "hgraph",
        "use_attribute_filter": {},
        "use_reorder": {}
    }})";

    return fmt::format(param_str,
                       param.base_codes_io_type,
                       param.base_codes_pq_dim,
                       param.base_codes_quantization_type,
                       param.graph_storage_type,
                       param.max_degree,
                       param.support_remove,
                       param.remove_flag_bit,
                       param.precise_codes_io_type,
                       param.precise_codes_quantization_type,
                       param.use_attribute_filter,
                       param.use_reorder);
}

TEST_CASE("HGraph Parameters CheckCompatibility", "[ut][HGraphParameter][CheckCompatibility]") {
    SECTION("wrong parameter type") {
        HGraphDefaultParam default_param;
        auto param_str = generate_hgraph_param(default_param);
        auto param = std::make_shared<vsag::HGraphParameter>();
        param->FromString(param_str);
        REQUIRE(param->CheckCompatibility(param));
        REQUIRE_FALSE(param->CheckCompatibility(std::make_shared<vsag::EmptyParameter>()));
    }

    TEST_COMPATIBILITY_CASE(
        "different base codes io type", base_codes_io_type, "memory_io", "block_memory_io", true)
    TEST_COMPATIBILITY_CASE("different pq dim", base_codes_pq_dim, 8, 16, false)
    TEST_COMPATIBILITY_CASE(
        "different base codes quantization type", base_codes_quantization_type, "sq4", "sq8", false)
    TEST_COMPATIBILITY_CASE("different graph type", graph_storage_type, "flat", "compressed", false)
    TEST_COMPATIBILITY_CASE("different max degree", max_degree, 26, 30, false)
    TEST_COMPATIBILITY_CASE("different support remove", support_remove, true, false, false)
    TEST_COMPATIBILITY_CASE("different remove flag bit", remove_flag_bit, 8, 16, false)
    TEST_COMPATIBILITY_CASE("different use reorder", use_reorder, true, false, false)
    TEST_COMPATIBILITY_CASE("different precise codes io type",
                            precise_codes_io_type,
                            "memory_io",
                            "block_memory_io",
                            true)
    TEST_COMPATIBILITY_CASE("different precise codes quantization type",
                            precise_codes_quantization_type,
                            "fp32",
                            "sq8",
                            false)
    TEST_COMPATIBILITY_CASE(
        "different use attribute filter", use_attribute_filter, true, false, false)
}
