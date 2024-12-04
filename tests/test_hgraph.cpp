
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

#include <spdlog/spdlog.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <limits>

#include "simd/simd.h"
#include "test_index.h"
#include "vsag/vsag.h"

TEST_CASE_PERSISTENT_FIXTURE(fixtures::TestIndex,
                             "HGraph Build & ContinueAdd Test",
                             "[ft][hgraph]") {
    auto dims = fixtures::get_common_used_dims(2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    std::string base_quantization_str = GENERATE("sq8", "fp32");
    const std::string name = "hgraph";
    for (auto& dim : dims) {
        auto param = fixtures::generate_hgraph_build_parameters_string(
            metric_type, dim, base_quantization_str);
        auto index = TestFactory(name, param, true);
        TestBuildIndex(index, dim);
        TestContinueAdd(index, dim);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::TestIndex, "HGraph Float General Test", "[ft][hgraph]") {
    auto dims = fixtures::get_common_used_dims(2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    std::string base_quantization_str = GENERATE("sq8", "fp32");
    const std::string name = "hgraph";
    auto search_parameters = R"(
    {
        "hgraph": {
            "ef_search": 100
        }
    }
    )";
    for (auto& dim : dims) {
        auto build_param = fixtures::generate_hgraph_build_parameters_string(
            metric_type, dim, base_quantization_str);
        FastGeneralTest(name, build_param, search_parameters, metric_type, dim);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::TestIndex,
                             "HGraph Factory Test With Exceptions",
                             "[ft][hgraph]") {
    auto name = "hgraph";
    SECTION("Empty parameters") {
        auto param = "{}";
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("No dim param") {
        auto param = R"(
        {{
            "dtype": "float32",
            "metric_type": "l2",
            "index_param": {{
                "base_quantization_type": "sq8"
            }}
        }})";
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid param") {
        auto metric = GENERATE("", "l4", "inner_product", "cosin", "hamming");
        constexpr const char* param_tmp = R"(
        {{
            "dtype": "float32",
            "metric_type": "{}",
            "dim": 23,
            "index_param": {{
                "base_quantization_type": "sq8"
            }}
        }})";
        auto param = fmt::format(param_tmp, metric);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid datatype param") {
        auto datatype = GENERATE("fp32", "uint8_t", "binary", "", "float", "int8");
        constexpr const char* param_tmp = R"(
        {{
            "dtype": "{}",
            "metric_type": "l2",
            "dim": 23,
            "index_param": {{
                "base_quantization_type": "sq8"
            }}
        }})";
        auto param = fmt::format(param_tmp, datatype);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid dim param") {
        int dim = GENERATE(-12, -1, 0);
        constexpr const char* param_tmp = R"(
        {{
            "dtype": "float32",
            "metric_type": "l2",
            "dim": {},
            "index_param": {{
                "base_quantization_type": "sq8"
            }}
        }})";
        auto param = fmt::format(param_tmp, dim);
        REQUIRE_THROWS(TestFactory(name, param, false));
        auto float_param = R"(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 3.51,
            "index_param": {
                "base_quantization_type": "sq8"
            }
        })";
        REQUIRE_THROWS(TestFactory(name, float_param, false));
    }

    SECTION("Miss hgraph param") {
        auto param = GENERATE(
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35,
                "index_param": {{
                }}
            }})",
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35
            }})");
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid hgraph param base_quantization_type") {
        auto base_quantization_types = GENERATE("pq", "fsa", "sq8_uniform");
        // TODO(LHT): test for float param
        constexpr const char* param_temp =
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35,
                "index_param": {{
                    "base_quantization_type": "{}"
                }}
            }})";
        auto param = fmt::format(param_temp, base_quantization_types);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Success") {
        auto dims = fixtures::get_common_used_dims(2);
        auto metric_type = GENERATE("l2", "ip", "cosine");
        std::string base_quantization_str = GENERATE("sq8", "fp32");
        for (auto& dim : dims) {
            auto build_param = fixtures::generate_hgraph_build_parameters_string(
                metric_type, dim, base_quantization_str);
            auto index = TestFactory(name, build_param, true);
            std::string metric = metric_type;
            auto key = KeyGenIndex(
                dim, dataset_base_count, "hgraph_" + metric + "_" + base_quantization_str);
            SaveIndex(key, index, IndexStatus::Factory);
        }
    }
}
