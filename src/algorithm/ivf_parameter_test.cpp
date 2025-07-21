
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

#include "ivf_parameter.h"

#include <catch2/catch_test_macros.hpp>

#include "parameter_test.h"

struct IVFDefaultParam {
    std::string buckect_io_type = "block_memory_io";
    std::string bucket_quantization_type = "sq8";
    int buckets_count = 3;
    bool use_residual = true;
    bool use_reorder = true;
    std::string precise_codes_io_type = "block_memory_io";
    std::string precise_codes_quantization_type = "fp32";
    std::string partition_strategy_type = "ivf";
    std::string ivf_train_type = "kmeans";
    int buckets_per_data = 1;
    bool use_attribute_filter = true;
};

std::string
generate_ivf_param(const IVFDefaultParam& param) {
    static constexpr auto param_str = R"({{
        "type": "ivf",
        "thread_count": 3,
        "buckets_params": {{
            "io_params": {{
                "type": "{}"
            }},
            "quantization_params": {{
                "type": "{}"
            }},
            "buckets_count": {},
            "use_residual": {}
        }},
        "use_reorder": {},
        "partition_strategy": {{
            "partition_strategy_type": "{}",
            "ivf_train_type": "{}",
            "gno_imi": {{
                "first_order_buckets_count": 200,
                "second_order_buckets_count": 50
            }}
        }},
        "precise_codes": {{
            "io_params": {{
                "type": "{}"
            }},
            "quantization_params": {{
                "type": "{}"
            }}
        }},
        "buckets_per_data": {},
        "use_attribute_filter": {}
    }})";
    return fmt::format(param_str,
                       param.buckect_io_type,
                       param.bucket_quantization_type,
                       param.buckets_count,
                       param.use_residual,
                       param.use_reorder,
                       param.partition_strategy_type,
                       param.ivf_train_type,
                       param.precise_codes_io_type,
                       param.precise_codes_quantization_type,
                       param.buckets_per_data,
                       param.use_attribute_filter);
}

TEST_CASE("IVF Parameters Test", "[ut][IVFParameter]") {
    IVFDefaultParam index_param;
    auto param_str = generate_ivf_param(index_param);

    vsag::JsonType param_json = vsag::JsonType::parse(param_str);
    auto param = std::make_shared<vsag::IVFParameter>();
    param->FromJson(param_json);
    REQUIRE(param->bucket_param->buckets_count == 3);
    REQUIRE(param->ivf_partition_strategy_parameter->partition_strategy_type ==
            vsag::IVFPartitionStrategyType::IVF);
    REQUIRE(param->ivf_partition_strategy_parameter->partition_train_type ==
            vsag::IVFNearestPartitionTrainerType::KMeansTrainer);
    REQUIRE(param->buckets_per_data == 1);
    REQUIRE(param->use_reorder == true);
    REQUIRE(param->thread_count == 3);
    REQUIRE(param->flatten_param->quantizer_parameter->GetTypeName() == "fp32");

    index_param.ivf_train_type = "random";
    index_param.partition_strategy_type = "gno_imi";
    index_param.buckets_per_data = 2;
    param_str = generate_ivf_param(index_param);
    param_json = vsag::JsonType::parse(param_str);
    param = std::make_shared<vsag::IVFParameter>();
    param->FromJson(param_json);
    REQUIRE(param->bucket_param->buckets_count == 200 * 50);
    REQUIRE(param->ivf_partition_strategy_parameter->partition_strategy_type ==
            vsag::IVFPartitionStrategyType::GNO_IMI);
    REQUIRE(param->ivf_partition_strategy_parameter->partition_train_type ==
            vsag::IVFNearestPartitionTrainerType::RandomTrainer);
    REQUIRE(param->ivf_partition_strategy_parameter->gnoimi_param->first_order_buckets_count ==
            200);
    REQUIRE(param->ivf_partition_strategy_parameter->gnoimi_param->second_order_buckets_count ==
            50);
    REQUIRE(param->buckets_per_data == 2);

    vsag::ParameterTest::TestToJson(param);

    param_str = R"(
    {
        "ivf": {
            "scan_buckets_count": 10
        }
    })";
    auto search_param = vsag::IVFSearchParameters::FromJson(param_str);
    REQUIRE(search_param.scan_buckets_count == 10);
    REQUIRE(search_param.first_order_scan_ratio == 1.0f);

    param_str = R"(
    {
        "ivf": {
            "scan_buckets_count": 20,
            "first_order_scan_ratio": 0.1
        }
    })";
    search_param = vsag::IVFSearchParameters::FromJson(param_str);
    REQUIRE(search_param.scan_buckets_count == 20);
    REQUIRE(search_param.first_order_scan_ratio == 0.1f);
}

#define TEST_COMPATIBILITY_CASE(section_name, param_member, val1, val2, expect_compatible) \
    SECTION(section_name) {                                                                \
        IVFDefaultParam param1;                                                            \
        IVFDefaultParam param2;                                                            \
        param1.param_member = val1;                                                        \
        param2.param_member = val2;                                                        \
        auto param_str1 = generate_ivf_param(param1);                                      \
        auto param_str2 = generate_ivf_param(param2);                                      \
        auto ivf_param1 = std::make_shared<vsag::IVFParameter>();                          \
        auto ivf_param2 = std::make_shared<vsag::IVFParameter>();                          \
        ivf_param1->FromString(param_str1);                                                \
        ivf_param2->FromString(param_str2);                                                \
        if (expect_compatible) {                                                           \
            REQUIRE(ivf_param1->CheckCompatibility(ivf_param2));                           \
        } else {                                                                           \
            REQUIRE_FALSE(ivf_param1->CheckCompatibility(ivf_param2));                     \
        }                                                                                  \
    }

TEST_CASE("IVF Parameters CheckCompatibility", "[ut][IVFParameter][CheckCompatibility]") {
    SECTION("wrong parameter type") {
        IVFDefaultParam index_param;
        auto param_str = generate_ivf_param(index_param);
        auto param = std::make_shared<vsag::IVFParameter>();
        param->FromString(param_str);
        REQUIRE(param->CheckCompatibility(param));
        REQUIRE_FALSE(param->CheckCompatibility(std::make_shared<vsag::EmptyParameter>()));
    }

    TEST_COMPATIBILITY_CASE("ivf buckets_count", buckets_count, 3, 4, false);
    TEST_COMPATIBILITY_CASE(
        "ivf bucket io type", buckect_io_type, "block_memory_io", "memory_io", true);
    TEST_COMPATIBILITY_CASE(
        "ivf bucket quantization type", bucket_quantization_type, "sq8", "fp32", false);
    TEST_COMPATIBILITY_CASE("ivf buckect use_residual", use_residual, true, false, false);
    TEST_COMPATIBILITY_CASE("ivf use_reorder", use_reorder, true, false, false);
    TEST_COMPATIBILITY_CASE(
        "ivf precise_codes io type", precise_codes_io_type, "block_memory_io", "memory_io", true);
    TEST_COMPATIBILITY_CASE("ivf precise_codes quantization type",
                            precise_codes_quantization_type,
                            "fp32",
                            "sq8",
                            false);
    TEST_COMPATIBILITY_CASE(
        "ivf partition_strategy_type", partition_strategy_type, "ivf", "gno_imi", false);
    TEST_COMPATIBILITY_CASE("ivf ivf_train_type", ivf_train_type, "kmeans", "random", true);
    TEST_COMPATIBILITY_CASE("ivf buckets_per_data", buckets_per_data, 3, 2, false);
    TEST_COMPATIBILITY_CASE("ivf use_attribute_filter", use_attribute_filter, true, false, false);
}
