
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
#include <numeric>

#include "parameter_test.h"
#include "utils/util_functions.h"

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
        "build_thread_count": 3,
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

    vsag::JsonType param_json = vsag::JsonType::Parse(param_str);
    auto param = std::make_shared<vsag::IVFParameter>();
    param->FromJson(param_json);
    REQUIRE(param->bucket_param->buckets_count == 3);
    REQUIRE(param->ivf_partition_strategy_parameter->partition_strategy_type ==
            vsag::IVFPartitionStrategyType::IVF);
    REQUIRE(param->ivf_partition_strategy_parameter->partition_train_type ==
            vsag::IVFNearestPartitionTrainerType::KMeansTrainer);
    REQUIRE(param->buckets_per_data == 1);
    REQUIRE(param->use_reorder == true);
    REQUIRE(param->build_thread_count == 3);
    REQUIRE(param->precise_codes_param->quantizer_parameter->GetTypeName() == "fp32");
    REQUIRE(param->train_sample_count == 65536L);

    index_param.ivf_train_type = "random";
    index_param.partition_strategy_type = "gno_imi";
    index_param.buckets_per_data = 2;
    param_str = generate_ivf_param(index_param);
    param_json = vsag::JsonType::Parse(param_str);
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

TEST_CASE("IVF Parameters Train Sample Count Test", "[ut][IVFParameter][train_sample_count]") {
    IVFDefaultParam index_param;
    auto param_str = generate_ivf_param(index_param);

    // Test valid values
    auto json_obj = vsag::JsonType::Parse(param_str);
    json_obj["ivf_train_sample_count"].SetInt(32767);
    auto modified_param_str = json_obj.Dump();

    vsag::JsonType param_json = vsag::JsonType::Parse(modified_param_str);
    auto param = std::make_shared<vsag::IVFParameter>();
    param->FromJson(param_json);
    REQUIRE(param->train_sample_count == 32767);

    json_obj["ivf_train_sample_count"].SetInt(512);
    modified_param_str = json_obj.Dump();

    param_json = vsag::JsonType::Parse(modified_param_str);
    param = std::make_shared<vsag::IVFParameter>();
    param->FromJson(param_json);
    REQUIRE(param->train_sample_count == 512);

    param_str = generate_ivf_param(index_param);
    param_json = vsag::JsonType::Parse(param_str);
    param = std::make_shared<vsag::IVFParameter>();
    param->FromJson(param_json);
    REQUIRE(param->train_sample_count == 65536L);

    // Test invalid value less than minimum 512
    json_obj = vsag::JsonType::Parse(param_str);
    json_obj["ivf_train_sample_count"].SetInt(100);  // Invalid value, less than minimum 512
    modified_param_str = json_obj.Dump();

    param_json = vsag::JsonType::Parse(modified_param_str);
    param = std::make_shared<vsag::IVFParameter>();

    REQUIRE_THROWS_AS(param->FromJson(param_json), vsag::VsagException);

    // Test invalid value exceeding maximum 65536
    json_obj = vsag::JsonType::Parse(param_str);
    json_obj["ivf_train_sample_count"].SetInt(1000000);
    modified_param_str = json_obj.Dump();

    param_json = vsag::JsonType::Parse(modified_param_str);
    param = std::make_shared<vsag::IVFParameter>();

    REQUIRE_THROWS_AS(param->FromJson(param_json), vsag::VsagException);
}

TEST_CASE("IVF Sampling Logic Test", "[ut][IVFParameter][sampling]") {
    SECTION("Train sample count affects actual sampling") {
        // This test conceptually verifies that different train_sample_count values
        // would lead to different sampling behavior in the IVF implementation
        // Note: Actual sampling behavior is tested in ivf.cpp unit tests

        IVFDefaultParam index_param;
        auto param_str = generate_ivf_param(index_param);

        // Test that the parameter correctly stores the configured sample count
        auto json_obj = vsag::JsonType::Parse(param_str);
        json_obj["ivf_train_sample_count"].SetInt(20000);
        auto modified_param_str = json_obj.Dump();

        vsag::JsonType param_json = vsag::JsonType::Parse(modified_param_str);
        auto param = std::make_shared<vsag::IVFParameter>();
        param->FromJson(param_json);
        REQUIRE(param->train_sample_count == 20000);

        // Verify that this value is different from the default
        REQUIRE(param->train_sample_count != 65536L);
    }
}

TEST_CASE("SampleTrainingData Function Test", "[ut][sample_train_data]") {
    // Create allocator
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    // Test with small dataset that should not be sampled
    auto small_dataset = vsag::Dataset::Make();
    const int64_t small_dim = 10;
    const int64_t small_count = 500;

    // Create test data
    std::vector<float> small_data(small_dim * small_count);
    std::iota(small_data.begin(), small_data.end(), 0.0f);

    std::vector<int64_t> small_ids(small_count);
    std::iota(small_ids.begin(), small_ids.end(), 0);

    small_dataset->Dim(small_dim)
        ->NumElements(small_count)
        ->Ids(small_ids.data())
        ->Float32Vectors(small_data.data())
        ->Owner(false);

    // Test that small dataset is returned as is
    auto result =
        vsag::sample_train_data(small_dataset, small_count, small_dim, 10000, allocator.get());
    REQUIRE(result == small_dataset);

    // Test with large dataset that should be sampled
    auto large_dataset = vsag::Dataset::Make();
    const int64_t large_dim = 10;
    const int64_t large_count = 10000;
    const int64_t sample_count = 5000;

    // Create test data
    std::vector<float> large_data(large_dim * large_count);
    std::iota(large_data.begin(), large_data.end(), 0.0f);

    std::vector<int64_t> large_ids(large_count);
    std::iota(large_ids.begin(), large_ids.end(), 0);

    large_dataset->Dim(large_dim)
        ->NumElements(large_count)
        ->Ids(large_ids.data())
        ->Float32Vectors(large_data.data())
        ->Owner(false);

    // Test that large dataset is sampled
    result = vsag::sample_train_data(
        large_dataset, large_count, large_dim, sample_count, allocator.get());
    REQUIRE(result != large_dataset);
    REQUIRE(result->GetNumElements() == sample_count);
    REQUIRE(result->GetDim() == large_dim);

    // Test with train_sample_count less than min_train_size
    // In this case, the function should use min_train_size (512) as the sample count
    const int64_t normal_count = 20000;
    auto normal_dataset = vsag::Dataset::Make();
    std::vector<float> normal_data(large_dim * normal_count);
    std::iota(normal_data.begin(), normal_data.end(), 0.0f);

    std::vector<int64_t> normal_ids(normal_count);
    std::iota(normal_ids.begin(), normal_ids.end(), 0);

    normal_dataset->Dim(large_dim)
        ->NumElements(normal_count)
        ->Ids(normal_ids.data())
        ->Float32Vectors(normal_data.data())
        ->Owner(false);

    // When train_sample_count is less than min_train_size (512),
    // the function should use MIN_TRAIN_SIZE as the sample count
    const int64_t small_sample_count = 100;  // Less than min_train_size (512)
    result = vsag::sample_train_data(
        normal_dataset, normal_count, large_dim, small_sample_count, allocator.get());
    REQUIRE(result != normal_dataset);
    REQUIRE(result->GetNumElements() == 512);  // Should use min_train_size
    REQUIRE(result->GetDim() == large_dim);
}
