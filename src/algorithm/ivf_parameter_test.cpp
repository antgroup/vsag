
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

TEST_CASE("IVF Parameters Test", "[ut][IVFParameter]") {
    auto param_str = R"({
        "type": "ivf",
        "buckets_params": {
            "io_params": {
                "type": "block_memory_io"
            },
            "quantization_params": {
                "type": "sq8"
            },
            "buckets_count": 3
        },
        "use_reorder": true,
        "precise_codes": {
            "io_params": {
                "type": "block_memory_io"
            },
            "quantization_params": {
                "type": "fp32"
            }
        }
    })";

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
    REQUIRE(param->flatten_param->quantizer_parameter->GetTypeName() == "fp32");

    param_str = R"({
        "type": "ivf",
        "buckets_params": {
            "io_params": {
                "type": "block_memory_io"
            },
            "quantization_params": {
                "type": "fp32"
            },
            "buckets_count": 3
        },
        "partition_strategy": {
            "partition_strategy_type": "gno_imi",
            "ivf_train_type": "random", 
            "gno_imi": {
                "first_order_buckets_count": 200,
                "second_order_buckets_count": 50
            }
        },
        "buckets_per_data": 2
    })";
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
