
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

#include "ivf_partition_strategy_parameter.h"

#include <catch2/catch_test_macros.hpp>

#include "parameter_test.h"

TEST_CASE("IVF Partition Strategy Parameters Test", "[ut][IVFPartitionStrategyParameters]") {
    auto param_str = R"({
        "partition_strategy_type": "gno_imi",
        "ivf_train_type": "random", 
        "gno_imi": {
            "first_order_buckets_count": 200,
            "second_order_buckets_count": 50
        }
    })";
    vsag::JsonType param_json = vsag::JsonType::parse(param_str);
    auto param = std::make_shared<vsag::IVFPartitionStrategyParameters>();
    param->FromJson(param_json);
    REQUIRE(param->partition_strategy_type == vsag::IVFPartitionStrategyType::GNO_IMI);
    REQUIRE(param->partition_train_type == vsag::IVFNearestPartitionTrainerType::RandomTrainer);
    REQUIRE(param->gnoimi_param->first_order_buckets_count == 200);
    REQUIRE(param->gnoimi_param->second_order_buckets_count == 50);

    vsag::ParameterTest::TestToJson(param);
}

TEST_CASE("IVF Partition Strategy Parameters CheckCompatibility",
          "[ut][IVFPartitionStrategyParameters]") {
    std::string param_str = R"(
    {
        "partition_strategy_type": "gno_imi",
        "ivf_train_type": "random",
        "gno_imi": {
            "first_order_buckets_count": 200,
            "second_order_buckets_count": 50
        }
    })";
    auto param = std::make_shared<vsag::IVFPartitionStrategyParameters>();
    param->FromString(param_str);
    REQUIRE(param->CheckCompatibility(param));
    auto other_type_param = std::make_shared<vsag::EmptyParameter>();
    REQUIRE_FALSE(param->CheckCompatibility(other_type_param));
}
