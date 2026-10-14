
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

#include "parameter_test.h"
#include "unittest.h"
TEST_CASE("IVF Partition Strategy Parameters Test", "[ut][IVFPartitionStrategyParameters]") {
    auto param_str = R"({
        "partition_strategy_type": "gno_imi",
        "ivf_train_type": "random", 
        "gno_imi": {
            "first_order_buckets_count": 200,
            "second_order_buckets_count": 50
        }
    })";
    vsag::JsonType param_json = vsag::JsonType::Parse(param_str);
    auto param = std::make_shared<vsag::IVFPartitionStrategyParameters>();
    param->FromJson(param_json);
    REQUIRE(param->partition_strategy_type == vsag::IVFPartitionStrategyType::GNO_IMI);
    REQUIRE(param->partition_train_type == vsag::IVFNearestPartitionTrainerType::RandomTrainer);
    REQUIRE(param->gnoimi_param->first_order_buckets_count == 200);
    REQUIRE(param->gnoimi_param->second_order_buckets_count == 50);
    REQUIRE(param->route_max_degree == 64);
    REQUIRE(param->route_ef_construction == 300);

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

TEST_CASE("IVF Partition Strategy Routing Parameters", "[ut][IVFPartitionStrategyParameters]") {
    auto param = std::make_shared<vsag::IVFPartitionStrategyParameters>();
    param->FromString(R"(
    {
        "partition_strategy_type": "ivf",
        "ivf_train_type": "kmeans",
        "route_max_degree": 128,
        "route_ef_construction": 512,
        "use_route_graph": false
    })");

    REQUIRE(param->route_max_degree == 128);
    REQUIRE(param->route_ef_construction == 512);
    REQUIRE_FALSE(param->use_route_graph);
    vsag::ParameterTest::TestToJson(param);

    auto compatible_layout_preference =
        std::make_shared<vsag::IVFPartitionStrategyParameters>(*param);
    compatible_layout_preference->use_route_graph = true;
    REQUIRE(param->CheckCompatibility(compatible_layout_preference));

    auto incompatible = std::make_shared<vsag::IVFPartitionStrategyParameters>(*param);
    incompatible->route_max_degree = 64;
    REQUIRE_FALSE(param->CheckCompatibility(incompatible));

    REQUIRE_THROWS(param->FromString(R"(
    {
        "partition_strategy_type": "ivf",
        "ivf_train_type": "kmeans",
        "route_max_degree": 0
    })"));
    REQUIRE_THROWS(param->FromString(R"(
    {
        "partition_strategy_type": "ivf",
        "ivf_train_type": "kmeans",
        "route_ef_construction": 0
    })"));
    REQUIRE_THROWS(param->FromString(R"(
    {
        "partition_strategy_type": "ivf",
        "ivf_train_type": "kmeans",
        "route_max_degree": 3
    })"));
    REQUIRE_THROWS(param->FromString(R"(
    {
        "partition_strategy_type": "ivf",
        "ivf_train_type": "kmeans",
        "route_max_degree": 512,
        "route_ef_construction": 300
    })"));
}

TEST_CASE("IVF Partition Strategy GPU Build Parameters", "[ut][IVFPartitionStrategyParameters]") {
    auto param = std::make_shared<vsag::IVFPartitionStrategyParameters>();

    // Off, on device 0, with both budgets derived, unless the caller says
    // otherwise. An index built without asking for the backend is unaffected.
    REQUIRE_FALSE(param->enable_gpu_build);
    REQUIRE(param->gpu_device_id == 0);
    REQUIRE(param->gpu_memory_budget == 0);
    REQUIRE(param->gpu_min_work_threshold == 0);

    param->FromString(R"(
    {
        "partition_strategy_type": "ivf",
        "ivf_train_type": "kmeans",
        "enable_gpu_build": true,
        "gpu_device_id": 3,
        "gpu_memory_budget": 1073741824,
        "gpu_min_work_threshold": 2147483648
    })");
    REQUIRE(param->enable_gpu_build);
    REQUIRE(param->gpu_device_id == 3);
    REQUIRE(param->gpu_memory_budget == 1073741824ULL);
    REQUIRE(param->gpu_min_work_threshold == 2147483648ULL);
    vsag::ParameterTest::TestToJson(param);

    // All three are read as signed and stored as an ordinal or a size, so a
    // negative has to be rejected rather than wrapped into a plausible value.
    //
    // Every case below carries ivf_train_type, without which FromJson throws for
    // its own reasons and the assertion would pass whatever the value is.
    REQUIRE_NOTHROW(param->FromString(R"(
    {
        "partition_strategy_type": "ivf",
        "ivf_train_type": "kmeans",
        "gpu_device_id": 1,
        "gpu_memory_budget": 1024,
        "gpu_min_work_threshold": 1024
    })"));
    REQUIRE_THROWS(param->FromString(R"(
    {
        "partition_strategy_type": "ivf",
        "ivf_train_type": "kmeans",
        "gpu_device_id": -1
    })"));
    REQUIRE_THROWS(param->FromString(R"(
    {
        "partition_strategy_type": "ivf",
        "ivf_train_type": "kmeans",
        "gpu_memory_budget": -1
    })"));
    REQUIRE_THROWS(param->FromString(R"(
    {
        "partition_strategy_type": "ivf",
        "ivf_train_type": "kmeans",
        "gpu_min_work_threshold": -1
    })"));

    // The ordinal is the only one of the three that narrows: it is read as a
    // 64-bit integer and stored in an int32_t, so both sides of that range are
    // pinned here. The two budgets widen into a uint64_t and have no such edge.
    REQUIRE_NOTHROW(param->FromString(R"(
    {
        "partition_strategy_type": "ivf",
        "ivf_train_type": "kmeans",
        "gpu_device_id": 2147483647
    })"));
    REQUIRE(param->gpu_device_id == 2147483647);
    REQUIRE_THROWS(param->FromString(R"(
    {
        "partition_strategy_type": "ivf",
        "ivf_train_type": "kmeans",
        "gpu_device_id": 2147483648
    })"));
}

TEST_CASE("IVF Partition Strategy GPU Settings Do Not Affect Compatibility",
          "[ut][IVFPartitionStrategyParameters]") {
    // These decide how the centroids are computed, not what the index looks
    // like afterwards, so an index trained on a device has to stay loadable by
    // a reader configured for the host and the other way round.
    auto host = std::make_shared<vsag::IVFPartitionStrategyParameters>();
    host->FromString(R"({"partition_strategy_type": "ivf", "ivf_train_type": "kmeans"})");

    auto device = std::make_shared<vsag::IVFPartitionStrategyParameters>(*host);
    device->enable_gpu_build = true;
    device->gpu_device_id = 2;
    device->gpu_memory_budget = 1ULL << 30;
    device->gpu_min_work_threshold = 1ULL << 31;

    REQUIRE(host->CheckCompatibility(device));
    REQUIRE(device->CheckCompatibility(host));
}
