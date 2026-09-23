
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

#include <fmt/format.h>

#include <iostream>
#include <limits>

#include "inner_string_params.h"
#include "utils/param_compat_macros.h"
#include "vsag/constants.h"

namespace vsag {

namespace {

/// The unsigned GPU knobs, read without letting a value past INT64_MAX through.
///
/// The cap is uniform because it belongs to the reading rather than to any one
/// knob: GetInt is what every value comes through, and a value past INT64_MAX
/// reaches it as a negative. gpu_device_id then narrows again to INT32_MAX at its
/// own call, so the shared ceiling never decides anything for it.
///
/// A whole number written as a float, `1e9`, is taken: it is not
/// `IsNumberUnsigned`, so it goes through GetInt, which converts. That is this
/// parser's doing rather than a guarantee of the JSON number model, which is why
/// ivf_parameter_test.cpp pins it instead of leaving it to be assumed.
///
/// GetInt wraps such a value to a negative, which the old `>= 0` check refused
/// with a message about the wrong bound, so the node is asked what it holds
/// before it is read. The ceiling itself is deliberate and not an accident of
/// GetInt: WorthOffloading turns the threshold into `min_work / dim` and adds
/// `k - 1` to it, and that sum has room only because the threshold cannot reach
/// the top of the range. A threshold of UINT64_MAX would wrap it and offload
/// every shape, which is the opposite of what such a threshold asks for.
uint64_t
gpu_unsigned_or_throw(const JsonType& json, const std::string& key) {
    const JsonType node = json[key];
    if (node.IsNumberUnsigned()) {
        const uint64_t value = node.GetUint64();
        CHECK_ARGUMENT(
            value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            fmt::format("{} must not exceed {}", key, std::numeric_limits<int64_t>::max()));
        return value;
    }
    const int64_t value = node.GetInt();
    CHECK_ARGUMENT(value >= 0, fmt::format("{} must not be negative", key));
    return static_cast<uint64_t>(value);
}

}  // namespace

IVFPartitionStrategyParameters::IVFPartitionStrategyParameters() = default;

void
IVFPartitionStrategyParameters::FromJson(const JsonType& json) {
    if (json[IVF_TRAIN_TYPE_KEY].GetString() == IVF_TRAIN_TYPE_KMEANS) {
        this->partition_train_type = IVFNearestPartitionTrainerType::KMeansTrainer;
    } else if (json[IVF_TRAIN_TYPE_KEY].GetString() == IVF_TRAIN_TYPE_RANDOM) {
        this->partition_train_type = IVFNearestPartitionTrainerType::RandomTrainer;
    }

    if (json[IVF_PARTITION_STRATEGY_TYPE_KEY].GetString() == IVF_PARTITION_STRATEGY_TYPE_NEAREST) {
        this->partition_strategy_type = IVFPartitionStrategyType::IVF;
    } else if (json[IVF_PARTITION_STRATEGY_TYPE_KEY].GetString() ==
               IVF_PARTITION_STRATEGY_TYPE_GNO_IMI) {
        this->partition_strategy_type = IVFPartitionStrategyType::GNO_IMI;
    }
    if (json.Contains(IVF_ROUTE_MAX_DEGREE_KEY)) {
        this->route_max_degree = static_cast<int32_t>(json[IVF_ROUTE_MAX_DEGREE_KEY].GetInt());
        CHECK_ARGUMENT(this->route_max_degree > 0, "route_max_degree must be positive");
    }
    if (json.Contains(IVF_ROUTE_EF_CONSTRUCTION_KEY)) {
        this->route_ef_construction =
            static_cast<int32_t>(json[IVF_ROUTE_EF_CONSTRUCTION_KEY].GetInt());
        CHECK_ARGUMENT(this->route_ef_construction > 0, "route_ef_construction must be positive");
    }
    CHECK_ARGUMENT(this->route_max_degree >= 4, "route_max_degree must be at least 4");
    CHECK_ARGUMENT(this->route_ef_construction >= this->route_max_degree,
                   "route_ef_construction must be no less than route_max_degree");

    if (json.Contains(IVF_USE_ROUTE_GRAPH_KEY)) {
        this->use_route_graph = json[IVF_USE_ROUTE_GRAPH_KEY].GetBool();
    }

    if (json.Contains(IVF_ENABLE_GPU_BUILD_KEY)) {
        this->enable_gpu_build = json[IVF_ENABLE_GPU_BUILD_KEY].GetBool();
    }
    // Each is checked against what it has to fit, so an out-of-range value is
    // refused with a message naming the bound it broke rather than taken.
    if (json.Contains(IVF_GPU_DEVICE_ID_KEY)) {
        const uint64_t device_id = gpu_unsigned_or_throw(json, IVF_GPU_DEVICE_ID_KEY);
        CHECK_ARGUMENT(device_id <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
                       fmt::format("{} must fit in 32 bits", IVF_GPU_DEVICE_ID_KEY));
        this->gpu_device_id = static_cast<int32_t>(device_id);
    }
    if (json.Contains(IVF_GPU_MEMORY_BUDGET_KEY)) {
        this->gpu_memory_budget = gpu_unsigned_or_throw(json, IVF_GPU_MEMORY_BUDGET_KEY);
    }
    if (json.Contains(IVF_GPU_MIN_WORK_THRESHOLD_KEY)) {
        this->gpu_min_work_threshold = gpu_unsigned_or_throw(json, IVF_GPU_MIN_WORK_THRESHOLD_KEY);
    }

    this->gnoimi_param = std::make_shared<GNOIMIParameter>();
    if (this->partition_strategy_type == IVFPartitionStrategyType::GNO_IMI) {
        CHECK_ARGUMENT(
            json.Contains(IVF_PARTITION_STRATEGY_TYPE_GNO_IMI),
            fmt::format("partition strategy parameters must contains {} when strategy type is {}",
                        IVF_PARTITION_STRATEGY_TYPE_GNO_IMI,
                        IVF_PARTITION_STRATEGY_TYPE_GNO_IMI));
        this->gnoimi_param->FromJson(json[IVF_PARTITION_STRATEGY_TYPE_GNO_IMI]);
    }
}

JsonType
IVFPartitionStrategyParameters::ToJson() const {
    JsonType json;
    if (this->partition_train_type == IVFNearestPartitionTrainerType::KMeansTrainer) {
        json[IVF_TRAIN_TYPE_KEY].SetString(IVF_TRAIN_TYPE_KMEANS);
    } else if (this->partition_train_type == IVFNearestPartitionTrainerType::RandomTrainer) {
        json[IVF_TRAIN_TYPE_KEY].SetString(IVF_TRAIN_TYPE_RANDOM);
    }

    if (this->partition_strategy_type == IVFPartitionStrategyType::IVF) {
        json[IVF_PARTITION_STRATEGY_TYPE_KEY].SetString(IVF_PARTITION_STRATEGY_TYPE_NEAREST);
    } else if (this->partition_strategy_type == IVFPartitionStrategyType::GNO_IMI) {
        json[IVF_PARTITION_STRATEGY_TYPE_KEY].SetString(IVF_PARTITION_STRATEGY_TYPE_GNO_IMI);
    }
    json[IVF_ROUTE_MAX_DEGREE_KEY].SetInt(this->route_max_degree);
    json[IVF_ROUTE_EF_CONSTRUCTION_KEY].SetInt(this->route_ef_construction);
    json[IVF_USE_ROUTE_GRAPH_KEY].SetBool(this->use_route_graph);
    json[IVF_ENABLE_GPU_BUILD_KEY].SetBool(this->enable_gpu_build);
    json[IVF_GPU_DEVICE_ID_KEY].SetInt(this->gpu_device_id);
    json[IVF_GPU_MEMORY_BUDGET_KEY].SetUint64(this->gpu_memory_budget);
    json[IVF_GPU_MIN_WORK_THRESHOLD_KEY].SetUint64(this->gpu_min_work_threshold);
    if (this->partition_strategy_type == IVFPartitionStrategyType::GNO_IMI) {
        json[IVF_PARTITION_STRATEGY_TYPE_GNO_IMI].SetJson(this->gnoimi_param->ToJson());
    }
    return json;
}

bool
IVFPartitionStrategyParameters::CheckCompatibility(const ParamPtr& other) const {
    PARAM_CAST_OR_RETURN(IVFPartitionStrategyParameters, p, other);
    CHECK_FIELD_EQ(*this, *p, partition_strategy_type);
    CHECK_FIELD_EQ(*this, *p, route_max_degree);
    CHECK_FIELD_EQ(*this, *p, route_ef_construction);
    // use_route_graph selects the layout only when a new index is trained. Serialized partitions
    // are self-describing and must remain loadable by readers created with either setting.
    CHECK_SUB_PARAM(*this, *p, gnoimi_param);
    return true;
}

}  // namespace vsag
