
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

#pragma once
#include <fmt/format.h>

#include "datacell/bucket_datacell_parameter.h"
#include "gno_imi_parameter.h"
#include "inner_string_params.h"
#include "parameter.h"
#include "typing.h"

namespace vsag {

enum class IVFNearestPartitionTrainerType {
    RandomTrainer = 0,
    KMeansTrainer = 1,
};

enum class IVFPartitionStrategyType {
    IVF = 0,
    GNO_IMI = 1,
};

// Historical kmeans training iteration counts, kept as defaults for backward compatibility.
constexpr int32_t DEFAULT_IVF_KMEANS_ITER_COUNT = 25;
constexpr int32_t DEFAULT_GNO_IMI_KMEANS_ITER_COUNT = 30;

class IVFPartitionStrategyParameters : public Parameter {
public:
    explicit IVFPartitionStrategyParameters();

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override;

    // Resolved kmeans training iteration count: the configured value when set, otherwise the
    // historical per-strategy default. Always positive.
    int32_t
    GetKMeansIterCount() const;

public:
    IVFNearestPartitionTrainerType partition_train_type{
        IVFNearestPartitionTrainerType::KMeansTrainer};
    IVFPartitionStrategyType partition_strategy_type{IVFPartitionStrategyType::IVF};
    int32_t route_max_degree{64};
    int32_t route_ef_construction{300};
    bool use_route_graph{true};
    GNOIMIParameterPtr gnoimi_param{nullptr};

    // Opt-in CUDA build backend, off by default. These decide how the centroids
    // are computed, not what the index looks like afterwards, which is why
    // CheckCompatibility ignores them.
    bool enable_gpu_build{false};
    // Iterations used by the kmeans centroid training (ivf_train_type == "kmeans").
    // Unset (-1) means "use the per-strategy default", which keeps the historical values:
    // 25 for the single-level IVF strategy and 30 for GNO-IMI.
    int32_t kmeans_iter_count{-1};
    int32_t gpu_device_id{0};
    uint64_t gpu_memory_budget{0};       // 0: derive from free device memory
    uint64_t gpu_min_work_threshold{0};  // 0: use the calibrated default
};

using IVFPartitionStrategyParametersPtr = std::shared_ptr<IVFPartitionStrategyParameters>;
}  // namespace vsag
