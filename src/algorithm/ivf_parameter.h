
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

#include "algorithm/ivf_partition/ivf_nearest_partition.h"
#include "algorithm/ivf_partition/ivf_partition_strategy_parameter.h"
#include "datacell/bucket_datacell_parameter.h"
#include "datacell/flatten_datacell_parameter.h"
#include "index_search_parameter.h"
#include "inner_index_parameter.h"
#include "inner_string_params.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

DEFINE_POINTER(IVFParameter);
class IVFParameter : public InnerIndexParameter {
public:
    explicit IVFParameter() = default;

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override;

public:
    BucketDataCellParamPtr bucket_param{nullptr};
    IVFPartitionStrategyParametersPtr ivf_partition_strategy_parameter{nullptr};
    BucketIdType buckets_per_data{1};
    int64_t train_sample_count{65536L};
};

class IVFSearchParameters : public IndexSearchParameter {
public:
    static IVFSearchParameters
    FromJson(const std::string& json_string);

public:
    int64_t scan_buckets_count{30};
    float topk_factor{2.0F};
    float first_order_scan_ratio{1.0F};

private:
    IVFSearchParameters() = default;
};

}  // namespace vsag
