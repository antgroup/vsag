
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
#include "data_cell/bucket_datacell_parameter.h"
#include "data_cell/flatten_datacell_parameter.h"
#include "inner_string_params.h"
#include "parameter.h"
#include "typing.h"

namespace vsag {
class IVFParameter : public Parameter {
public:
    explicit IVFParameter();

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() override;

public:
    BucketDataCellParamPtr bucket_param{nullptr};
    IVFPartitionStrategyParametersPtr ivf_partition_strategy_parameter{nullptr};
    BucketIdType buckets_per_data{1};

    bool use_reorder{false};

    bool use_attribute_filter{false};

    FlattenDataCellParamPtr flatten_param{nullptr};
};

using IVFParameterPtr = std::shared_ptr<IVFParameter>;

class IVFSearchParameters {
public:
    static IVFSearchParameters
    FromJson(const std::string& json_string) {
        JsonType params = JsonType::parse(json_string);

        IVFSearchParameters obj;

        // set obj.scan_buckets_count
        CHECK_ARGUMENT(params.contains(INDEX_TYPE_IVF),
                       fmt::format("parameters must contains {}", INDEX_TYPE_IVF));

        CHECK_ARGUMENT(params[INDEX_TYPE_IVF].contains(IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT),
                       fmt::format("parameters[{}] must contains {}",
                                   INDEX_TYPE_IVF,
                                   IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT));
        obj.scan_buckets_count = params[INDEX_TYPE_IVF][IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT];

        if (params[INDEX_TYPE_IVF].contains(IVF_SEARCH_PARAM_FACTOR)) {
            obj.topk_factor = params[INDEX_TYPE_IVF][IVF_SEARCH_PARAM_FACTOR];
        }

        if (params[INDEX_TYPE_IVF].contains(GNO_IMI_SEARCH_PARAM_FIRST_ORDER_SCAN_RATIO)) {
            obj.first_order_scan_ratio =
                params[INDEX_TYPE_IVF][GNO_IMI_SEARCH_PARAM_FIRST_ORDER_SCAN_RATIO];
        }
        return obj;
    }

public:
    int64_t scan_buckets_count{30};
    float topk_factor{2.0F};
    float first_order_scan_ratio{1.0F};

private:
    IVFSearchParameters() = default;
};

}  // namespace vsag
