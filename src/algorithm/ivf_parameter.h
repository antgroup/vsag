
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
};

class IVFSearchParameters {
public:
    static IVFSearchParameters
    FromJson(const std::string& json_string) {
        JsonType params = JsonType::Parse(json_string);

        IVFSearchParameters obj;

        // set obj.scan_buckets_count
        CHECK_ARGUMENT(params.Contains(INDEX_TYPE_IVF),
                       fmt::format("parameters must contains {}", INDEX_TYPE_IVF));

        CHECK_ARGUMENT(params[INDEX_TYPE_IVF].Contains(IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT),
                       fmt::format("parameters[{}] must contains {}",
                                   INDEX_TYPE_IVF,
                                   IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT));
        obj.scan_buckets_count =
            params[INDEX_TYPE_IVF][IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT].GetInt();

        if (params[INDEX_TYPE_IVF].Contains(SEARCH_PARAM_FACTOR)) {
            obj.topk_factor = params[INDEX_TYPE_IVF][SEARCH_PARAM_FACTOR].GetFloat();
        }

        if (params[INDEX_TYPE_IVF].Contains(GNO_IMI_SEARCH_PARAM_FIRST_ORDER_SCAN_RATIO)) {
            obj.first_order_scan_ratio =
                params[INDEX_TYPE_IVF][GNO_IMI_SEARCH_PARAM_FIRST_ORDER_SCAN_RATIO].GetFloat();
        }

        if (params[INDEX_TYPE_IVF].Contains(IVF_SEARCH_PARALLELISM)) {
            obj.parallel_search_thread_count =
                params[INDEX_TYPE_IVF][IVF_SEARCH_PARALLELISM].GetInt();
        }

        if (params[INDEX_TYPE_IVF].Contains(SEARCH_MAX_TIME_COST_MS)) {
            obj.timeout_ms = params[INDEX_TYPE_IVF][SEARCH_MAX_TIME_COST_MS].GetInt();
            obj.enable_time_record = true;
        }

        return obj;
    }

public:
    int64_t scan_buckets_count{30};
    float topk_factor{2.0F};
    float first_order_scan_ratio{1.0F};
    int64_t parallel_search_thread_count{1};
    double timeout_ms{std::numeric_limits<double>::max()};
    bool enable_time_record{false};

private:
    IVFSearchParameters() = default;
};

}  // namespace vsag
