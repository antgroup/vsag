
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

#include <fmt/format.h>

#include "inner_string_params.h"
#include "vsag/constants.h"
namespace vsag {

void
IVFParameter::FromJson(const JsonType& json) {
    InnerIndexParameter::FromJson(json);

    if (json.Contains(BUCKET_PER_DATA_KEY)) {
        this->buckets_per_data = static_cast<BucketIdType>(json[BUCKET_PER_DATA_KEY].GetInt());
    }

    this->bucket_param = std::make_shared<BucketDataCellParameter>();

    if (json.Contains(IVF_TRAIN_SAMPLE_COUNT_KEY)) {
        this->train_sample_count = json[IVF_TRAIN_SAMPLE_COUNT_KEY].GetInt();
        CHECK_ARGUMENT(
            this->train_sample_count >= 512,
            fmt::format("ivf_train_sample_count must be greater than or equal to 512, got: {}",
                        this->train_sample_count));
        CHECK_ARGUMENT(
            this->train_sample_count <= 65536L,
            fmt::format("ivf_train_sample_count must be less than or equal to 65536, got: {}",
                        this->train_sample_count));
    }

    CHECK_ARGUMENT(json.Contains(BUCKET_PARAMS_KEY),
                   fmt::format("ivf parameters must contains {}", BUCKET_PARAMS_KEY));

    this->bucket_param->FromJson(json[BUCKET_PARAMS_KEY]);

    this->ivf_partition_strategy_parameter = std::make_shared<IVFPartitionStrategyParameters>();
    if (json.Contains(IVF_PARTITION_STRATEGY_PARAMS_KEY)) {
        this->ivf_partition_strategy_parameter->FromJson(json[IVF_PARTITION_STRATEGY_PARAMS_KEY]);
    }

    if (this->ivf_partition_strategy_parameter->partition_strategy_type ==
        IVFPartitionStrategyType::GNO_IMI) {
        this->bucket_param->buckets_count = static_cast<BucketIdType>(
            this->ivf_partition_strategy_parameter->gnoimi_param->first_order_buckets_count *
            this->ivf_partition_strategy_parameter->gnoimi_param->second_order_buckets_count);
    }
}

JsonType
IVFParameter::ToJson() const {
    JsonType json = InnerIndexParameter::ToJson();
    json[TYPE_KEY].SetString(INDEX_IVF);
    json[BUCKET_PARAMS_KEY].SetJson(this->bucket_param->ToJson());
    json[IVF_PARTITION_STRATEGY_PARAMS_KEY].SetJson(
        this->ivf_partition_strategy_parameter->ToJson());
    json[BUCKET_PER_DATA_KEY].SetInt(this->buckets_per_data);
    json[IVF_TRAIN_SAMPLE_COUNT_KEY].SetInt(this->train_sample_count);
    return json;
}
bool
IVFParameter::CheckCompatibility(const ParamPtr& other) const {
    if (not InnerIndexParameter::CheckCompatibility(other)) {
        return false;
    }
    auto ivf_param = std::dynamic_pointer_cast<IVFParameter>(other);
    if (not ivf_param) {
        logger::error("IVFParameter::CheckCompatibility: other parameter is not IVFParameter");
        return false;
    }

    if (this->buckets_per_data != ivf_param->buckets_per_data) {
        logger::error("IVFParameter::CheckCompatibility: buckets_per_data mismatch");
        return false;
    }

    if (not this->bucket_param->CheckCompatibility(ivf_param->bucket_param)) {
        logger::error("IVFParameter::CheckCompatibility: bucket_param mismatch");
        return false;
    }

    if (not this->ivf_partition_strategy_parameter->CheckCompatibility(
            ivf_param->ivf_partition_strategy_parameter)) {
        logger::error(
            "IVFParameter::CheckCompatibility: ivf_partition_strategy_parameter "
            "mismatch");
        return false;
    }
    return true;
}

IVFSearchParameters
IVFSearchParameters::FromJson(const std::string& json_string) {
    JsonType params = JsonType::Parse(json_string);

    IVFSearchParameters obj;

    CHECK_ARGUMENT(params.Contains(INDEX_TYPE_IVF),
                   fmt::format("parameters must contains {}", INDEX_TYPE_IVF));

    obj.IndexSearchParameter::FromJson(params[INDEX_TYPE_IVF]);

    // set obj.scan_buckets_count
    CHECK_ARGUMENT(params[INDEX_TYPE_IVF].Contains(IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT),
                   fmt::format("parameters[{}] must contains {}",
                               INDEX_TYPE_IVF,
                               IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT));
    obj.scan_buckets_count = params[INDEX_TYPE_IVF][IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT].GetInt();

    // set obj.topk_factor
    if (params[INDEX_TYPE_IVF].Contains(SEARCH_PARAM_FACTOR)) {
        obj.topk_factor = params[INDEX_TYPE_IVF][SEARCH_PARAM_FACTOR].GetFloat();
    }

    // set obj.first_order_scan_ratio
    if (params[INDEX_TYPE_IVF].Contains(GNO_IMI_SEARCH_PARAM_FIRST_ORDER_SCAN_RATIO)) {
        obj.first_order_scan_ratio =
            params[INDEX_TYPE_IVF][GNO_IMI_SEARCH_PARAM_FIRST_ORDER_SCAN_RATIO].GetFloat();
    }
    return obj;
}
}  // namespace vsag
