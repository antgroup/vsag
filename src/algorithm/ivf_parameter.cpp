
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

    if (json.contains(BUCKET_PER_DATA_KEY)) {
        this->buckets_per_data = json[BUCKET_PER_DATA_KEY];
    }

    this->bucket_param = std::make_shared<BucketDataCellParameter>();
    CHECK_ARGUMENT(json.contains(BUCKET_PARAMS_KEY),
                   fmt::format("ivf parameters must contains {}", BUCKET_PARAMS_KEY));
    this->bucket_param->FromJson(json[BUCKET_PARAMS_KEY]);

    this->ivf_partition_strategy_parameter = std::make_shared<IVFPartitionStrategyParameters>();
    if (json.contains(IVF_PARTITION_STRATEGY_PARAMS_KEY)) {
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
    json[TYPE_KEY] = INDEX_IVF;
    json[BUCKET_PARAMS_KEY] = this->bucket_param->ToJson();
    json[IVF_PARTITION_STRATEGY_PARAMS_KEY] = this->ivf_partition_strategy_parameter->ToJson();
    json[BUCKET_PER_DATA_KEY] = this->buckets_per_data;
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
}  // namespace vsag
