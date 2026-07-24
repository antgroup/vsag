
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
#include "utils/param_compat_macros.h"
#include "vsag/constants.h"
namespace vsag {

void
IVFParameter::FromJson(const JsonType& json) {
    InnerIndexParameter::FromJson(json);

    this->precise_codes_layout = PRECISE_CODES_LAYOUT_VALUE_FLAT;
    if (json.Contains(PRECISE_CODES_LAYOUT_KEY)) {
        this->precise_codes_layout = json[PRECISE_CODES_LAYOUT_KEY].GetString();
    }
    CHECK_ARGUMENT(
        this->precise_codes_layout == PRECISE_CODES_LAYOUT_VALUE_FLAT ||
            this->precise_codes_layout == PRECISE_CODES_LAYOUT_VALUE_BUCKET,
        fmt::format("invalid precise_codes_layout: {}, supported values are \"{}\" and \"{}\"",
                    this->precise_codes_layout,
                    PRECISE_CODES_LAYOUT_VALUE_FLAT,
                    PRECISE_CODES_LAYOUT_VALUE_BUCKET));

    if (this->precise_codes_layout == PRECISE_CODES_LAYOUT_VALUE_BUCKET) {
        CHECK_ARGUMENT(this->use_reorder, "precise_codes_layout=bucket requires use_reorder=true");
        CHECK_ARGUMENT(this->reorder_source == HGRAPH_REORDER_SOURCE_PRECISE,
                       "precise_codes_layout=bucket requires reorder_source=precise");
        CHECK_ARGUMENT(this->precise_codes_param != nullptr &&
                           this->precise_codes_param->name == FLATTEN_DATA_CELL,
                       "precise_codes_layout=bucket requires ordinary flatten precise_codes");
        CHECK_ARGUMENT(this->precise_codes_param->quantizer_parameter->GetTypeName() !=
                           QUANTIZATION_TYPE_VALUE_PQFS,
                       "precise_codes_layout=bucket does not support pqfs precise quantization");
        CHECK_ARGUMENT(
            this->precise_codes_param->io_parameter == nullptr ||
                this->precise_codes_param->io_parameter->GetTypeName() != IO_TYPE_VALUE_MMAP_IO,
            "precise_codes_layout=bucket does not support mmap_io");
    }

    if (json.Contains(BUCKET_PER_DATA_KEY)) {
        this->buckets_per_data = static_cast<BucketIdType>(json[BUCKET_PER_DATA_KEY].GetInt());
    }

    this->bucket_param = std::make_shared<BucketDataCellParameter>();

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

bool
IVFParameter::UsesDiskBackedPreciseBucket() const {
    if (this->precise_codes_layout != PRECISE_CODES_LAYOUT_VALUE_BUCKET ||
        this->precise_codes_param == nullptr ||
        this->precise_codes_param->io_parameter == nullptr) {
        return false;
    }
    const auto io_type = this->precise_codes_param->io_parameter->GetTypeName();
    return io_type != IO_TYPE_VALUE_MEMORY_IO && io_type != IO_TYPE_VALUE_BLOCK_MEMORY_IO;
}

JsonType
IVFParameter::ToJson() const {
    JsonType json = InnerIndexParameter::ToJson();
    json[TYPE_KEY].SetString(INDEX_IVF);
    json[BUCKET_PARAMS_KEY].SetJson(this->bucket_param->ToJson());
    json[IVF_PARTITION_STRATEGY_PARAMS_KEY].SetJson(
        this->ivf_partition_strategy_parameter->ToJson());
    json[PRECISE_CODES_LAYOUT_KEY].SetString(this->precise_codes_layout);
    json[BUCKET_PER_DATA_KEY].SetInt(this->buckets_per_data);
    return json;
}
bool
IVFParameter::CheckCompatibility(const ParamPtr& other) const {
    if (not InnerIndexParameter::CheckCompatibility(other)) {
        return false;
    }
    PARAM_CAST_OR_RETURN(IVFParameter, p, other);
    CHECK_FIELD_EQ(*this, *p, precise_codes_layout);
    CHECK_FIELD_EQ(*this, *p, buckets_per_data);
    CHECK_SUB_PARAM(*this, *p, bucket_param);
    CHECK_SUB_PARAM(*this, *p, ivf_partition_strategy_parameter);
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

    if (params[INDEX_TYPE_IVF].Contains(IVF_SEARCH_PARAM_DISABLE_BUCKET_SCAN)) {
        obj.disable_bucket_scan =
            params[INDEX_TYPE_IVF][IVF_SEARCH_PARAM_DISABLE_BUCKET_SCAN].GetBool();
    }

    // set obj.first_order_scan_ratio
    if (params[INDEX_TYPE_IVF].Contains(GNO_IMI_SEARCH_PARAM_FIRST_ORDER_SCAN_RATIO)) {
        obj.first_order_scan_ratio =
            params[INDEX_TYPE_IVF][GNO_IMI_SEARCH_PARAM_FIRST_ORDER_SCAN_RATIO].GetFloat();
    }

    return obj;
}
}  // namespace vsag
