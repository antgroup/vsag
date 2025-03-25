
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

#include <fmt/format-inl.h>

#include "inner_string_params.h"
#include "vsag/constants.h"
namespace vsag {

IVFParameter::IVFParameter() = default;

void
IVFParameter::FromJson(const JsonType& json) {
    if (json[IVF_TRAIN_TYPE_KEY] == IVF_TRAIN_TYPE_KMEANS) {
        this->partition_train_type = IVFNearestPartitionTrainerType::KMeansTrainer;
    } else if (json[IVF_TRAIN_TYPE_KEY] == IVF_TRAIN_TYPE_RANDOM) {
        this->partition_train_type = IVFNearestPartitionTrainerType::RandomTrainer;
    }
    this->bucket_param = std::make_shared<BucketDataCellParameter>();
    CHECK_ARGUMENT(json.contains(BUCKET_PARAMS_KEY),
                   fmt::format("ivf parameters must contains {}", BUCKET_PARAMS_KEY));
    this->bucket_param->FromJson(json[BUCKET_PARAMS_KEY]);

    if (json.contains(IVF_USE_REORDER_KEY)) {
        this->use_reorder = json[IVF_USE_REORDER_KEY];
    }

    if (this->use_reorder) {
        CHECK_ARGUMENT(json.contains(IVF_PRECISE_CODES_KEY),
                       fmt::format("ivf parameters must contains {} when enable reorder",
                                   IVF_PRECISE_CODES_KEY));
        this->flatten_param = std::make_shared<FlattenDataCellParameter>();
        this->flatten_param->FromJson(json[IVF_PRECISE_CODES_KEY]);
    }
}

JsonType
IVFParameter::ToJson() {
    JsonType json;
    json["type"] = INDEX_IVF;
    json[BUCKET_PARAMS_KEY] = this->bucket_param->ToJson();
    json[IVF_USE_REORDER_KEY] = this->use_reorder;
    if (use_reorder) {
        json[IVF_PRECISE_CODES_KEY] = this->flatten_param->ToJson();
    }
    return json;
}
}  // namespace vsag
