
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

#include "residual_quantizer_parameter.h"

#include "impl/logger/logger.h"

namespace vsag {

ResidualQuantizerParameter::ResidualQuantizerParameter()
    : QuantizerParameter(QUANTIZATION_TYPE_VALUE_RQ) {
}

void
ResidualQuantizerParameter::FromJson(const JsonType& json) {
    base_quantizer_json_ = json;

    if (json.Contains(RQ_CENTROIDS_COUNT)) {
        this->centroids_count_ = json[RQ_CENTROIDS_COUNT].GetInt();
    }

    if (json.Contains(RQ_BASE_QUANTIZATION_TYPE)) {
        auto quantizer_type = json[RQ_BASE_QUANTIZATION_TYPE].GetString();
        base_quantizer_json_[QUANTIZATION_TYPE_KEY].SetString(quantizer_type);
    } else {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format(
                "inner_param: \"{}\" must contains \"rq_base_quantization_type\" when use rq"),
            json.GetString());
    }
}

JsonType
ResidualQuantizerParameter::ToJson() const {
    JsonType json = base_quantizer_json_;
    json[QUANTIZATION_TYPE_KEY].SetString(QUANTIZATION_TYPE_VALUE_RQ);
    return json;
}

bool
ResidualQuantizerParameter::CheckCompatibility(const ParamPtr& other) const {
    auto rq_param = std::dynamic_pointer_cast<ResidualQuantizerParameter>(other);
    if (not rq_param) {
        logger::error(
            "ResidualQuantizerParameter::CheckCompatibility: other parameter is not a "
            "ResidualQuantizerParameter");
        return false;
    }

    if (this->centroids_count_ != rq_param->centroids_count_) {
        return false;
    }

    return this->base_quantizer_json_[QUANTIZATION_TYPE_KEY].GetString() ==
           rq_param->base_quantizer_json_[QUANTIZATION_TYPE_KEY].GetString();
}
}  // namespace vsag
