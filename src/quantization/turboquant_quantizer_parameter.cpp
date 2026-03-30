
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

#include "turboquant_quantizer_parameter.h"

#include <fmt/format.h>

#include "impl/logger/logger.h"
#include "inner_string_params.h"

namespace vsag {

TurboQuantizerParameter::TurboQuantizerParameter()
    : QuantizerParameter(QUANTIZATION_TYPE_VALUE_TURBOQUANT) {
}

void
TurboQuantizerParameter::FromJson(const JsonType& json) {
    auto qjl_projection_dim = static_cast<int64_t>(this->qjl_projection_dim_);

    if (json.Contains(TURBOQUANT_BITS_PER_DIM_KEY)) {
        this->bits_per_dim_ = json[TURBOQUANT_BITS_PER_DIM_KEY].GetInt();
    }
    if (json.Contains(USE_FHT_KEY)) {
        this->use_fht_ = json[USE_FHT_KEY].GetBool();
    }
    if (json.Contains(TURBOQUANT_ENABLE_QJL_KEY)) {
        this->enable_qjl_ = json[TURBOQUANT_ENABLE_QJL_KEY].GetBool();
    }
    if (json.Contains(TURBOQUANT_QJL_PROJECTION_DIM_KEY)) {
        qjl_projection_dim = json[TURBOQUANT_QJL_PROJECTION_DIM_KEY].GetInt();
    }

    if (this->bits_per_dim_ < 2 or this->bits_per_dim_ > 8) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("currently, only support turboquant_bits_per_dim in [2, 8], but got {}",
                        this->bits_per_dim_));
    }

    if (qjl_projection_dim < 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "turboquant_qjl_projection_dim must be non-negative");
    }
    if (this->enable_qjl_ and qjl_projection_dim == 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "turboquant_qjl_projection_dim must be positive when QJL is enabled");
    }
    this->qjl_projection_dim_ = static_cast<uint64_t>(qjl_projection_dim);
}

JsonType
TurboQuantizerParameter::ToJson() const {
    JsonType json;
    json[TYPE_KEY].SetString(QUANTIZATION_TYPE_VALUE_TURBOQUANT);
    json[TURBOQUANT_BITS_PER_DIM_KEY].SetInt(this->bits_per_dim_);
    json[USE_FHT_KEY].SetBool(this->use_fht_);
    json[TURBOQUANT_ENABLE_QJL_KEY].SetBool(this->enable_qjl_);
    json[TURBOQUANT_QJL_PROJECTION_DIM_KEY].SetInt(this->qjl_projection_dim_);
    return json;
}

bool
TurboQuantizerParameter::CheckCompatibility(const ParamPtr& other) const {
    auto turboquant_param = std::dynamic_pointer_cast<TurboQuantizerParameter>(other);
    if (not turboquant_param) {
        logger::error(
            "TurboQuantizerParameter::CheckCompatibility: other parameter is not a "
            "TurboQuantizerParameter");
        return false;
    }

    if (this->bits_per_dim_ != turboquant_param->bits_per_dim_) {
        logger::error(
            "TurboQuantizerParameter::CheckCompatibility: bits per dim do not match: "
            "{} vs {}",
            this->bits_per_dim_,
            turboquant_param->bits_per_dim_);
        return false;
    }
    if (this->use_fht_ != turboquant_param->use_fht_) {
        logger::error(
            "TurboQuantizerParameter::CheckCompatibility: use_fht flag does not match: {} vs {}",
            this->use_fht_,
            turboquant_param->use_fht_);
        return false;
    }
    if (this->enable_qjl_ != turboquant_param->enable_qjl_) {
        logger::error(
            "TurboQuantizerParameter::CheckCompatibility: enable_qjl flag does not match: {} vs "
            "{}",
            this->enable_qjl_,
            turboquant_param->enable_qjl_);
        return false;
    }
    if (this->qjl_projection_dim_ != turboquant_param->qjl_projection_dim_) {
        logger::error(
            "TurboQuantizerParameter::CheckCompatibility: qjl projection dim does not "
            "match: {} vs {}",
            this->qjl_projection_dim_,
            turboquant_param->qjl_projection_dim_);
        return false;
    }

    return true;
}

}  // namespace vsag
