
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

#include "rabitq_quantizer_parameter.h"

#include <cmath>

#include "inner_string_params.h"
#include "utils/param_compat_macros.h"

namespace vsag {

RaBitQuantizerParameter::RaBitQuantizerParameter()
    : QuantizerParameter(QUANTIZATION_TYPE_VALUE_RABITQ) {
}

void
RaBitQuantizerParameter::FromJson(const JsonType& json) {
    if (json.Contains(PCA_DIM_KEY)) {
        this->pca_dim_ = json[PCA_DIM_KEY].GetUint64();
    }
    if (json.Contains(RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY)) {
        this->num_bits_per_dim_query_ =
            json[RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY].GetUint64();
    }

    if (num_bits_per_dim_query_ != 4 and num_bits_per_dim_query_ != 32) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("currently, only support rabitq_bits_per_dim_query = 4 or 32, but got {}",
                        num_bits_per_dim_query_));
    }

    if (json.Contains(RABITQ_QUANTIZATION_BITS_PER_DIM_BASE_KEY)) {
        this->num_bits_per_dim_base_ = json[RABITQ_QUANTIZATION_BITS_PER_DIM_BASE_KEY].GetUint64();
    }

    if (num_bits_per_dim_base_ > 8 or num_bits_per_dim_base_ < 1) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("currently, only support rabitq_bits_per_dim_base in [1, 8], but got {}",
                        num_bits_per_dim_base_));
    }
    if (json.Contains(RABITQ_QUANTIZATION_BITS_PER_DIM_FILTER_KEY)) {
        this->num_bits_per_dim_filter_ =
            json[RABITQ_QUANTIZATION_BITS_PER_DIM_FILTER_KEY].GetUint64();
    }
    if (num_bits_per_dim_filter_ < 1 or num_bits_per_dim_filter_ > num_bits_per_dim_base_) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("rabitq_bits_per_dim_filter must be in [1, rabitq_bits_per_dim_base], "
                        "but got {} with rabitq_bits_per_dim_base={}",
                        num_bits_per_dim_filter_,
                        num_bits_per_dim_base_));
    }
    if (json.Contains(RABITQ_QUANTIZATION_VERSION_KEY)) {
        this->rabitq_version_ = json[RABITQ_QUANTIZATION_VERSION_KEY].GetString();
    }
    if (this->rabitq_version_ != DEFAULT_RABITQ_VERSION and
        not IsSplitVersion(this->rabitq_version_)) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("unsupported rabitq_version: {}. Supported values are standard, split",
                        rabitq_version_));
    }
    if (IsSplitVersion(this->rabitq_version_)) {
        this->rabitq_version_ = RABITQ_VERSION_SPLIT;
    }
    if (this->rabitq_version_ == RABITQ_VERSION_SPLIT and this->num_bits_per_dim_query_ != 32) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "rabitq_version=split requires rabitq_bits_per_dim_query=32");
    }
    if (json.Contains(RABITQ_QUANTIZATION_ERROR_RATE_KEY)) {
        this->rabitq_error_rate_ = json[RABITQ_QUANTIZATION_ERROR_RATE_KEY].GetFloat();
    }
    if (not std::isfinite(this->rabitq_error_rate_) || this->rabitq_error_rate_ <= 0.0F) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("rabitq_error_rate must be finite and positive, got {}",
                                        rabitq_error_rate_));
    }
    if (json.Contains(USE_FHT_KEY)) {
        this->use_fht_ = json[USE_FHT_KEY].GetBool();
    }
}

JsonType
RaBitQuantizerParameter::ToJson() const {
    JsonType json;
    json[TYPE_KEY].SetString(QUANTIZATION_TYPE_VALUE_RABITQ);
    json[PCA_DIM_KEY].SetUint64(this->pca_dim_);
    json[RABITQ_QUANTIZATION_VERSION_KEY].SetString(this->rabitq_version_);
    json[RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY].SetUint64(this->num_bits_per_dim_query_);
    json[RABITQ_QUANTIZATION_BITS_PER_DIM_BASE_KEY].SetUint64(this->num_bits_per_dim_base_);
    json[RABITQ_QUANTIZATION_BITS_PER_DIM_FILTER_KEY].SetUint64(this->num_bits_per_dim_filter_);
    json[RABITQ_QUANTIZATION_ERROR_RATE_KEY].SetFloat(this->rabitq_error_rate_);
    json[USE_FHT_KEY].SetBool(this->use_fht_);
    return json;
}

bool
RaBitQuantizerParameter::CheckCompatibility(const ParamPtr& other) const {
    PARAM_CAST_OR_RETURN(RaBitQuantizerParameter, p, other);
    CHECK_FIELD_EQ(*this, *p, pca_dim_);
    CHECK_FIELD_EQ(*this, *p, num_bits_per_dim_query_);
    CHECK_FIELD_EQ(*this, *p, num_bits_per_dim_base_);
    CHECK_FIELD_EQ(*this, *p, num_bits_per_dim_filter_);
    CHECK_FIELD_EQ(*this, *p, rabitq_version_);
    CHECK_FIELD_EQ(*this, *p, use_fht_);
    return true;
}
}  // namespace vsag
