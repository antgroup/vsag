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

#include "saq_quantizer_parameter.h"

#include <fmt/format.h>

#include <cmath>

#include "inner_string_params.h"
#include "utils/param_compat_macros.h"
#include "vsag_exception.h"

namespace vsag {

SAQQuantizerParameter::SAQQuantizerParameter() : QuantizerParameter(QUANTIZATION_TYPE_VALUE_SAQ) {
}

void
SAQQuantizerParameter::FromJson(const JsonType& json) {
    if (json.Contains(SAQ_AVG_BITS_KEY)) {
        CHECK_ARGUMENT(json[SAQ_AVG_BITS_KEY].IsNumber(), "saq_avg_bits must be a number");
        avg_bits_ = json[SAQ_AVG_BITS_KEY].GetFloat();
    }
    const bool valid_avg_bits =
        std::isfinite(avg_bits_) and avg_bits_ >= 1.0F and avg_bits_ <= 8.0F;
    CHECK_ARGUMENT(valid_avg_bits,
                   fmt::format("saq_avg_bits must be finite and in [1, 8], got {}", avg_bits_));

    if (json.Contains(SAQ_SEGMENT_COUNT_KEY)) {
        CHECK_ARGUMENT(json[SAQ_SEGMENT_COUNT_KEY].IsNumberInteger(),
                       "saq_segment_count must be an integer");
        CHECK_ARGUMENT(json[SAQ_SEGMENT_COUNT_KEY].IsNumberUnsigned(),
                       "saq_segment_count must not be negative");
        segment_count_ = json[SAQ_SEGMENT_COUNT_KEY].GetUint64();
    }

    if (json.Contains(SAQ_ADJUSTMENT_ROUNDS_KEY)) {
        CHECK_ARGUMENT(json[SAQ_ADJUSTMENT_ROUNDS_KEY].IsNumberInteger(),
                       "saq_adjustment_rounds must be an integer");
        CHECK_ARGUMENT(json[SAQ_ADJUSTMENT_ROUNDS_KEY].IsNumberUnsigned(),
                       "saq_adjustment_rounds must not be negative");
        adjustment_rounds_ = json[SAQ_ADJUSTMENT_ROUNDS_KEY].GetUint64();
    }
    const bool valid_adjustment_rounds =
        adjustment_rounds_ >= MIN_ADJUSTMENT_ROUNDS and adjustment_rounds_ <= MAX_ADJUSTMENT_ROUNDS;
    CHECK_ARGUMENT(valid_adjustment_rounds,
                   fmt::format("saq_adjustment_rounds must be in [{}, {}], got {}",
                               MIN_ADJUSTMENT_ROUNDS,
                               MAX_ADJUSTMENT_ROUNDS,
                               adjustment_rounds_));

    if (json.Contains(SAQ_USE_PCA_KEY)) {
        CHECK_ARGUMENT(json[SAQ_USE_PCA_KEY].IsBool(), "saq_use_pca must be a boolean");
        use_pca_ = json[SAQ_USE_PCA_KEY].GetBool();
    }
    if (json.Contains(SAQ_RANDOM_ROTATION_KEY)) {
        CHECK_ARGUMENT(json[SAQ_RANDOM_ROTATION_KEY].IsBool(),
                       "saq_random_rotation must be a boolean");
        random_rotation_ = json[SAQ_RANDOM_ROTATION_KEY].GetBool();
    }
}

JsonType
SAQQuantizerParameter::ToJson() const {
    JsonType json;
    json[TYPE_KEY].SetString(QUANTIZATION_TYPE_VALUE_SAQ);
    json[SAQ_AVG_BITS_KEY].SetFloat(avg_bits_);
    json[SAQ_SEGMENT_COUNT_KEY].SetUint64(segment_count_);
    json[SAQ_ADJUSTMENT_ROUNDS_KEY].SetUint64(adjustment_rounds_);
    json[SAQ_USE_PCA_KEY].SetBool(use_pca_);
    json[SAQ_RANDOM_ROTATION_KEY].SetBool(random_rotation_);
    return json;
}

bool
SAQQuantizerParameter::CheckCompatibility(const ParamPtr& other) const {
    PARAM_CAST_OR_RETURN(SAQQuantizerParameter, p, other);
    CHECK_FIELD_EQ(*this, *p, avg_bits_);
    CHECK_FIELD_EQ(*this, *p, segment_count_);
    CHECK_FIELD_EQ(*this, *p, adjustment_rounds_);
    CHECK_FIELD_EQ(*this, *p, use_pca_);
    CHECK_FIELD_EQ(*this, *p, random_rotation_);
    return true;
}

}  // namespace vsag
