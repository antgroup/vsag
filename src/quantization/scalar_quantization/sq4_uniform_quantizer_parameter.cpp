
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

#include "sq4_uniform_quantizer_parameter.h"

#include "inner_string_params.h"
#include "utils/param_compat_macros.h"

namespace vsag {
SQ4UniformQuantizerParameter::SQ4UniformQuantizerParameter()
    : QuantizerParameter(QUANTIZATION_TYPE_VALUE_SQ4_UNIFORM) {
}

void
SQ4UniformQuantizerParameter::FromJson(const JsonType& json) {
    if (json.Contains(SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY)) {
        this->trunc_rate_ =
            json[SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY].GetFloat();  // TODO(LHT): Check value
    }
}

JsonType
SQ4UniformQuantizerParameter::ToJson() const {
    JsonType json;
    json[TYPE_KEY].SetString(QUANTIZATION_TYPE_VALUE_SQ4_UNIFORM);
    json[SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY].SetFloat(this->trunc_rate_);
    return json;
}
bool
SQ4UniformQuantizerParameter::CheckCompatibility(const ParamPtr& other) const {
    PARAM_CAST_OR_RETURN(SQ4UniformQuantizerParameter, p, other);
    CHECK_FIELD_EQ(*this, *p, trunc_rate_);
    return true;
}
}  // namespace vsag
