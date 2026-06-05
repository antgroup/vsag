
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

#include "pq_fastscan_quantizer_parameter.h"

#include "inner_string_params.h"
#include "utils/param_compat_macros.h"

namespace vsag {

PQFastScanQuantizerParameter::PQFastScanQuantizerParameter()
    : QuantizerParameter(QUANTIZATION_TYPE_VALUE_PQFS) {
}

void
PQFastScanQuantizerParameter::FromJson(const JsonType& json) {
    if (json.Contains(PRODUCT_QUANTIZATION_DIM_KEY) &&
        json[PRODUCT_QUANTIZATION_DIM_KEY].IsNumberInteger()) {
        this->pq_dim_ = json[PRODUCT_QUANTIZATION_DIM_KEY].GetInt();
    }
}

JsonType
PQFastScanQuantizerParameter::ToJson() const {
    JsonType json;
    json[TYPE_KEY].SetString(QUANTIZATION_TYPE_VALUE_PQFS);
    json[PRODUCT_QUANTIZATION_DIM_KEY].SetInt(this->pq_dim_);
    return json;
}

bool
PQFastScanQuantizerParameter::CheckCompatibility(const ParamPtr& other) const {
    PARAM_CAST_OR_RETURN(PQFastScanQuantizerParameter, p, other);
    CHECK_FIELD_EQ(*this, *p, pq_dim_);
    return true;
}
}  // namespace vsag
