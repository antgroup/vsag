
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

#include "int8_quantizer_parameter.h"

#include "inner_string_params.h"

namespace vsag {
INT8QuantizerParameter::INT8QuantizerParameter()
    : QuantizerParameter(QUANTIZATION_TYPE_VALUE_INT8) {
}

void
INT8QuantizerParameter::FromJson(const JsonType& json) {
    if (json.Contains(HOLD_MOLDS)) {
        hold_molds = json[HOLD_MOLDS].GetBool();
    }
}

JsonType
INT8QuantizerParameter::ToJson() const {
    JsonType json;
    json[TYPE_KEY].SetString(QUANTIZATION_TYPE_VALUE_INT8);
    return json;
}
}  // namespace vsag
