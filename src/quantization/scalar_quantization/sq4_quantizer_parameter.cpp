
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

#include "sq4_quantizer_parameter.h"

#include "inner_string_params.h"

namespace vsag {

SQ4QuantizerParameter::SQ4QuantizerParameter() : QuantizerParameter(QUANTIZATION_TYPE_VALUE_SQ4) {
}

void
SQ4QuantizerParameter::FromJson(const JsonType& json) {
}

JsonType
SQ4QuantizerParameter::ToJson() const {
    JsonType json;
    json[QUANTIZATION_TYPE_KEY] = QUANTIZATION_TYPE_VALUE_SQ4;
    return json;
}
}  // namespace vsag
