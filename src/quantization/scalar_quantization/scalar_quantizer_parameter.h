
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

#pragma once

#include "inner_string_params.h"
#include "quantization/quantizer_parameter.h"
#include "utils/pointer_define.h"
namespace vsag {
template <int bit = 8>
class ScalarQuantizerParameter : public QuantizerParameter {
public:
    ScalarQuantizerParameter();

    ~ScalarQuantizerParameter() override = default;

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;
};

template <int bit>
ScalarQuantizerParameter<bit>::ScalarQuantizerParameter()
    : QuantizerParameter(QUANTIZATION_TYPE_VALUE_SQ8) {
    static_assert(bit == 4 || bit == 8, "bit must be 4 or 8");
    if constexpr (bit == 8) {
        this->name_ = QUANTIZATION_TYPE_VALUE_SQ8;
    } else if constexpr (bit == 4) {
        this->name_ = QUANTIZATION_TYPE_VALUE_SQ4;
    }
}

template <int bit>
void
ScalarQuantizerParameter<bit>::FromJson(const JsonType& json) {
}

template <int bit>
JsonType
ScalarQuantizerParameter<bit>::ToJson() const {
    JsonType json;
    json[TYPE_KEY].SetString(this->GetTypeName());
    return json;
}

}  // namespace vsag
