
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

#include "quantization/quantizer_parameter.h"
#include "typing.h"

namespace vsag {
class SparseQuantizerParameter : public QuantizerParameter {
public:
    SparseQuantizerParameter() : QuantizerParameter(QUANTIZATION_TYPE_VALUE_SPARSE) {
    }

    ~SparseQuantizerParameter() override = default;

    void
    FromJson(const JsonType& json) override {
    }

    JsonType
    ToJson() override {
        JsonType json;
        json[QUANTIZATION_TYPE_KEY] = this->GetTypeName();
        return json;
    }
};

using SparseQuantizerParamPtr = std::shared_ptr<SparseQuantizerParameter>;
}  // namespace vsag
