
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

/**
 * @file sparse_quantizer_parameter.h
 * @brief Parameter class for sparse quantizer configuration.
 */

#pragma once

#include "quantization/quantizer_parameter.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER2(SparseQuantizerParam, SparseQuantizerParameter);

/**
 * @brief Parameter class for sparse quantizer.
 *
 * Simple parameter class with no additional configuration options.
 */
class SparseQuantizerParameter : public QuantizerParameter {
public:
    SparseQuantizerParameter() : QuantizerParameter(QUANTIZATION_TYPE_VALUE_SPARSE) {
    }

    ~SparseQuantizerParameter() override = default;

    /**
     * @brief Parses parameters from JSON object (no-op for sparse).
     * @param json JSON configuration object.
     */
    void
    FromJson(const JsonType& json) override {
    }

    /**
     * @brief Converts parameters to JSON object.
     * @return JSON configuration object with type name.
     */
    JsonType
    ToJson() const override {
        JsonType json;
        json[TYPE_KEY].SetString(this->GetTypeName());
        return json;
    }
};
}  // namespace vsag
