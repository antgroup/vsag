
/**
 * @file sq4_uniform_quantizer_parameter.h
 * @brief Parameter class for SQ4 uniform quantizer configuration.
 */

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
#include "utils/pointer_define.h"

namespace vsag {

DEFINE_POINTER2(SQ4UniformQuantizerParam, SQ4UniformQuantizerParameter)

/**
 * @brief Parameter class for configuring SQ4 uniform quantizer.
 */
class SQ4UniformQuantizerParameter : public QuantizerParameter {
public:
    /**
     * @brief Constructs an SQ4UniformQuantizerParameter with default settings.
     */
    SQ4UniformQuantizerParameter();

    ~SQ4UniformQuantizerParameter() override = default;

    /**
     * @brief Loads parameter from JSON configuration.
     * @param json JSON object containing parameter values.
     */
    void
    FromJson(const JsonType& json) override;

    /**
     * @brief Exports parameter to JSON format.
     * @return JSON object with parameter values.
     */
    JsonType
    ToJson() const override;

    /**
     * @brief Checks compatibility with another parameter.
     * @param other Other parameter to compare.
     * @return True if parameters are compatible.
     */
    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override;

public:
    float trunc_rate_{0.05F};  /// Truncation rate for outlier handling
};
}  // namespace vsag
