
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
 * @file transform_quantizer_parameter.h
 * @brief Parameter class for transform quantizer configuration.
 */

#pragma once

#include "inner_string_params.h"
#include "quantization/quantizer_parameter.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER2(TransformQuantizerParam, TransformQuantizerParameter)

/**
 * @brief Parameter class for transform quantizer.
 *
 * Holds configuration for transformation chain and base quantizer.
 */
class TransformQuantizerParameter : public QuantizerParameter {
public:
    TransformQuantizerParameter();

    ~TransformQuantizerParameter() override = default;

    /**
     * @brief Parses parameters from JSON object.
     * @param json JSON configuration object.
     */
    void
    FromJson(const JsonType& json) override;

    /**
     * @brief Converts parameters to JSON object.
     * @return JSON configuration object.
     */
    JsonType
    ToJson() const override;

    /**
     * @brief Checks compatibility with another parameter.
     * @param other Another parameter to check.
     * @return True if parameters are compatible.
     */
    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override;

    /**
     * @brief Splits a string by delimiter.
     * @param input Input string.
     * @param delimiter Delimiter character.
     * @return Vector of split strings.
     */
    static std::vector<std::string>
    SplitString(const std::string& input, char delimiter = ',');

    /**
     * @brief Merges strings with delimiter.
     * @param vec Vector of strings to merge.
     * @param delimiter Delimiter character.
     * @return Merged string.
     */
    static std::string
    MergeStrings(const std::vector<std::string>& vec, char delimiter = ',');

    /**
     * @brief Gets the base quantization type name.
     * @return Quantization type string.
     */
    std::string
    GetBottomQuantizationName() const {
        return base_quantizer_json_[TYPE_KEY].GetString();
    }

public:
    std::vector<std::string> tq_chain_;  ///< Transformation chain type names.
    JsonType base_quantizer_json_;       ///< Base quantizer configuration JSON.
};
}  // namespace vsag
