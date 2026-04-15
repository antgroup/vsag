
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
 * @file quantizer_parameter.h
 * @brief Base parameter class for quantizer configuration.
 */

#pragma once

#include "parameter.h"
#include "utils/pointer_define.h"
namespace vsag {
DEFINE_POINTER2(QuantizerParam, QuantizerParameter);

/**
 * @brief Base class for quantizer parameters.
 *
 * This class provides a common interface for quantizer configuration,
 * including JSON parsing and type validation.
 */
class QuantizerParameter : public Parameter {
public:
    /**
     * @brief Creates a quantizer parameter instance from JSON configuration.
     *
     * @param json The JSON configuration object.
     * @return A shared pointer to the created quantizer parameter.
     */
    static QuantizerParamPtr
    GetQuantizerParameterByJson(const JsonType& json);

public:
    /**
     * @brief Gets the quantizer type name.
     *
     * @return The type name string.
     */
    inline std::string
    GetTypeName() const {
        return this->name_;
    }

    /**
     * @brief Checks if the given quantization type is valid.
     *
     * @param type_name The quantization type name to validate.
     * @return True if the type is valid, false otherwise.
     */
    static bool
    IsValidQuantizationType(const std::string& type_name);

protected:
    /**
     * @brief Constructs a QuantizerParameter with the given name.
     *
     * @param name The quantizer type name.
     */
    explicit QuantizerParameter(std::string name);

    std::string name_{};  /// Quantizer type name
};

}  // namespace vsag
