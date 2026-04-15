/**
 * @file flatten_datacell_parameter.h
 * @brief Parameter class for flatten data cell configuration.
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

#include "flatten_interface_parameter.h"
#include "io/io_parameter.h"
#include "parameter.h"
#include "quantization/quantizer_parameter.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER2(FlattenDataCellParam, FlattenDataCellParameter);

/**
 * @brief Parameter class for flatten data cell configuration.
 *
 * This class extends FlattenInterfaceParameter and provides configuration
 * for flatten-based data cell storage with JSON serialization support.
 */
class FlattenDataCellParameter : public FlattenInterfaceParameter {
public:
    /**
     * @brief Constructs a FlattenDataCellParameter.
     */
    explicit FlattenDataCellParameter();

    /**
     * @brief Loads parameters from JSON configuration.
     *
     * @param json JSON configuration object.
     */
    void
    FromJson(const JsonType& json) override;

    /**
     * @brief Exports parameters to JSON format.
     *
     * @return JSON object containing the parameter values.
     */
    JsonType
    ToJson() const override;

    /**
     * @brief Checks compatibility with another parameter.
     *
     * @param other Another parameter to compare with.
     * @return True if parameters are compatible, false otherwise.
     */
    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override;
};
}  // namespace vsag