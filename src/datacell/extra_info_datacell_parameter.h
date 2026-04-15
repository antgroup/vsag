/**
 * @file extra_info_datacell_parameter.h
 * @brief Parameter class for extra info data cell configuration.
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

#include "io/io_parameter.h"
#include "parameter.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER2(ExtraInfoDataCellParam, ExtraInfoDataCellParameter);

/**
 * @brief Parameter class for extra info data cell configuration.
 *
 * This class provides configuration parameters for storing additional metadata
 * associated with vectors, with IO parameter support.
 */
class ExtraInfoDataCellParameter : public Parameter {
public:
    /**
     * @brief Constructs an ExtraInfoDataCellParameter.
     */
    explicit ExtraInfoDataCellParameter();

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
    CheckCompatibility(const ParamPtr& other) const override;

public:
    IOParamPtr io_parameter{nullptr};  ///< IO configuration for extra info storage
};
}  // namespace vsag