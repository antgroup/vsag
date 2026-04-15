/**
 * @file attribute_inverted_interface_parameter.h
 * @brief Parameter class for attribute inverted index interface configuration.
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
#include "quantization/quantizer_parameter.h"

namespace vsag {

/**
 * @brief Parameter class for attribute inverted index interface configuration.
 *
 * This class provides configuration parameters for attribute-based inverted
 * index storage with bucket support.
 */
class AttributeInvertedInterfaceParameter : public Parameter {
public:
    /**
     * @brief Constructs an AttributeInvertedInterfaceParameter.
     */
    explicit AttributeInvertedInterfaceParameter() = default;

    /**
     * @brief Default destructor.
     */
    ~AttributeInvertedInterfaceParameter() override = default;

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
    bool has_buckets_{false};  ///< Whether the inverted index uses bucket organization
};

using AttributeInvertedInterfaceParamPtr = std::shared_ptr<AttributeInvertedInterfaceParameter>;

}  // namespace vsag