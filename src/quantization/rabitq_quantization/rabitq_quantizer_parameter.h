
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
 * @file rabitq_quantizer_parameter.h
 * @brief Parameter class for RaBitQ quantizer configuration.
 */

#pragma once

#include "quantization/quantizer_parameter.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER2(RaBitQuantizerParam, RaBitQuantizerParameter);

/**
 * @brief Parameter class for RaBitQ quantizer.
 *
 * Holds configuration for RaBitQ quantization including PCA dimension,
 * bits per dimension, and Fast Hadamard Transform options.
 */
class RaBitQuantizerParameter : public QuantizerParameter {
public:
    RaBitQuantizerParameter();

    ~RaBitQuantizerParameter() override = default;

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

public:
    uint64_t pca_dim_{0};                  ///< PCA dimension for MRQ.
    uint64_t num_bits_per_dim_query_{32};  ///< Bits per dimension for query quantization.
    uint64_t num_bits_per_dim_base_{1};    ///< Bits per dimension for base quantization.
    bool use_fht_{false};                  ///< Whether to use Fast Hadamard Transform.
};
}  // namespace vsag
