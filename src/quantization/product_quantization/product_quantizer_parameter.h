
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
 * @file product_quantizer_parameter.h
 * @brief Parameter configuration for ProductQuantizer.
 */

#pragma once

#include "quantization/quantizer_parameter.h"
#include "utils/pointer_define.h"
namespace vsag {
DEFINE_POINTER2(ProductQuantizerParam, ProductQuantizerParameter);

/**
 * @brief Parameter class for ProductQuantizer configuration.
 *
 * Stores configuration parameters for product quantization including
 * the number of subspaces and bits per code.
 */
class ProductQuantizerParameter : public QuantizerParameter {
public:
    /**
     * @brief Constructs a ProductQuantizerParameter with default values.
     */
    ProductQuantizerParameter();

    ~ProductQuantizerParameter() override = default;

    /**
     * @brief Loads parameters from JSON configuration.
     * @param json JSON object containing parameter values.
     */
    void
    FromJson(const JsonType& json) override;

    /**
     * @brief Exports parameters to JSON format.
     * @return JSON object containing current parameter values.
     */
    JsonType
    ToJson() const override;

    /**
     * @brief Checks compatibility with another parameter set.
     * @param other Another parameter object to compare against.
     * @return true if parameters are compatible, false otherwise.
     */
    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override;

public:
    int64_t pq_dim_{1};   /// Number of subspaces for product quantization
    int64_t pq_bits_{8};  /// Bits per PQ code (typically 8)
};
}  // namespace vsag
