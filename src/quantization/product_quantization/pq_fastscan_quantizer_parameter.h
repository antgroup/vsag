
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
 * @file pq_fastscan_quantizer_parameter.h
 * @brief Parameter configuration for PQFastScanQuantizer.
 */

#pragma once

#include "quantization/quantizer_parameter.h"
#include "utils/pointer_define.h"
namespace vsag {
DEFINE_POINTER2(PQFastScanQuantizerParam, PQFastScanQuantizerParameter);

/**
 * @brief Parameter class for PQFastScanQuantizer configuration.
 *
 * Stores configuration parameters for PQ FastScan quantization,
 * which uses 4-bit codes for SIMD-optimized distance computation.
 */
class PQFastScanQuantizerParameter : public QuantizerParameter {
public:
    /**
     * @brief Constructs a PQFastScanQuantizerParameter with default values.
     */
    PQFastScanQuantizerParameter();

    ~PQFastScanQuantizerParameter() override = default;

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
    int64_t pq_dim_{1};  /// Number of subspaces for product quantization
};
}  // namespace vsag
