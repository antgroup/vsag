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

/// @file vector_transformer_parameter.h
/// @brief Parameter configuration for vector transformers.

#pragma once

#include "index_common_param.h"
#include "quantization/quantizer_parameter.h"

namespace vsag {

/// @brief Parameter class for vector transformer configuration.
///
/// This class holds configuration parameters for vector transformers,
/// including dimensions for PCA and MRLE transformations.
class VectorTransformerParameter : public Parameter {
public:
    VectorTransformerParameter() = default;

    ~VectorTransformerParameter() override = default;

    /// @brief Loads parameters from JSON configuration.
    /// @param[in] json JSON object containing parameter values.
    void
    FromJson(const JsonType& json) override;

    /// @brief Converts parameters to JSON format.
    /// @return JSON object representing the parameters.
    JsonType
    ToJson() const override;

    /// @brief Checks compatibility with another parameter object.
    /// @param[in] other Another parameter object to compare with.
    /// @return True if parameters are compatible, false otherwise.
    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override;

public:
    uint32_t input_dim_{0};  /// Input vector dimension
    uint32_t pca_dim_{0};    /// Output dimension for PCA transformation
    uint32_t mrle_dim_{0};   /// Output dimension for MRLE transformation
};

}  // namespace vsag