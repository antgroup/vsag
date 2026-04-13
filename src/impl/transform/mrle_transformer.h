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

/// @file mrle_transformer.h
/// @brief MRLE (Mean Residual Learning Encoding) transformer.

#pragma once

#include "metric_type.h"
#include "simd/normalize.h"
#include "vector_transformer.h"
#include "vsag_exception.h"

namespace vsag {

/// @brief Metadata structure for MRLE transformation.
struct MRLETMeta : public TransformerMeta {};

/// @brief MRLE (Mean Residual Learning Encoding) transformer.
///
/// This template class implements MRLE transformation with optional cosine
/// normalization support for different metric types.
/// @tparam metric The metric type for transformation (default: L2SQR).
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class MRLETransformer : public VectorTransformer {
public:
    /// @brief Constructs an MRLE transformer.
    /// @param[in] allocator Memory allocator for internal operations.
    /// @param[in] input_dim Input vector dimension.
    /// @param[in] output_dim Output vector dimension.
    explicit MRLETransformer(Allocator* allocator, int64_t input_dim, int64_t output_dim)
        : VectorTransformer(allocator, input_dim, output_dim) {
        this->type_ = VectorTransformerType::MRLE;
    }

    ~MRLETransformer() override = default;

    /// @brief Transforms a vector using MRLE.
    /// @param[in] original_vec Original vector data.
    /// @param[out] transformed_vec Buffer for transformed vector.
    /// @return Pointer to transformer metadata.
    TransformerMetaPtr
    Transform(const float* original_vec, float* transformed_vec) const override {
        auto meta = std::make_shared<MRLETMeta>();
        memcpy(transformed_vec, original_vec, this->output_dim_ * sizeof(float));
        if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
            Normalize(transformed_vec, transformed_vec, this->output_dim_);
        }
        return meta;
    }

    /// @brief Performs inverse transformation (not implemented).
    /// @param[in] transformed_vec Transformed vector.
    /// @param[out] original_vec Buffer for original vector.
    /// @throws VsagException Always throws as not implemented.
    void
    InverseTransform(const float* transformed_vec, float* original_vec) const override {
        throw VsagException(ErrorType::INTERNAL_ERROR, "InverseTransform not implement");
    }

    /// @brief Serializes the transformer state (no-op for MRLE).
    /// @param[in] writer Stream writer for serialization.
    void
    Serialize(StreamWriter& writer) const override{};

    /// @brief Deserializes the transformer state (no-op for MRLE).
    /// @param[in] reader Stream reader for deserialization.
    void
    Deserialize(StreamReader& reader) override{};

    /// @brief Trains the transformer (no-op for MRLE).
    /// @param[in] data Training data array.
    /// @param[in] count Number of vectors in training data.
    void
    Train(const float* data, uint64_t count) override{};
};

}  // namespace vsag