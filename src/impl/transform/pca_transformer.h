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

/// @file pca_transformer.h
/// @brief PCA (Principal Component Analysis) transformer for dimensionality reduction.

#pragma once

#include "vector_transformer.h"

namespace vsag {

/// @brief Metadata structure for PCA transformation.
struct PCAMeta : public TransformerMeta {
    float residual_norm;  /// Residual norm after transformation

    /// @brief Encodes metadata into a byte buffer.
    /// @param[out] code Buffer to store encoded metadata.
    void
    EncodeMeta(uint8_t* code) override {
        *((float*)code) = residual_norm;
    }

    /// @brief Decodes metadata from a byte buffer.
    /// @param[in] code Buffer containing encoded metadata.
    /// @param[in] align_size Alignment size (unused).
    void
    DecodeMeta(uint8_t* code, uint32_t align_size) override {
        residual_norm = *((float*)code);
    }

    /// @brief Gets the metadata size in bytes.
    /// @return Size of float type.
    static uint32_t
    GetMetaSize() {
        return sizeof(float);
    }

    /// @brief Gets the aligned metadata size in bytes.
    /// @param[in] align_size Alignment size.
    /// @return Maximum of float size and alignment size.
    static uint32_t
    GetMetaSize(uint32_t align_size) {
        return std::max(static_cast<uint32_t>(sizeof(float)), align_size);
    }

    /// @brief Gets the alignment size for metadata.
    /// @return Size of float type.
    static uint32_t
    GetAlignSize() {
        return sizeof(float);
    }
};

/// @brief PCA transformer for dimensionality reduction.
///
/// This class implements Principal Component Analysis for reducing vector
/// dimensionality while preserving maximum variance.
class PCATransformer : public VectorTransformer {
public:
    /// @brief Constructs a PCA transformer.
    /// @param[in] allocator Memory allocator for internal operations.
    /// @param[in] input_dim Input vector dimension.
    /// @param[in] output_dim Output vector dimension (reduced dimension).
    explicit PCATransformer(Allocator* allocator, int64_t input_dim, int64_t output_dim);

    /// @brief Trains the PCA transformer with sample data.
    /// @param[in] data Training data array.
    /// @param[in] count Number of vectors in training data.
    void
    Train(const float* data, uint64_t count) override;

    /// @brief Serializes the transformer state.
    /// @param[in] writer Stream writer for serialization.
    void
    Serialize(StreamWriter& writer) const override;

    /// @brief Deserializes the transformer state.
    /// @param[in] reader Stream reader for deserialization.
    void
    Deserialize(StreamReader& reader) override;

    /// @brief Transforms a vector using PCA.
    /// @param[in] input_vec Input vector data.
    /// @param[out] output_vec Buffer for transformed vector.
    /// @return Pointer to transformer metadata.
    TransformerMetaPtr
    Transform(const float* input_vec, float* output_vec) const override;

    /// @brief Performs inverse transformation (reconstruction).
    /// @param[in] input_vec Transformed vector.
    /// @param[out] output_vec Buffer for reconstructed vector.
    void
    InverseTransform(const float* input_vec, float* output_vec) const override;

    /// @brief Recovers the original distance from transformed distance.
    /// @param[in] dist Transformed distance value.
    /// @param[in] meta1 Metadata for first vector.
    /// @param[in] meta2 Metadata for second vector.
    /// @return Recovered distance value.
    float
    RecoveryDistance(float dist, const uint8_t* meta1, const uint8_t* meta2) const override {
        return dist;
    };

    /// @brief Gets the metadata size in bytes.
    /// @return Size of PCA metadata.
    uint32_t
    GetMetaSize() const override {
        return PCAMeta::GetMetaSize();
    }

    /// @brief Gets the aligned metadata size in bytes.
    /// @param[in] align_size Alignment size.
    /// @return Aligned metadata size.
    uint32_t
    GetMetaSize(uint32_t align_size) const override {
        return PCAMeta::GetMetaSize(align_size);
    }

    /// @brief Gets the alignment size for metadata.
    /// @return Alignment size in bytes.
    uint32_t
    GetAlignSize() const override {
        return PCAMeta::GetAlignSize();
    }

public:
    /// @brief Copies the PCA matrix for testing purposes.
    /// @param[out] out_pca_matrix Buffer to store the PCA matrix.
    void
    CopyPCAMatrixForTest(float* out_pca_matrix) const;

    /// @brief Copies the mean vector for testing purposes.
    /// @param[out] out_mean Buffer to store the mean vector.
    void
    CopyMeanForTest(float* out_mean) const;

    /// @brief Sets the mean vector for testing purposes.
    /// @param[in] input_mean Input mean vector.
    void
    SetMeanForTest(const float* input_mean);

    /// @brief Sets the PCA matrix for testing purposes.
    /// @param[in] input_pca_matrix Input PCA matrix.
    void
    SetPCAMatrixForTest(const float* input_pca_matrix);

    /// @brief Computes the column-wise mean of training data.
    /// @param[in] data Training data array.
    /// @param[in] count Number of vectors in training data.
    void
    ComputeColumnMean(const float* data, uint64_t count);

    /// @brief Computes the covariance matrix of centralized data.
    /// @param[in] centralized_data Centralized data array.
    /// @param[in] count Number of vectors in data.
    /// @param[out] covariance_matrix Buffer for covariance matrix.
    void
    ComputeCovarianceMatrix(const float* centralized_data,
                            uint64_t count,
                            float* covariance_matrix) const;

    /// @brief Performs eigenvalue decomposition on covariance matrix.
    /// @param[in] covariance_matrix Covariance matrix data.
    /// @return True if decomposition succeeded, false otherwise.
    bool
    PerformEigenDecomposition(const float* covariance_matrix);

    /// @brief Centralizes data by subtracting the mean.
    /// @param[in] original_data Original data array.
    /// @param[out] centralized_data Buffer for centralized data.
    void
    CentralizeData(const float* original_data, float* centralized_data) const;

private:
    Vector<float> pca_matrix_;  /// PCA transformation matrix [input_dim_ * output_dim_]
    Vector<float> mean_;        /// Column mean vector [input_dim_ * 1]
};

}  // namespace vsag