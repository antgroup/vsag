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

/// @file random_orthogonal_transformer.h
/// @brief Random orthogonal matrix transformer for vector transformation.

#pragma once

#include "vector_transformer.h"

namespace vsag {

/// @brief Metadata structure for random orthogonal matrix transformation.
struct ROMMeta : public TransformerMeta {};

/// @brief Random orthogonal matrix transformer.
///
/// This class implements vector transformation using a randomly generated
/// orthogonal matrix, which preserves distances between vectors.
class RandomOrthogonalMatrix : public VectorTransformer {
public:
    /// @brief Constructs a random orthogonal matrix transformer.
    /// @param[in] allocator Memory allocator for internal operations.
    /// @param[in] dim Vector dimension.
    /// @param[in] retries Maximum number of retries for matrix generation.
    explicit RandomOrthogonalMatrix(Allocator* allocator,
                                    int64_t dim,
                                    uint64_t retries = MAX_RETRIES);

    ~RandomOrthogonalMatrix() override = default;

    /// @brief Transforms a vector using the orthogonal matrix.
    /// @param[in] original_vec Original vector data.
    /// @param[out] transformed_vec Buffer for transformed vector.
    /// @return Pointer to transformer metadata.
    TransformerMetaPtr
    Transform(const float* original_vec, float* transformed_vec) const override;

    /// @brief Performs inverse transformation.
    /// @param[in] transformed_vec Transformed vector.
    /// @param[out] original_vec Buffer for original vector.
    void
    InverseTransform(const float* transformed_vec, float* original_vec) const override;

    /// @brief Serializes the transformer state.
    /// @param[in] writer Stream writer for serialization.
    void
    Serialize(StreamWriter& writer) const override;

    /// @brief Deserializes the transformer state.
    /// @param[in] reader Stream reader for deserialization.
    void
    Deserialize(StreamReader& reader) override;

    /// @brief Trains the transformer (no-op for random orthogonal matrix).
    /// @param[in] data Training data array.
    /// @param[in] count Number of vectors in training data.
    void
    Train(const float* data, uint64_t count) override;

public:
    /// @brief Copies the orthogonal matrix to output buffer.
    /// @param[out] out_matrix Buffer to store the matrix.
    void
    CopyOrthogonalMatrix(float* out_matrix) const;

    /// @brief Generates a random orthogonal matrix.
    /// @return True if generation succeeded, false otherwise.
    bool
    GenerateRandomOrthogonalMatrix();

    /// @brief Generates a random orthogonal matrix with retry logic.
    void
    GenerateRandomOrthogonalMatrixWithRetry();

    /// @brief Computes the determinant of the orthogonal matrix.
    /// @return Determinant value.
    double
    ComputeDeterminant() const;

public:
    static constexpr uint64_t MAX_RETRIES = 3;  /// Maximum retry count for matrix generation

private:
    Vector<float> orthogonal_matrix_;     /// Orthogonal transformation matrix
    const uint64_t generate_retries_{0};  /// Number of retries for generation
};

}  // namespace vsag