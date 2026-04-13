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

/// @file vector_transformer.h
/// @brief Base class for vector transformation operations.

#pragma once

#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/pointer_define.h"

namespace vsag {

class Allocator;
DEFINE_POINTER(VectorTransformer);
DEFINE_POINTER(TransformerMeta);

/// @brief Enumeration of vector transformer types.
enum class VectorTransformerType { NONE, PCA, RANDOM_ORTHOGONAL, FHT, RESIDUAL, NORMALIZE, MRLE };

/// @brief Base metadata structure for transformer-specific information.
struct TransformerMeta {
    /// @brief Encodes metadata into a byte buffer.
    /// @param[out] code Buffer to store encoded metadata.
    virtual void
    EncodeMeta(uint8_t* code) {
        return;
    };

    /// @brief Decodes metadata from a byte buffer.
    /// @param[in] code Buffer containing encoded metadata.
    /// @param[in] align_size Alignment size for decoding.
    virtual void
    DecodeMeta(uint8_t* code, uint32_t align_size) {
        return;
    };
};

/// @brief Abstract base class for vector transformation operations.
///
/// This class provides the interface for transforming vectors between different
/// dimensional spaces, with support for serialization and metadata handling.
class VectorTransformer {
public:
    /// @brief Constructs a vector transformer with specified dimensions.
    /// @param[in] allocator Memory allocator for internal operations.
    /// @param[in] input_dim Input vector dimension.
    /// @param[in] output_dim Output vector dimension.
    explicit VectorTransformer(Allocator* allocator, int64_t input_dim, int64_t output_dim);

    /// @brief Constructs a vector transformer with equal input and output dimensions.
    /// @param[in] allocator Memory allocator for internal operations.
    /// @param[in] input_dim Vector dimension (both input and output).
    explicit VectorTransformer(Allocator* allocator, int64_t input_dim)
        : VectorTransformer(allocator, input_dim, input_dim) {
    }

    virtual ~VectorTransformer() = default;

    /// @brief Transforms an input vector to output vector.
    /// @param[in] input_vec Input vector data.
    /// @param[out] output_vec Output vector buffer.
    /// @return Pointer to transformer metadata, or nullptr if no metadata.
    virtual TransformerMetaPtr
    Transform(const float* input_vec, float* output_vec) const {
        return nullptr;
    };

    /// @brief Serializes the transformer state to a stream.
    /// @param[in] writer Stream writer for serialization.
    virtual void
    Serialize(StreamWriter& writer) const = 0;

    /// @brief Deserializes the transformer state from a stream.
    /// @param[in] reader Stream reader for deserialization.
    virtual void
    Deserialize(StreamReader& reader) = 0;

    /// @brief Recovers the original distance from transformed distance.
    /// @param[in] dist Transformed distance value.
    /// @param[in] meta1 Metadata for first vector.
    /// @param[in] meta2 Metadata for second vector.
    /// @return Recovered distance value.
    virtual float
    RecoveryDistance(float dist, const uint8_t* meta1, const uint8_t* meta2) const {
        return dist;
    };

    /// @brief Gets the metadata size in bytes.
    /// @return Original metadata size.
    virtual uint32_t
    GetMetaSize() const {
        return 0;
    };

    /// @brief Gets the aligned metadata size in bytes.
    /// @param[in] align_size Alignment size.
    /// @return Aligned metadata size.
    virtual uint32_t
    GetMetaSize(uint32_t align_size) const {
        return 0;
    };

    /// @brief Gets the alignment size for metadata.
    /// @return Alignment size in bytes.
    virtual uint32_t
    GetAlignSize() const {
        return 0;
    }

public:
    /// @brief Trains the transformer with sample data.
    /// @param[in] data Training data array.
    /// @param[in] count Number of vectors in training data.
    virtual void
    Train(const float* data, uint64_t count){};

    /// @brief Performs inverse transformation from output to input space.
    /// @param[in] input_vec Transformed vector.
    /// @param[out] output_vec Buffer for original vector.
    virtual void
    InverseTransform(const float* input_vec, float* output_vec) const;

    /// @brief Gets the input vector dimension.
    /// @return Input dimension.
    [[nodiscard]] int64_t
    GetInputDim() const {
        return this->input_dim_;
    }

    /// @brief Gets the output vector dimension.
    /// @return Output dimension.
    [[nodiscard]] int64_t
    GetOutputDim() const {
        return this->output_dim_;
    }

    /// @brief Gets the transformer type.
    /// @return Transformer type enumeration value.
    [[nodiscard]] VectorTransformerType
    GetType() const {
        return this->type_;
    }

    /// @brief Gets the extra code size in bytes.
    /// @return Extra code size.
    [[nodiscard]] uint32_t
    GetExtraCodeSize() const {
        return this->extra_code_size_;
    }

protected:
    uint32_t extra_code_size_{0};          /// Extra code size (e.g., sizeof(float))
    int64_t input_dim_{0};                 /// Input vector dimension
    int64_t output_dim_{0};                /// Output vector dimension
    Allocator* const allocator_{nullptr};  /// Memory allocator

    VectorTransformerType type_{VectorTransformerType::NONE};  /// Transformer type
};

}  // namespace vsag