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

/// @file fht_kac_rotate_transformer.h
/// @brief FHT (Fast Hadamard Transform) with KAC rotation transformer.

#include <cstdint>
#include <cstring>

#include "vector_transformer.h"

namespace vsag {

/// @brief Metadata structure for FHT transformation.
struct FHTMeta : public TransformerMeta {};

/// @brief FHT (Fast Hadamard Transform) with KAC rotation transformer.
///
/// This class implements vector transformation using Fast Hadamard Transform
/// combined with KAC (Kac's) rotation for dimensionality reduction.
class FhtKacRotator : public VectorTransformer {
public:
    /// @brief Constructs an FHT KAC rotator.
    /// @param[in] allocator Memory allocator for internal operations.
    /// @param[in] dim Vector dimension.
    explicit FhtKacRotator(Allocator* allocator, int64_t dim);

    ~FhtKacRotator() override = default;

    /// @brief Transforms a vector using FHT with KAC rotation.
    /// @param[in] data Input vector data.
    /// @param[out] rotated_vec Buffer for rotated vector.
    /// @return Pointer to transformer metadata.
    TransformerMetaPtr
    Transform(const float* data, float* rotated_vec) const override;

    /// @brief Performs inverse transformation.
    /// @param[in] data Transformed vector.
    /// @param[out] rotated_vec Buffer for original vector.
    void
    InverseTransform(const float* data, float* rotated_vec) const override;

    /// @brief Serializes the transformer state.
    /// @param[in] writer Stream writer for serialization.
    void
    Serialize(StreamWriter& writer) const override;

    /// @brief Deserializes the transformer state.
    /// @param[in] reader Stream reader for deserialization.
    void
    Deserialize(StreamReader& reader) override;

    /// @brief Trains the transformer with sample data.
    /// @param[in] data Training data array.
    /// @param[in] count Number of vectors in training data.
    void
    Train(const float* data, uint64_t count) override;

    /// @brief Trains the transformer (internal initialization).
    void
    Train();

    /// @brief Copies the flip pattern to output buffer.
    /// @param[out] out_flip Buffer to store flip pattern.
    void
    CopyFlip(uint8_t* out_flip) const;

    static constexpr uint64_t BYTE_LEN = 8;  /// Number of bits in a byte
    static constexpr int ROUND = 4;          /// Number of rotation rounds

private:
    uint64_t flip_offset_{0};    /// Offset for flip pattern
    std::vector<uint8_t> flip_;  /// Flip pattern data
    uint64_t trunc_dim_{0};      /// Truncated dimension
    float fac_{0};               /// Scaling factor
};

}  //namespace vsag