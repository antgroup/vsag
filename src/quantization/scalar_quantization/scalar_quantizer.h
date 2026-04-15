
/**
 * @file scalar_quantizer.h
 * @brief Scalar quantizer for 4-bit and 8-bit vector quantization.
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

#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"
#include "scalar_quantizer_parameter.h"

namespace vsag {

/***
 * @brief Scalar Quantizer stores vectors in 4-bit or 8-bit quantized format.
 *
 * code layout (SQ8):
 * +------------------------+
 * | sq8-code               |
 * | [dim * 1B]             |
 * +------------------------+
 *
 * code layout (SQ4):
 * +------------------------+
 * | sq4-code               |
 * | [dim * 0.5B, packed]   |
 * +------------------------+
 *
 * - sq-code: quantized values per dimension (required)
 * - SQ8: 8 bits per dimension, range [0, 255]
 * - SQ4: 4 bits per dimension, packed 2 values per byte, range [0, 15]
 */
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR, int bit = 8>
class ScalarQuantizer : public Quantizer<ScalarQuantizer<metric, bit>> {
public:
    /**
     * @brief Constructs a ScalarQuantizer with dimension and allocator.
     * @param dim Vector dimension.
     * @param allocator Memory allocator for buffer management.
     */
    explicit ScalarQuantizer(int dim, Allocator* allocator);

    /**
     * @brief Constructs a ScalarQuantizer from parameter object.
     * @param param Scalar quantizer parameter shared pointer.
     * @param common_param Common index parameters.
     */
    explicit ScalarQuantizer(const std::shared_ptr<ScalarQuantizerParameter<bit>>& param,
                             const IndexCommonParam& common_param);

    /**
     * @brief Constructs a ScalarQuantizer from generic quantizer parameter.
     * @param param Generic quantizer parameter pointer.
     * @param common_param Common index parameters.
     */
    explicit ScalarQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    /**
     * @brief Trains the quantizer with input data.
     * @param data Training data array.
     * @param count Number of vectors in training data.
     * @return True if training succeeded.
     */
    bool
    TrainImpl(const float* data, uint64_t count);

    /**
     * @brief Encodes a single vector into quantized codes.
     * @param data Input vector data.
     * @param codes Output quantized codes buffer.
     * @return True if encoding succeeded.
     */
    bool
    EncodeOneImpl(const float* data, uint8_t* codes) const;

    /**
     * @brief Decodes quantized codes back to floating-point vector.
     * @param codes Input quantized codes.
     * @param data Output decoded vector data.
     * @return True if decoding succeeded.
     */
    bool
    DecodeOneImpl(const uint8_t* codes, float* data);

    /**
     * @brief Computes distance between two quantized vectors.
     * @param codes1 First quantized vector codes.
     * @param codes2 Second quantized vector codes.
     * @return Computed distance value.
     */
    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2);

    /**
     * @brief Prepares query vector for distance computation.
     * @param query Query vector data.
     * @param computer Computer object for storing query state.
     */
    void
    ProcessQueryImpl(const float* query, Computer<ScalarQuantizer<metric, bit>>& computer) const;

    /**
     * @brief Computes distance from query to a quantized vector.
     * @param computer Computer object containing query state.
     * @param codes Quantized vector codes.
     * @param dists Output distance array.
     */
    void
    ComputeDistImpl(Computer<ScalarQuantizer>& computer, const uint8_t* codes, float* dists) const;

    /**
     * @brief Serializes quantizer state to stream writer.
     * @param writer Stream writer for output.
     */
    void
    SerializeImpl(StreamWriter& writer);

    /**
     * @brief Deserializes quantizer state from stream reader.
     * @param reader Stream reader for input.
     */
    void
    DeserializeImpl(StreamReader& reader);

    /**
     * @brief Returns the quantizer type name.
     * @return String identifier (SQ8 or SQ4).
     */
    [[nodiscard]] std::string
    NameImpl() const {
        static_assert(bit == 4 || bit == 8, "bit must be 4 or 8");
        if constexpr (bit == 8) {
            return QUANTIZATION_TYPE_VALUE_SQ8;
        } else if constexpr (bit == 4) {
            return QUANTIZATION_TYPE_VALUE_SQ4;
        }
    }

public:
    static constexpr int BIT_PER_DIM = bit;                    /// Bits per dimension
    static constexpr int MAX_CODE_PER_DIM = 1 << BIT_PER_DIM;  /// Maximum code value (2^bits)

private:
    std::vector<float> lower_bound_{};  /// Lower bound per dimension
    std::vector<float> diff_{};         /// Difference (upper - lower) per dimension
};

/// Type alias for 8-bit scalar quantizer
template <MetricType metric>
using SQ8Quantizer = ScalarQuantizer<metric, 8>;

/// Type alias for 4-bit scalar quantizer
template <MetricType metric>
using SQ4Quantizer = ScalarQuantizer<metric, 4>;

}  // namespace vsag
