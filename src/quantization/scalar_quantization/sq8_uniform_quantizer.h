
/**
 * @file sq8_uniform_quantizer.h
 * @brief 8-bit uniform scalar quantizer implementation.
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
#include "sq8_uniform_quantizer_parameter.h"

namespace vsag {

/**
 * @brief 8-bit uniform scalar quantizer.
 *
 * This quantizer uses uniform quantization with 8 bits per dimension,
 * providing a good balance between compression ratio and precision.
 *
 * @tparam metric Distance metric type (default: L2SQR).
 */
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class SQ8UniformQuantizer : public Quantizer<SQ8UniformQuantizer<metric>> {
public:
    using norm_type = uint64_t;  /// Type for norm value storage
    using sum_type = float;      /// Type for sum value storage

    /**
     * @brief Constructs a SQ8UniformQuantizer with dimension and allocator.
     * @param dim Vector dimension.
     * @param allocator Memory allocator for buffer management.
     */
    explicit SQ8UniformQuantizer(int dim, Allocator* allocator);

    /**
     * @brief Constructs a SQ8UniformQuantizer from parameter object.
     * @param param SQ8 uniform quantizer parameter pointer.
     * @param common_param Common index parameters.
     */
    SQ8UniformQuantizer(const SQ8UniformQuantizerParamPtr& param,
                        const IndexCommonParam& common_param);

    /**
     * @brief Constructs a SQ8UniformQuantizer from generic quantizer parameter.
     * @param param Generic quantizer parameter pointer.
     * @param common_param Common index parameters.
     */
    SQ8UniformQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    ~SQ8UniformQuantizer() = default;

    /**
     * @brief Trains the quantizer with input data.
     * @param data Training data array.
     * @param count Number of vectors in training data.
     * @return True if training succeeded.
     */
    bool
    TrainImpl(const DataType* data, uint64_t count);

    /**
     * @brief Encodes a single vector into quantized codes.
     * @param data Input vector data.
     * @param codes Output quantized codes buffer.
     * @return True if encoding succeeded.
     */
    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes) const;

    /**
     * @brief Decodes quantized codes back to floating-point vector.
     * @param codes Input quantized codes.
     * @param data Output decoded vector data.
     * @return True if decoding succeeded.
     */
    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data);

    /**
     * @brief Computes distance between two quantized vectors.
     * @param codes1 First quantized vector codes.
     * @param codes2 Second quantized vector codes.
     * @return Computed distance value.
     */
    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const;

    /**
     * @brief Prepares query vector for distance computation.
     * @param query Query vector data.
     * @param computer Computer object for storing query state.
     */
    void
    ProcessQueryImpl(const DataType* query, Computer<SQ8UniformQuantizer>& computer) const;

    /**
     * @brief Computes distance from query to a quantized vector.
     * @param computer Computer object containing query state.
     * @param codes Quantized vector codes.
     * @param dists Output distance array.
     */
    void
    ComputeDistImpl(Computer<SQ8UniformQuantizer>& computer,
                    const uint8_t* codes,
                    float* dists) const;

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
     * @return String identifier (SQ8_UNIFORM).
     */
    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_SQ8_UNIFORM;
    }

private:
    DataType lower_bound_{0};  /// Lower bound for quantization
    DataType diff_{0};         /// Difference (upper - lower) for quantization

    /***
     * code layout: sq-code(fixed) + norm(opt) + sum(opt)
     * for L2 and COSINE, norm is needed for fast computation
     * for IP and COSINE, sum is needed for restoring original distance
     */
    uint64_t offset_code_{0};  /// Offset to code section
    uint64_t offset_norm_{0};  /// Offset to norm section
    uint64_t offset_sum_{0};   /// Offset to sum section
    float scalar_rate_{0.0F};  /// Scalar rate for uniform quantization
};

}  // namespace vsag
