
/**
 * @file sq4_uniform_quantizer.h
 * @brief 4-bit uniform scalar quantizer implementation.
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

#include "inner_string_params.h"
#include "metric_type.h"
#include "quantization/quantizer.h"
#include "utils/pointer_define.h"

namespace vsag {

DEFINE_POINTER2(SQ4UniformQuantizerParam, SQ4UniformQuantizerParameter);
DEFINE_POINTER2(QuantizerParam, QuantizerParameter);
class IndexCommonParam;

/// Type alias for sum value storage
using sum_type = float;

/**
 * @brief 4-bit uniform scalar quantizer with optional truncation.
 *
 * This quantizer uses uniform quantization with 4 bits per dimension,
 * providing higher compression than 8-bit quantizers at the cost of precision.
 *
 * @tparam metric Distance metric type (default: L2SQR).
 */
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class SQ4UniformQuantizer : public Quantizer<SQ4UniformQuantizer<metric>> {
public:
    /**
     * @brief Constructs a SQ4UniformQuantizer with dimension and allocator.
     * @param dim Vector dimension.
     * @param allocator Memory allocator for buffer management.
     * @param trunc_rate Truncation rate for outlier handling (default: 0.05).
     */
    explicit SQ4UniformQuantizer(int dim, Allocator* allocator, float trunc_rate = 0.05F);

    /**
     * @brief Constructs a SQ4UniformQuantizer from parameter object.
     * @param param SQ4 uniform quantizer parameter pointer.
     * @param common_param Common index parameters.
     */
    explicit SQ4UniformQuantizer(const SQ4UniformQuantizerParamPtr& param,
                                 const IndexCommonParam& common_param);

    /**
     * @brief Constructs a SQ4UniformQuantizer from generic quantizer parameter.
     * @param param Generic quantizer parameter pointer.
     * @param common_param Common index parameters.
     */
    explicit SQ4UniformQuantizer(const QuantizerParamPtr& param,
                                 const IndexCommonParam& common_param);

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
    ProcessQueryImpl(const DataType* query, Computer<SQ4UniformQuantizer>& computer) const;

    /**
     * @brief Computes distance from query to a quantized vector.
     * @param computer Computer object containing query state.
     * @param codes Quantized vector codes.
     * @param dists Output distance array.
     */
    void
    ComputeDistImpl(Computer<SQ4UniformQuantizer>& computer,
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
     * @return String identifier (SQ4_UNIFORM).
     */
    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_SQ4_UNIFORM;
    }

public:
    /**
     * @brief Gets the lower bound and difference values.
     * @return Pair of (lower_bound, diff).
     */
    [[nodiscard]] std::pair<DataType, DataType>
    GetLBandDiff() const {
        return {lower_bound_, diff_};
    }

    /**
     * @brief Gets the sum value from encoded codes.
     * @param codes Quantized codes buffer.
     * @return Sum value stored in codes.
     */
    DataType
    GetCodesSum(const uint8_t* codes) const {
        return *(sum_type*)(codes + offset_codes_sum_);
    }

private:
    DataType lower_bound_{0};  /// Lower bound for quantization
    DataType diff_{0};         /// Difference (upper - lower) for quantization

    /***
     * code layout: sq-code(fixed) + norm(opt) + sum(opt)
     * for L2 and COSINE, norm is needed for fast computation
     * for IP and COSINE, sum is needed for restoring original distance
     */
    uint64_t offset_code_{0};       /// Offset to code section
    uint64_t offset_norm_{0};       /// Offset to norm section
    uint64_t offset_sum_{0};        /// Offset to sum section
    uint64_t offset_codes_sum_{0};  /// Offset to codes sum section

    float scalar_rate_{0.0F};  /// Scalar rate for uniform quantization
    float trunc_rate_{0.05F};  /// Truncation rate for outlier handling
};

}  // namespace vsag
