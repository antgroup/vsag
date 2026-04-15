
/**
 * @file bf16_quantizer.h
 * @brief BF16 (bfloat16) scalar quantizer implementation.
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

#include "bf16_quantizer_parameter.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"

namespace vsag {

/***
 * @brief BF16 Quantizer stores vectors in 16-bit bfloat16 floating-point format.
 *
 * code layout:
 * +------------------------+
 * | bf16-code              |
 * | [dim * 2B]             |
 * +------------------------+
 *
 * - bf16-code: bfloat16 (Brain Float) values (required)
 * - 1 sign bit, 8 exponent bits, 7 mantissa bits
 * - Same exponent range as FP32, less precision
 */
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class BF16Quantizer : public Quantizer<BF16Quantizer<metric>> {
public:
    /**
     * @brief Constructs a BF16Quantizer with dimension and allocator.
     * @param dim Vector dimension.
     * @param allocator Memory allocator for buffer management.
     */
    explicit BF16Quantizer(int dim, Allocator* allocator);

    /**
     * @brief Constructs a BF16Quantizer from parameter object.
     * @param param BF16 quantizer parameter pointer.
     * @param common_param Common index parameters.
     */
    explicit BF16Quantizer(const BF16QuantizerParamPtr& param,
                           const IndexCommonParam& common_param);

    /**
     * @brief Constructs a BF16Quantizer from generic quantizer parameter.
     * @param param Generic quantizer parameter pointer.
     * @param common_param Common index parameters.
     */
    explicit BF16Quantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    /**
     * @brief Trains the quantizer with input data (no-op for BF16).
     * @param data Training data array.
     * @param count Number of vectors in training data.
     * @return True (BF16 requires no training).
     */
    bool
    TrainImpl(const DataType* data, uint64_t count);

    /**
     * @brief Encodes a single vector into BF16 format.
     * @param data Input vector data.
     * @param codes Output BF16 codes buffer.
     * @return True if encoding succeeded.
     */
    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes) const;

    /**
     * @brief Decodes BF16 codes back to floating-point vector.
     * @param codes Input BF16 codes.
     * @param data Output decoded vector data.
     * @return True if decoding succeeded.
     */
    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data);

    /**
     * @brief Computes distance between two BF16 vectors.
     * @param codes1 First BF16 vector codes.
     * @param codes2 Second BF16 vector codes.
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
    ProcessQueryImpl(const DataType* query, Computer<BF16Quantizer>& computer) const;

    /**
     * @brief Computes distance from query to a BF16 vector.
     * @param computer Computer object containing query state.
     * @param codes BF16 vector codes.
     * @param dists Output distance array.
     */
    void
    ComputeDistImpl(Computer<BF16Quantizer>& computer, const uint8_t* codes, float* dists) const;

    /**
     * @brief Serializes quantizer state to stream writer (no-op for BF16).
     * @param writer Stream writer for output.
     */
    void
    SerializeImpl(StreamWriter& writer){};

    /**
     * @brief Deserializes quantizer state from stream reader (no-op for BF16).
     * @param reader Stream reader for input.
     */
    void
    DeserializeImpl(StreamReader& reader){};

    /**
     * @brief Returns the quantizer type name.
     * @return String identifier (BF16).
     */
    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_BF16;
    }
};

}  // namespace vsag
