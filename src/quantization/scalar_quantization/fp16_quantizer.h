
/**
 * @file fp16_quantizer.h
 * @brief FP16 (half-precision) scalar quantizer implementation.
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

#include "fp16_quantizer_parameter.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"

namespace vsag {

/***
 * @brief FP16 Quantizer stores vectors in 16-bit half-precision floating-point format.
 *
 * code layout:
 * +------------------------+
 * | fp16-code              |
 * | [dim * 2B]             |
 * +------------------------+
 *
 * - fp16-code: IEEE 754 half-precision floats (required)
 * - 1 sign bit, 5 exponent bits, 10 mantissa bits
 */
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class FP16Quantizer : public Quantizer<FP16Quantizer<metric>> {
public:
    /**
     * @brief Constructs an FP16Quantizer with dimension and allocator.
     * @param dim Vector dimension.
     * @param allocator Memory allocator for buffer management.
     */
    explicit FP16Quantizer(int dim, Allocator* allocator);

    /**
     * @brief Constructs an FP16Quantizer from parameter object.
     * @param param FP16 quantizer parameter pointer.
     * @param common_param Common index parameters.
     */
    explicit FP16Quantizer(const FP16QuantizerParamPtr& param,
                           const IndexCommonParam& common_param);

    /**
     * @brief Constructs an FP16Quantizer from generic quantizer parameter.
     * @param param Generic quantizer parameter pointer.
     * @param common_param Common index parameters.
     */
    explicit FP16Quantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    /**
     * @brief Trains the quantizer with input data (no-op for FP16).
     * @param data Training data array.
     * @param count Number of vectors in training data.
     * @return True (FP16 requires no training).
     */
    bool
    TrainImpl(const DataType* data, uint64_t count);

    /**
     * @brief Encodes a single vector into FP16 format.
     * @param data Input vector data.
     * @param codes Output FP16 codes buffer.
     * @return True if encoding succeeded.
     */
    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes) const;

    /**
     * @brief Decodes FP16 codes back to floating-point vector.
     * @param codes Input FP16 codes.
     * @param data Output decoded vector data.
     * @return True if decoding succeeded.
     */
    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data);

    /**
     * @brief Computes distance between two FP16 vectors.
     * @param codes1 First FP16 vector codes.
     * @param codes2 Second FP16 vector codes.
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
    ProcessQueryImpl(const DataType* query, Computer<FP16Quantizer>& computer) const;

    /**
     * @brief Computes distance from query to an FP16 vector.
     * @param computer Computer object containing query state.
     * @param codes FP16 vector codes.
     * @param dists Output distance array.
     */
    void
    ComputeDistImpl(Computer<FP16Quantizer>& computer, const uint8_t* codes, float* dists) const;

    /**
     * @brief Serializes quantizer state to stream writer (no-op for FP16).
     * @param writer Stream writer for output.
     */
    void
    SerializeImpl(StreamWriter& writer){};

    /**
     * @brief Deserializes quantizer state from stream reader (no-op for FP16).
     * @param reader Stream reader for input.
     */
    void
    DeserializeImpl(StreamReader& reader){};

    /**
     * @brief Returns the quantizer type name.
     * @return String identifier (FP16).
     */
    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_FP16;
    }
};

}  // namespace vsag
