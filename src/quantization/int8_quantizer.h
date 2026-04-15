
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

/**
 * @file int8_quantizer.h
 * @brief INT8 quantizer for storing vectors in 8-bit integer format.
 */

#pragma once

#include <cstdint>

#include "index_common_param.h"
#include "inner_string_params.h"
#include "int8_quantizer_parameter.h"
#include "quantization/computer.h"
#include "quantization/quantizer_parameter.h"
#include "quantizer.h"

namespace vsag {

/**
 * @brief INT8 Quantizer stores vectors in 8-bit integer format.
 *
 * Code layout:
 * +----------------+----------------+
 * | int8-code      | mold (opt)     |
 * | [dim * 1B]     | [4B]           |
 * +----------------+----------------+
 *
 * - int8-code: quantized 8-bit integer values (required)
 * - mold: sqrt(sum(vec^2)) for normalization (optional, cosine only)
 *
 * @tparam metric Distance metric type.
 */
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class INT8Quantizer : public Quantizer<INT8Quantizer<metric>> {
public:
    /**
     * @brief Constructs an INT8 quantizer with dimension and allocator.
     * @param dim Vector dimension.
     * @param allocator Memory allocator.
     */
    explicit INT8Quantizer(int dim, Allocator* allocator);

    /**
     * @brief Constructs an INT8 quantizer from parameter pointer.
     * @param param INT8 quantizer parameter pointer.
     * @param common_param Common index parameters.
     */
    INT8Quantizer(const INT8QuantizerParamPtr& param, const IndexCommonParam& common_param);

    /**
     * @brief Constructs an INT8 quantizer from base parameter pointer.
     * @param param Quantizer parameter pointer.
     * @param common_param Common index parameters.
     */
    INT8Quantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    ~INT8Quantizer() override = default;

    /**
     * @brief Trains the quantizer with given data.
     * @param data Training data.
     * @param count Number of vectors.
     * @return True if training succeeded.
     */
    bool
    TrainImpl(const DataType* data, uint64_t count);

    /**
     * @brief Encodes a single vector.
     * @param data Input vector data.
     * @param codes Output code buffer.
     * @return True if encoding succeeded.
     */
    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes);

    /**
     * @brief Encodes a batch of vectors.
     * @param data Input vector data.
     * @param codes Output code buffer.
     * @param count Number of vectors.
     * @return True if encoding succeeded.
     */
    bool
    EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count);

    /**
     * @brief Decodes a single code to vector.
     * @param codes Input code buffer.
     * @param data Output vector data.
     * @return True if decoding succeeded.
     */
    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data);

    /**
     * @brief Decodes a batch of codes to vectors.
     * @param codes Input code buffer.
     * @param data Output vector data.
     * @param count Number of vectors.
     * @return True if decoding succeeded.
     */
    bool
    DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count);

    /**
     * @brief Computes distance between two codes.
     * @param codes1 First code buffer.
     * @param codes2 Second code buffer.
     * @return Computed distance.
     */
    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2);

    /**
     * @brief Serializes the quantizer to stream.
     * @param writer Stream writer.
     */
    void
    SerializeImpl(StreamWriter& writer){};

    /**
     * @brief Deserializes the quantizer from stream.
     * @param reader Stream reader.
     */
    void
    DeserializeImpl(StreamReader& reader){};

    /**
     * @brief Processes a query vector for distance computation.
     * @param query Query vector data.
     * @param computer Computer object to store query codes.
     */
    void
    ProcessQueryImpl(const DataType* query, Computer<INT8Quantizer<metric>>& computer) const;

    /**
     * @brief Computes distance between query code and base code.
     * @param computer Computer object containing query codes.
     * @param codes Base code buffer.
     * @param dists Output distance array.
     */
    void
    ComputeDistImpl(Computer<INT8Quantizer<metric>>& computer,
                    const uint8_t* codes,
                    float* dists) const;

    /**
     * @brief Computes distances for a batch of codes.
     * @param computer Computer object containing query codes.
     * @param count Number of codes.
     * @param codes Base code buffer.
     * @param dists Output distance array.
     */
    void
    ScanBatchDistImpl(Computer<INT8Quantizer<metric>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    /**
     * @brief Releases resources held by computer.
     * @param computer Computer object to release.
     */
    void
    ReleaseComputerImpl(Computer<INT8Quantizer<metric>>& computer) const;

    /**
     * @brief Gets the quantizer name.
     * @return Quantizer type name string.
     */
    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_INT8;
    }
};
}  // namespace vsag
