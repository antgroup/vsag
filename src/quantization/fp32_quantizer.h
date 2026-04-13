
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
 * @file fp32_quantizer.h
 * @brief FP32 (32-bit floating-point) quantizer implementation.
 *
 * This file defines the FP32Quantizer class which stores vectors in their
 * original 32-bit floating-point format without compression.
 */

#pragma once

#include "fp32_quantizer_parameter.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantizer.h"

namespace vsag {

/**
 * @brief FP32 Quantizer stores vectors in 32-bit floating-point format.
 *
 * This quantizer preserves the original floating-point precision of vectors.
 * It serves as a no-compression baseline and is useful when exact precision
 * is required or for comparison with other quantization methods.
 *
 * Code layout:
 * +----------------+----------------+
 * | float32-code   | mold (opt)     |
 * | [dim * 4B]     | [4B]           |
 * +----------------+----------------+
 *
 * - float32-code: original vector data (required)
 * - mold: sqrt(sum(vec^2)) for cosine similarity (optional, cosine with hold_molds)
 *
 * @tparam metric The distance metric type (default: L2SQR).
 */
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class FP32Quantizer : public Quantizer<FP32Quantizer<metric>> {
public:
    /**
     * @brief Constructs an FP32Quantizer with dimension and allocator.
     *
     * @param dim The dimensionality of input vectors.
     * @param allocator Pointer to the allocator for memory management.
     */
    explicit FP32Quantizer(int dim, Allocator* allocator);

    /**
     * @brief Constructs an FP32Quantizer from FP32QuantizerParam.
     *
     * @param param The FP32 quantizer parameters.
     * @param common_param Common index parameters including allocator.
     */
    FP32Quantizer(const FP32QuantizerParamPtr& param, const IndexCommonParam& common_param);

    /**
     * @brief Constructs an FP32Quantizer from generic QuantizerParam.
     *
     * @param param The quantizer parameters.
     * @param common_param Common index parameters including allocator.
     */
    FP32Quantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    ~FP32Quantizer() override = default;

    /**
     * @brief Trains the quantizer (no-op for FP32).
     *
     * @param data Pointer to the input data.
     * @param count Number of vectors to train on.
     * @return Always true for FP32 quantizer.
     */
    bool
    TrainImpl(const float* data, uint64_t count);

    /**
     * @brief Encodes a single vector (copy for FP32).
     *
     * @param data Pointer to the input vector.
     * @param codes Output buffer for the encoded code.
     * @return True if encoding was successful.
     */
    bool
    EncodeOneImpl(const float* data, uint8_t* codes);

    /**
     * @brief Encodes a batch of vectors (copy for FP32).
     *
     * @param data Pointer to the input vectors.
     * @param codes Output buffer for encoded codes.
     * @param count Number of vectors to encode.
     * @return True if encoding was successful.
     */
    bool
    EncodeBatchImpl(const float* data, uint8_t* codes, uint64_t count);

    /**
     * @brief Decodes a single encoded code (copy for FP32).
     *
     * @param codes Pointer to the encoded code.
     * @param data Output buffer for the decoded vector.
     * @return True if decoding was successful.
     */
    bool
    DecodeOneImpl(const uint8_t* codes, float* data);

    /**
     * @brief Decodes a batch of encoded codes (copy for FP32).
     *
     * @param codes Pointer to the encoded codes.
     * @param data Output buffer for decoded vectors.
     * @param count Number of codes to decode.
     * @return True if decoding was successful.
     */
    bool
    DecodeBatchImpl(const uint8_t* codes, float* data, uint64_t count);

    /**
     * @brief Computes distance between two encoded codes.
     *
     * @param codes1 Pointer to the first encoded code.
     * @param codes2 Pointer to the second encoded code.
     * @return The computed distance.
     */
    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2);

    /**
     * @brief Serializes the quantizer state (no-op for FP32).
     *
     * @param writer The stream writer for output.
     */
    void
    SerializeImpl(StreamWriter& writer){};

    /**
     * @brief Deserializes the quantizer state (no-op for FP32).
     *
     * @param reader The stream reader for input.
     */
    void
    DeserializeImpl(StreamReader& reader){};

    /**
     * @brief Processes a query vector for distance computation.
     *
     * @param query Pointer to the query vector.
     * @param computer Reference to the computer object.
     */
    void
    ProcessQueryImpl(const float* query, Computer<FP32Quantizer<metric>>& computer) const;

    /**
     * @brief Computes distance between processed query and an encoded code.
     *
     * @param computer Reference to the computer containing processed query.
     * @param codes Pointer to the encoded code.
     * @param dists Output pointer for the computed distance.
     */
    void
    ComputeDistImpl(Computer<FP32Quantizer<metric>>& computer,
                    const uint8_t* codes,
                    float* dists) const;

    /**
     * @brief Computes distances for a batch of encoded codes.
     *
     * @param computer Reference to the computer containing processed query.
     * @param count Number of codes to process.
     * @param codes Pointer to the batch of encoded codes.
     * @param dists Output array for computed distances.
     */
    void
    ScanBatchDistImpl(Computer<FP32Quantizer<metric>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    /**
     * @brief Computes distances for four encoded codes in batch.
     *
     * @param computer Reference to the computer containing processed query.
     * @param codes1 Pointer to the first encoded code.
     * @param codes2 Pointer to the second encoded code.
     * @param codes3 Pointer to the third encoded code.
     * @param codes4 Pointer to the fourth encoded code.
     * @param dists1 Output reference for the first distance.
     * @param dists2 Output reference for the second distance.
     * @param dists3 Output reference for the third distance.
     * @param dists4 Output reference for the fourth distance.
     */
    void
    ComputeDistsBatch4Impl(Computer<FP32Quantizer<metric>>& computer,
                           const uint8_t* codes1,
                           const uint8_t* codes2,
                           const uint8_t* codes3,
                           const uint8_t* codes4,
                           float& dists1,
                           float& dists2,
                           float& dists3,
                           float& dists4) const;

    /**
     * @brief Releases resources associated with the computer.
     *
     * @param computer Reference to the computer to release.
     */
    void
    ReleaseComputerImpl(Computer<FP32Quantizer<metric>>& computer) const;

    /**
     * @brief Gets the name of the FP32 quantizer.
     *
     * @return The quantizer name string "fp32".
     */
    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_FP32;
    }
};

}  // namespace vsag
