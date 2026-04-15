
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
 * @file quantizer_adapter.h
 * @brief Adapter class for quantizers with different data types.
 *
 * This file defines the QuantizerAdapter template class that wraps an existing
 * quantizer to support different input data types (int8_t, uint16_t).
 */

#pragma once

#include <memory>
#include <string>
#include <type_traits>

#include "data_type.h"
#include "index_common_param.h"
#include "quantization/computer.h"
#include "quantization/product_quantization/product_quantizer.h"
#include "quantization/quantizer.h"
#include "quantization/quantizer_parameter.h"

namespace vsag {

/**
 * @brief Adapter class for quantizers with different data types.
 *
 * This template class wraps an existing quantizer to support input data types
 * other than float32, specifically int8_t and uint16_t. It performs data type
 * conversion before passing data to the inner quantizer.
 *
 * @tparam QuantT The inner quantizer type.
 * @tparam DataT The input data type (int8_t or uint16_t).
 */
template <typename QuantT, typename DataT>
class QuantizerAdapter : public Quantizer<QuantizerAdapter<QuantT, DataT>> {
    static_assert(std::is_same_v<DataT, int8_t> || std::is_same_v<DataT, uint16_t>,
                  "QuantizerAdapter currently only supports int8_t and uint16_t data types");

public:
    /**
     * @brief Constructs a QuantizerAdapter from parameters.
     *
     * @param param The quantizer parameters.
     * @param common_param Common index parameters including allocator.
     */
    explicit QuantizerAdapter(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    ~QuantizerAdapter() override = default;

    /**
     * @brief Trains the inner quantizer with the provided data.
     *
     * @param data Pointer to the input data.
     * @param count Number of vectors to train on.
     * @return True if training was successful.
     */
    bool
    TrainImpl(const DataType* data, uint64_t count);

    /**
     * @brief Encodes a single vector.
     *
     * @param data Pointer to the input vector.
     * @param codes Output buffer for the encoded code.
     * @return True if encoding was successful.
     */
    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes);

    /**
     * @brief Encodes a batch of vectors.
     *
     * @param data Pointer to the input vectors.
     * @param codes Output buffer for encoded codes.
     * @param count Number of vectors to encode.
     * @return True if encoding was successful.
     */
    bool
    EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count);

    /**
     * @brief Decodes a single encoded code.
     *
     * @param codes Pointer to the encoded code.
     * @param data Output buffer for the decoded vector.
     * @return True if decoding was successful.
     */
    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data);

    /**
     * @brief Decodes a batch of encoded codes.
     *
     * @param codes Pointer to the encoded codes.
     * @param data Output buffer for decoded vectors.
     * @param count Number of codes to decode.
     * @return True if decoding was successful.
     */
    bool
    DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count);

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
     * @brief Serializes the adapter state to a stream.
     *
     * @param writer The stream writer for output.
     */
    void
    SerializeImpl(StreamWriter& writer);

    /**
     * @brief Deserializes the adapter state from a stream.
     *
     * @param reader The stream reader for input.
     */
    void
    DeserializeImpl(StreamReader& reader);

    /**
     * @brief Processes a query vector for distance computation.
     *
     * @param query Pointer to the query vector.
     * @param computer Reference to the computer object.
     */
    void
    ProcessQueryImpl(const DataType* query,
                     Computer<QuantizerAdapter<QuantT, DataT>>& computer) const;

    /**
     * @brief Computes distance between processed query and an encoded code.
     *
     * @param computer Reference to the computer containing processed query.
     * @param codes Pointer to the encoded code.
     * @param dists Output pointer for the computed distance.
     */
    void
    ComputeDistImpl(Computer<QuantizerAdapter<QuantT, DataT>>& computer,
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
    ScanBatchDistImpl(Computer<QuantizerAdapter<QuantT, DataT>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    /**
     * @brief Releases resources associated with the computer.
     *
     * @param computer Reference to the computer to release.
     */
    void
    ReleaseComputerImpl(Computer<QuantizerAdapter<QuantT, DataT>>& computer) const;

    /**
     * @brief Gets the name of the adapter quantizer.
     *
     * @return The quantizer name including adapter prefix.
     */
    [[nodiscard]] std::string
    NameImpl() const {
        return std::string("QUANTIZATION_ADAPTER_") + inner_quantizer_->Name();
    }

private:
    using Base = Quantizer<QuantT>;
    std::shared_ptr<QuantT> inner_quantizer_{nullptr};  /// The wrapped inner quantizer
    DataTypes data_type_{DataTypes::DATA_TYPE_FLOAT};   /// Input data type
};

#define TEMPLATE_QUANTIZER_ADAPTER(QuantType, DataT)                                  \
    template class QuantizerAdapter<QuantType<MetricType::METRIC_TYPE_L2SQR>, DataT>; \
    template class QuantizerAdapter<QuantType<MetricType::METRIC_TYPE_IP>, DataT>;    \
    template class QuantizerAdapter<QuantType<MetricType::METRIC_TYPE_COSINE>, DataT>;
}  // namespace vsag
