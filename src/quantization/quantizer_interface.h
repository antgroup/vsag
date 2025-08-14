
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

#include <cstdint>
#include <memory>

#include "computer.h"
#include "index/index_common_param.h"
#include "logger.h"
#include "metric_type.h"
#include "quantizer_parameter.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/function_exists_check.h"

namespace vsag {

using DataType = float;
class QuantizerInterface;
using QuantizerInterfacePtr = std::shared_ptr<QuantizerInterface>;

/**
 * @class Quantizer
 * @brief This class is used for quantization and encoding/decoding of data.
 */
class QuantizerInterface {
public:
    QuantizerInterface() = default;

    static QuantizerInterfacePtr
    MakeInstance(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

public:
    /**
     * @brief Trains the model using the provided data.
     *
     * @param data Pointer to the input data.
     * @param count The number of elements in the data array.
     * @return True if training was successful; False otherwise.
     */
    virtual bool
    Train(const DataType* data, uint64_t count) = 0;

    /**
     * @brief Re-Train the model using the provided data.
     *
     * @param data Pointer to the input data.
     * @param count The number of elements in the data array.
     * @return True if training was successful; False otherwise.
     */
    virtual bool
    ReTrain(const DataType* data, uint64_t count) = 0;

    /**
     * @brief Encodes one element from the input data into a code.
     *
     * @param data Pointer to the input data.
     * @param codes Output buffer where the encoded code will be stored.
     * @return True if encoding was successful; False otherwise.
     */
    virtual bool
    EncodeOne(const DataType* data, uint8_t* codes) = 0;

    /**
     * @brief Encodes multiple elements from the input data into codes.
     *
     * @param data Pointer to the input data.
     * @param codes Output buffer where the encoded codes will be stored.
     * @param count The number of elements to encode.
     * @return True if encoding was successful; False otherwise.
     */
    virtual bool
    EncodeBatch(const DataType* data, uint8_t* codes, uint64_t count) = 0;

    /**
     * @brief Decodes an encoded code back into its original data representation.
     *
     * @param codes Pointer to the encoded code.
     * @param data Output buffer where the decoded data will be stored.
     * @return True if decoding was successful; False otherwise.
     */
    virtual bool
    DecodeOne(const uint8_t* codes, DataType* data) = 0;

    /**
     * @brief Decodes multiple encoded codes back into their original data representations.
     *
     * @param codes Pointer to the encoded codes.
     * @param data Output buffer where the decoded data will be stored.
     * @param count The number of elements to decode.
     * @return True if decoding was successful; False otherwise.
     */
    virtual bool
    DecodeBatch(const uint8_t* codes, DataType* data, uint64_t count) = 0;

    /**
     * @brief Compute the distance between two encoded codes.
     *
     * @tparam float the computed distance.
     * @param codes1 Pointer to the first encoded code.
     * @param codes2 Pointer to the second encoded code.
     * @return The computed distance between the decoded data points.
     */
    virtual float
    Compute(const uint8_t* codes1, const uint8_t* codes2) = 0;

    virtual void
    Serialize(StreamWriter& writer) = 0;

    virtual void
    Deserialize(StreamReader& reader) = 0;

    virtual ComputerInterfacePtr
    FactoryComputer() = 0;

    virtual void
    ProcessQuery(const DataType* query, ComputerInterfacePtr computer) const = 0;

    virtual void
    ComputeDist(ComputerInterfacePtr computer, const uint8_t* codes, float* dists) const = 0;

    virtual float
    ComputeDist(ComputerInterfacePtr computer, const uint8_t* codes) const = 0;

    virtual void
    ScanBatchDists(ComputerInterfacePtr computer,
                   uint64_t count,
                   const uint8_t* codes,
                   float* dists) const = 0;

    virtual void
    ComputeDistsBatch4(ComputerInterfacePtr computer,
                       const uint8_t* codes1,
                       const uint8_t* codes2,
                       const uint8_t* codes3,
                       const uint8_t* codes4,
                       float& dists1,
                       float& dists2,
                       float& dists3,
                       float& dists4) const = 0;

    virtual void
    ReleaseComputer(ComputerInterfacePtr computer) const = 0;

    [[nodiscard]] virtual std::string
    Name() const = 0;

    [[nodiscard]] virtual MetricType
    Metric() const = 0;

    virtual void
    Package32(const uint8_t* codes, uint8_t* packaged_codes, int64_t valid_size) const = 0;

    virtual void
    Unpack32(const uint8_t* packaged_codes, uint8_t* codes) const = 0;

    /**
     * @brief Get the size of the encoded code in bytes.
     *
     * @return The code size in bytes.
     */
    virtual uint64_t
    GetCodeSize() const = 0;

    virtual uint64_t
    GetQueryCodeSize() const = 0;

    /**
     * @brief Get the dimensionality of the input data.
     *
     * @return The dimensionality of the input data.
     */
    virtual int
    GetDim() const = 0;
};

}  // namespace vsag
