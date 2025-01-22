
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

#include "../logger.h"
#include "computer.h"
#include "metric_type.h"
#include "stream_reader.h"
#include "stream_writer.h"

namespace vsag {
using DataType = float;

/**
 * @class Quantizer
 * @brief This class is used for quantization and encoding/decoding of data.
 */
template <typename T>
class Quantizer {
public:
    explicit Quantizer<T>(int dim, Allocator* allocator)
        : dim_(dim), code_size_(dim * sizeof(DataType)), allocator_(allocator){};

    ~Quantizer() = default;

    /**
     * @brief Trains the model using the provided data.
     *
     * @param data Pointer to the input data.
     * @param count The number of elements in the data array.
     * @return True if training was successful; False otherwise.
     */
    bool
    Train(const DataType* data, uint64_t count) {
        return cast().TrainImpl(data, count);
    }

    /**
     * @brief Re-Train the model using the provided data.
     *
     * @param data Pointer to the input data.
     * @param count The number of elements in the data array.
     * @return True if training was successful; False otherwise.
     */
    bool
    ReTrain(const DataType* data, uint64_t count) {
        this->is_trained_ = false;
        return cast().TrainImpl(data, count);
    }

    /**
     * @brief Encodes one element from the input data into a code.
     *
     * @param data Pointer to the input data.
     * @param codes Output buffer where the encoded code will be stored.
     * @return True if encoding was successful; False otherwise.
     */
    bool
    EncodeOne(const DataType* data, uint8_t* codes) {
        return cast().EncodeOneImpl(data, codes);
    }

    /**
     * @brief Encodes multiple elements from the input data into codes.
     *
     * @param data Pointer to the input data.
     * @param codes Output buffer where the encoded codes will be stored.
     * @param count The number of elements to encode.
     * @return True if encoding was successful; False otherwise.
     */
    bool
    EncodeBatch(const DataType* data, uint8_t* codes, uint64_t count) {
        return cast().EncodeBatchImpl(data, codes, count);
    }

    /**
     * @brief Decodes an encoded code back into its original data representation.
     *
     * @param codes Pointer to the encoded code.
     * @param data Output buffer where the decoded data will be stored.
     * @return True if decoding was successful; False otherwise.
     */
    bool
    DecodeOne(const uint8_t* codes, DataType* data) {
        return cast().DecodeOneImpl(codes, data);
    }

    /**
     * @brief Decodes multiple encoded codes back into their original data representations.
     *
     * @param codes Pointer to the encoded codes.
     * @param data Output buffer where the decoded data will be stored.
     * @param count The number of elements to decode.
     * @return True if decoding was successful; False otherwise.
     */
    bool
    DecodeBatch(const uint8_t* codes, DataType* data, uint64_t count) {
        return cast().DecodeBatchImpl(codes, data, count);
    }

    /**
     * @brief Compute the distance between two encoded codes.
     *
     * @tparam float the computed distance.
     * @param codes1 Pointer to the first encoded code.
     * @param codes2 Pointer to the second encoded code.
     * @return The computed distance between the decoded data points.
     */
    inline float
    Compute(const uint8_t* codes1, const uint8_t* codes2) {
        return cast().ComputeImpl(codes1, codes2);
    }

    inline void
    Serialize(StreamWriter& writer) {
        StreamWriter::WriteObj(writer, this->dim_);
        StreamWriter::WriteObj(writer, this->metric_);
        StreamWriter::WriteObj(writer, this->code_size_);
        StreamWriter::WriteObj(writer, this->is_trained_);
        return cast().SerializeImpl(writer);
    }

    inline void
    Deserialize(StreamReader& reader) {
        StreamReader::ReadObj(reader, this->dim_);
        StreamReader::ReadObj(reader, this->metric_);
        StreamReader::ReadObj(reader, this->code_size_);
        StreamReader::ReadObj(reader, this->is_trained_);
        return cast().DeserializeImpl(reader);
    }

    std::shared_ptr<Computer<T>>
    FactoryComputer() {
        return std::make_shared<Computer<T>>(static_cast<T*>(this));
    }

    inline void
    ProcessQuery(const DataType* query, Computer<T>& computer) const {
        return cast().ProcessQueryImpl(query, computer);
    }

    inline void
    ComputeDist(Computer<T>& computer, const uint8_t* codes, float* dists) const {
        return cast().ComputeDistImpl(computer, codes, dists);
    }

    inline float
    ComputeDist(Computer<T>& computer, const uint8_t* codes) const {
        float dist = 0.0f;
        cast().ComputeDistImpl(computer, codes, &dist);
        return dist;
    }

    inline void
    ComputeBatchDists(Computer<T>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const {
        return cast().ComputeBatchDistImpl(computer, count, codes, dists);
    }

    inline void
    ReleaseComputer(Computer<T>& computer) const {
        cast().ReleaseComputerImpl(computer);
    }

    [[nodiscard]] virtual std::string
    Name() const {
        return cast().NameImpl();
    }

    [[nodiscard]] MetricType
    Metric() const {
        return this->metric_;
    }

    /**
     * @brief Get the size of the encoded code in bytes.
     *
     * @return The code size in bytes.
     */
    inline uint64_t
    GetCodeSize() const {
        return this->code_size_;
    }

    /**
     * @brief Get the dimensionality of the input data.
     *
     * @return The dimensionality of the input data.
     */
    inline int
    GetDim() const {
        return this->dim_;
    }

private:
    inline T&
    cast() {
        return static_cast<T&>(*this);
    }

    inline const T&
    cast() const {
        return static_cast<const T&>(*this);
    }

    friend T;

private:
    uint64_t dim_{0};
    uint64_t code_size_{0};
    bool is_trained_{false};
    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};
    Allocator* const allocator_{nullptr};
};

}  // namespace vsag
