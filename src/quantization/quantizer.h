
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
#include "logger.h"
#include "metric_type.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/function_exists_check.h"

namespace vsag {
using DataType = float;

/**
 * @class Quantizer
 * @brief This class is used for quantization and encoding/decoding of data.
 */
template <typename QuantT>
class Quantizer {
public:
    explicit Quantizer<QuantT>(int dim, Allocator* allocator)
        : dim_(dim), code_size_(dim * sizeof(DataType)), allocator_(allocator){};

    virtual ~Quantizer() = default;

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

    std::shared_ptr<Computer<QuantT>>
    FactoryComputer() {
        return std::make_shared<Computer<QuantT>>(static_cast<QuantT*>(this));
    }

    inline void
    ProcessQuery(const DataType* query, Computer<QuantT>& computer) const {
        return cast().ProcessQueryImpl(query, computer);
    }

    inline void
    ComputeDist(Computer<QuantT>& computer, const uint8_t* codes, float* dists) const {
        return cast().ComputeDistImpl(computer, codes, dists);
    }

    inline float
    ComputeDist(Computer<QuantT>& computer, const uint8_t* codes) const {
        float dist = 0.0F;
        cast().ComputeDistImpl(computer, codes, &dist);
        return dist;
    }

    inline void
    ScanBatchDists(Computer<QuantT>& computer,
                   uint64_t count,
                   const uint8_t* codes,
                   float* dists) const {
        return cast().ScanBatchDistImpl(computer, count, codes, dists);
    }

    inline void
    ComputeDistsBatch4(Computer<QuantT>& computer,
                       const uint8_t* codes1,
                       const uint8_t* codes2,
                       const uint8_t* codes3,
                       const uint8_t* codes4,
                       float& dists1,
                       float& dists2,
                       float& dists3,
                       float& dists4) const {
        if constexpr (has_ComputeDistsBatch4Impl<QuantT>::value) {
            cast().ComputeDistsBatch4Impl(
                computer, codes1, codes2, codes3, codes4, dists1, dists2, dists3, dists4);
        } else {
            cast().ComputeDistImpl(computer, codes1, &dists1);
            cast().ComputeDistImpl(computer, codes2, &dists2);
            cast().ComputeDistImpl(computer, codes3, &dists3);
            cast().ComputeDistImpl(computer, codes4, &dists4);
        }
    }

    inline void
    ReleaseComputer(Computer<QuantT>& computer) const {
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

    virtual void
    Package32(const uint8_t* codes, uint8_t* packaged_codes, int64_t valid_size) const {};

    virtual void
    Unpack32(const uint8_t* packaged_codes, uint8_t* codes) const {};

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
    inline QuantT&
    cast() {
        return static_cast<QuantT&>(*this);
    }

    inline const QuantT&
    cast() const {
        return static_cast<const QuantT&>(*this);
    }

    friend QuantT;

private:
    uint64_t dim_{0};
    uint64_t code_size_{0};
    bool is_trained_{false};
    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};
    Allocator* const allocator_{nullptr};

    GENERATE_HAS_MEMBER_FUNCTION(ComputeDistsBatch4Impl,
                                 void,
                                 std::declval<Computer<QuantT>&>(),
                                 std::declval<const uint8_t*>(),
                                 std::declval<const uint8_t*>(),
                                 std::declval<const uint8_t*>(),
                                 std::declval<const uint8_t*>(),
                                 std::declval<float&>(),
                                 std::declval<float&>(),
                                 std::declval<float&>(),
                                 std::declval<float&>())
};

#define TEMPLATE_QUANTIZER(Name)                        \
    template class Name<MetricType::METRIC_TYPE_L2SQR>; \
    template class Name<MetricType::METRIC_TYPE_IP>;    \
    template class Name<MetricType::METRIC_TYPE_COSINE>;
}  // namespace vsag
