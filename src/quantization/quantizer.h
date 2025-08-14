
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
#include "quantizer_interface.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/function_exists_check.h"

namespace vsag {
using DataType = float;

template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class TransformQuantizer;

/**
 * @class Quantizer
 * @brief This class is used for quantization and encoding/decoding of data.
 */
template <typename QuantT>
class Quantizer : public QuantizerInterface {
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
    Train(const DataType* data, uint64_t count) override {
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
    ReTrain(const DataType* data, uint64_t count) override {
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
    EncodeOne(const DataType* data, uint8_t* codes) override {
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
    EncodeBatch(const DataType* data, uint8_t* codes, uint64_t count) override {
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
    DecodeOne(const uint8_t* codes, DataType* data) override {
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
    DecodeBatch(const uint8_t* codes, DataType* data, uint64_t count) override {
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
    Compute(const uint8_t* codes1, const uint8_t* codes2) override {
        return cast().ComputeImpl(codes1, codes2);
    }

    inline void
    Serialize(StreamWriter& writer) override {
        StreamWriter::WriteObj(writer, this->dim_);
        StreamWriter::WriteObj(writer, this->metric_);
        StreamWriter::WriteObj(writer, this->code_size_);
        StreamWriter::WriteObj(writer, this->is_trained_);
        return cast().SerializeImpl(writer);
    }

    inline void
    Deserialize(StreamReader& reader) override {
        StreamReader::ReadObj(reader, this->dim_);
        StreamReader::ReadObj(reader, this->metric_);
        StreamReader::ReadObj(reader, this->code_size_);
        StreamReader::ReadObj(reader, this->is_trained_);
        return cast().DeserializeImpl(reader);
    }

    ComputerInterfacePtr
    FactoryComputer() override {
        auto computer_ptr =
            std::make_shared<Computer<QuantT>>(static_cast<QuantT*>(this), allocator_);
        if constexpr (std::is_same_v<QuantT, TransformQuantizer<MetricType::METRIC_TYPE_L2SQR>> or
                      std::is_same_v<QuantT, TransformQuantizer<MetricType::METRIC_TYPE_IP>> or
                      std::is_same_v<QuantT, TransformQuantizer<MetricType::METRIC_TYPE_COSINE>>) {
            computer_ptr->inner_computer_ = cast().quantizer_->FactoryComputer();
        }
        return computer_ptr;
    }

    inline void
    ProcessQuery(const DataType* query, ComputerInterfacePtr computer) const override {
        auto computer_ptr = std::dynamic_pointer_cast<Computer<QuantT>>(computer);
        return cast().ProcessQueryImpl(query, *computer_ptr);
    }

    inline void
    ComputeDist(ComputerInterfacePtr computer, const uint8_t* codes, float* dists) const override {
        auto computer_ptr = std::dynamic_pointer_cast<Computer<QuantT>>(computer);
        return cast().ComputeDistImpl(*computer_ptr, codes, dists);
    }

    inline float
    ComputeDist(ComputerInterfacePtr computer, const uint8_t* codes) const override {
        float dist = 0.0F;
        auto computer_ptr = std::dynamic_pointer_cast<Computer<QuantT>>(computer);
        cast().ComputeDistImpl(*computer_ptr, codes, &dist);
        return dist;
    }

    inline void
    ScanBatchDists(ComputerInterfacePtr computer,
                   uint64_t count,
                   const uint8_t* codes,
                   float* dists) const override {
        auto computer_ptr = std::dynamic_pointer_cast<Computer<QuantT>>(computer);
        return cast().ScanBatchDistImpl(*computer_ptr, count, codes, dists);
    }

    inline void
    ComputeDistsBatch4(ComputerInterfacePtr computer,
                       const uint8_t* codes1,
                       const uint8_t* codes2,
                       const uint8_t* codes3,
                       const uint8_t* codes4,
                       float& dists1,
                       float& dists2,
                       float& dists3,
                       float& dists4) const override {
        auto computer_ptr = std::dynamic_pointer_cast<Computer<QuantT>>(computer);
        if constexpr (has_ComputeDistsBatch4Impl<QuantT>::value) {
            cast().ComputeDistsBatch4Impl(
                *computer_ptr, codes1, codes2, codes3, codes4, dists1, dists2, dists3, dists4);
        } else {
            cast().ComputeDistImpl(*computer_ptr, codes1, &dists1);
            cast().ComputeDistImpl(*computer_ptr, codes2, &dists2);
            cast().ComputeDistImpl(*computer_ptr, codes3, &dists3);
            cast().ComputeDistImpl(*computer_ptr, codes4, &dists4);
        }
    }

    inline void
    ReleaseComputer(ComputerInterfacePtr computer) const override {
        auto computer_ptr = std::dynamic_pointer_cast<Computer<QuantT>>(computer);
        cast().ReleaseComputerImpl(*computer_ptr);
    }

    [[nodiscard]] virtual std::string
    Name() const override {
        return cast().NameImpl();
    }

    [[nodiscard]] MetricType
    Metric() const override {
        return this->metric_;
    }
    [[nodiscard]] bool
    HoldMolds() const {
        return this->hold_molds_;
    }

    virtual void
    Package32(const uint8_t* codes, uint8_t* packaged_codes, int64_t valid_size) const override{};

    virtual void
    Unpack32(const uint8_t* packaged_codes, uint8_t* codes) const override{};

    /**
     * @brief Get the size of the encoded code in bytes.
     *
     * @return The code size in bytes.
     */
    inline uint64_t
    GetCodeSize() const override {
        return this->code_size_;
    }

    inline uint64_t
    GetQueryCodeSize() const override {
        return this->query_code_size_;
    }

    /**
     * @brief Get the dimensionality of the input data.
     *
     * @return The dimensionality of the input data.
     */
    inline int
    GetDim() const override {
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
    uint64_t query_code_size_{0};
    uint64_t code_size_{0};
    bool is_trained_{false};
    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};
    Allocator* const allocator_{nullptr};
    bool hold_molds_{false};

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
