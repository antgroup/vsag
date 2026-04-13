
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
 * @file quantizer.h
 * @brief Base template class for vector quantization operations.
 *
 * This file defines the Quantizer template class which provides CRTP-based
 * interface for encoding, decoding, and distance computation on quantized vectors.
 */

#pragma once

#include <cstdint>
#include <memory>

#include "computer.h"
#include "metric_type.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/function_exists_check.h"

namespace vsag {

/**
 * @class Quantizer
 * @brief CRTP base template class for vector quantization operations.
 *
 * This class provides a unified interface for encoding, decoding, training,
 * and distance computation on quantized vectors. Derived classes must implement
 * the required `*Impl` methods following the CRTP pattern.
 *
 * @tparam QuantT The derived quantizer type implementing the quantization logic.
 */
template <typename QuantT>
class Quantizer {
public:
    /**
     * @brief Constructs a Quantizer with the specified dimension and allocator.
     *
     * @param dim The dimensionality of input vectors.
     * @param allocator Pointer to the allocator for memory management.
     */
    explicit Quantizer<QuantT>(int dim, Allocator* allocator)
        : dim_(dim), code_size_(dim * sizeof(float)), allocator_(allocator){};

    virtual ~Quantizer() = default;

    /**
     * @brief Trains the model using the provided data.
     *
     * @param data Pointer to the input data.
     * @param count The number of elements in the data array.
     * @return True if training was successful; False otherwise.
     */
    bool
    Train(const float* data, uint64_t count) {
        if (this->is_trained_) {
            return true;
        }
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
    ReTrain(const float* data, uint64_t count) {
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
    EncodeOne(const float* data, uint8_t* codes) {
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
    EncodeBatch(const float* data, uint8_t* codes, uint64_t count) {
        return cast().EncodeBatchImpl(data, codes, count);
    }

    /**
     * @brief Default implementation of EncodeBatchImpl using loop.
     * Subclasses can override for optimized batch encoding.
     */
    bool
    EncodeBatchImpl(const float* data, uint8_t* codes, uint64_t count) {
        for (uint64_t i = 0; i < count; ++i) {
            if (!cast().EncodeOneImpl(data + i * dim_, codes + i * code_size_)) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Decodes an encoded code back into its original data representation.
     *
     * @param codes Pointer to the encoded code.
     * @param data Output buffer where the decoded data will be stored.
     * @return True if decoding was successful; False otherwise.
     */
    bool
    DecodeOne(const uint8_t* codes, float* data) {
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
    DecodeBatch(const uint8_t* codes, float* data, uint64_t count) {
        return cast().DecodeBatchImpl(codes, data, count);
    }

    /**
     * @brief Default implementation of DecodeBatchImpl using loop.
     * Subclasses can override for optimized batch decoding.
     */
    bool
    DecodeBatchImpl(const uint8_t* codes, float* data, uint64_t count) {
        for (uint64_t i = 0; i < count; ++i) {
            if (!cast().DecodeOneImpl(codes + i * code_size_, data + i * dim_)) {
                return false;
            }
        }
        return true;
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

    /**
     * @brief Serializes the quantizer state to a stream writer.
     *
     * @param writer The stream writer for output.
     */
    inline void
    Serialize(StreamWriter& writer) {
        StreamWriter::WriteObj(writer, this->dim_);
        StreamWriter::WriteObj(writer, this->metric_);
        StreamWriter::WriteObj(writer, this->code_size_);
        StreamWriter::WriteObj(writer, this->is_trained_);
        return cast().SerializeImpl(writer);
    }

    /**
     * @brief Deserializes the quantizer state from a stream reader.
     *
     * @param reader The stream reader for input.
     */
    inline void
    Deserialize(StreamReader& reader) {
        StreamReader::ReadObj(reader, this->dim_);
        StreamReader::ReadObj(reader, this->metric_);
        StreamReader::ReadObj(reader, this->code_size_);
        StreamReader::ReadObj(reader, this->is_trained_);
        return cast().DeserializeImpl(reader);
    }

    /**
     * @brief Creates a computer object for distance computation.
     *
     * @return A shared pointer to the computer instance.
     */
    std::shared_ptr<Computer<QuantT>>
    FactoryComputer() {
        return std::make_shared<Computer<QuantT>>(static_cast<QuantT*>(this), allocator_);
    }

    /**
     * @brief Processes a query vector for distance computation.
     *
     * @param query Pointer to the query vector data.
     * @param computer Reference to the computer object for storing processed query.
     */
    inline void
    ProcessQuery(const float* query, Computer<QuantT>& computer) const {
        return cast().ProcessQueryImpl(query, computer);
    }

    /**
     * @brief Computes distance between processed query and encoded codes.
     *
     * @param computer Reference to the computer containing processed query.
     * @param codes Pointer to the encoded codes.
     * @param dists Output array for computed distances.
     */
    inline void
    ComputeDist(Computer<QuantT>& computer, const uint8_t* codes, float* dists) const {
        return cast().ComputeDistImpl(computer, codes, dists);
    }

    /**
     * @brief Computes a single distance between processed query and one encoded code.
     *
     * @param computer Reference to the computer containing processed query.
     * @param codes Pointer to a single encoded code.
     * @return The computed distance value.
     */
    inline float
    ComputeDist(Computer<QuantT>& computer, const uint8_t* codes) const {
        float dist = 0.0F;
        cast().ComputeDistImpl(computer, codes, &dist);
        return dist;
    }

    /**
     * @brief Computes distances for a batch of encoded codes.
     *
     * @param computer Reference to the computer containing processed query.
     * @param count Number of codes to process.
     * @param codes Pointer to the batch of encoded codes.
     * @param dists Output array for computed distances.
     */
    inline void
    ScanBatchDists(Computer<QuantT>& computer,
                   uint64_t count,
                   const uint8_t* codes,
                   float* dists) const {
        return cast().ScanBatchDistImpl(computer, count, codes, dists);
    }

    /**
     * @brief Computes distances for four encoded codes in batch.
     *
     * This optimized batch method computes distances for four codes at once,
     * potentially using SIMD instructions for better performance.
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

    /**
     * @brief Releases resources associated with the computer.
     *
     * @param computer Reference to the computer to release.
     */
    inline void
    ReleaseComputer(Computer<QuantT>& computer) const {
        cast().ReleaseComputerImpl(computer);
    }

    /**
     * @brief Default implementation of ReleaseComputerImpl.
     * Subclasses can override for custom release logic.
     */
    void
    ReleaseComputerImpl(Computer<QuantT>& computer) const {
        allocator_->Deallocate(computer.buf_);
    }

    /**
     * @brief Default implementation of ScanBatchDistImpl using loop.
     * Subclasses can override for optimized batch distance computation.
     */
    void
    ScanBatchDistImpl(Computer<QuantT>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const {
        for (uint64_t i = 0; i < count; ++i) {
            cast().ComputeDistImpl(computer, codes + i * code_size_, dists + i);
        }
    }

    /**
     * @brief Gets the name of the quantizer.
     *
     * @return The quantizer name string.
     */
    [[nodiscard]] virtual std::string
    Name() const {
        return cast().NameImpl();
    }

    /**
     * @brief Gets the metric type used by this quantizer.
     *
     * @return The metric type enum value.
     */
    [[nodiscard]] MetricType
    Metric() const {
        return this->metric_;
    }

    /**
     * @brief Checks if the quantizer holds mold (norm) data.
     *
     * @return True if molds are held, false otherwise.
     */
    [[nodiscard]] bool
    HoldMolds() const {
        return this->hold_molds_;
    }

    /**
     * @brief Packages 32 codes into a compact format.
     *
     * @param codes Pointer to the original codes.
     * @param packaged_codes Output buffer for packaged codes.
     * @param valid_size Number of valid codes to package.
     */
    virtual void
    Package32(const uint8_t* codes, uint8_t* packaged_codes, int64_t valid_size) const {};

    /**
     * @brief Unpacks 32 codes from compact format to original format.
     *
     * @param packaged_codes Pointer to the packaged codes.
     * @param codes Output buffer for unpacked codes.
     */
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
     * @brief Gets the code size for query vectors in bytes.
     *
     * @return The query code size in bytes.
     */
    inline uint64_t
    GetQueryCodeSize() const {
        return this->query_code_size_;
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
    uint64_t dim_{0};                                   /// Dimensionality of input vectors
    uint64_t query_code_size_{0};                       /// Code size for query vectors
    uint64_t code_size_{0};                             /// Code size for data vectors
    bool is_trained_{false};                            /// Whether the quantizer has been trained
    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};  /// Distance metric type
    Allocator* const allocator_{nullptr};               /// Memory allocator
    bool hold_molds_{false};                            /// Whether to hold mold (norm) data

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
