
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

#include "metric_type.h"
namespace vsag {
using DataType = float;

template <typename T>
class Quantizer;

template <typename T>
class Computer {
public:
    ~Computer() {
        delete[] buf_;
    }

    explicit Computer(const T& quantizer) : quantizer_(&quantizer){};

    void
    SetQuery(const DataType* query) {
        quantizer_->ProcessQuery(query, *this);
    }

    inline void
    ComputeDist(const uint8_t* codes, float* dists) {
        quantizer_->ComputeDist(*this, codes, dists);
    }

    const T* quantizer_{nullptr};

    uint8_t* buf_{nullptr};  // todo how to alloc and free
};

/**
 * @class Quantizer
 * @brief This class is used for quantization and encoding/decoding of data.
 */
template <typename T>
class Quantizer {
public:
    explicit Quantizer<T>(int dim) : dim_(dim), codeSize_(dim * sizeof(DataType)){};

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
        this->isTrained_ = false;
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

    std::unique_ptr<Computer<T>>
    FactoryComputer() {
        return std::make_unique<Computer<T>>(cast());
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
        float dist = 0.;
        cast().ComputeDistImpl(computer, codes, dist);
        return dist;
    }

    /**
     * @brief Get the size of the encoded code in bytes.
     *
     * @return The code size in bytes.
     */
    inline uint64_t
    GetCodeSize() const {
        return this->codeSize_;
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

    uint64_t codeSize_{0};

    bool isTrained_{false};

    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};
};

}  // namespace vsag
