
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
 * @file computer.h
 * @brief Computer classes for distance computation with quantized vectors.
 *
 * This file defines the ComputerInterface and Computer template classes used
 * for efficient distance computation between query vectors and quantized data.
 */

#pragma once

#include <cstdint>
#include <memory>

#include "metric_type.h"
#include "typing.h"
#include "utils/pointer_define.h"
#include "vsag/allocator.h"

namespace vsag {
DEFINE_POINTER(ComputerInterface);

using DataType = float;

/**
 * @brief Abstract interface for computer objects.
 *
 * This base class provides a common interface for distance computation
 * with quantized vectors.
 */
class ComputerInterface {
public:
    ComputerInterface() = default;

    virtual ~ComputerInterface() = default;
};

/**
 * @brief Template class for distance computation with quantized vectors.
 *
 * This class wraps a quantizer and provides methods for processing query
 * vectors and computing distances against quantized data codes.
 *
 * @tparam T The quantizer type.
 */
template <typename T>
class Computer : public ComputerInterface {
public:
    /**
     * @brief Constructs a Computer with the given quantizer and allocator.
     *
     * @param quantizer Pointer to the quantizer for distance computation.
     * @param allocator Pointer to the allocator for memory management.
     */
    explicit Computer(const T* quantizer, Allocator* allocator)
        : quantizer_(quantizer), allocator_(allocator), raw_query_(allocator){};

    ~Computer() override {
        if (quantizer_) {
            quantizer_->ReleaseComputer(*this);
        }
    }

    /**
     * @brief Sets the query vector for distance computation.
     *
     * @param query Pointer to the query vector data.
     */
    void
    SetQuery(const DataType* query) {
        quantizer_->ProcessQuery(query, *this);
    }

    /**
     * @brief Computes distance for a single encoded code.
     *
     * @param codes Pointer to the encoded code.
     * @param dists Output pointer for the computed distance.
     */
    inline void
    ComputeDist(const uint8_t* codes, float* dists) {
        quantizer_->ComputeDist(*this, codes, dists);
    }

    /**
     * @brief Computes distances for a batch of encoded codes.
     *
     * @param count Number of codes to process.
     * @param codes Pointer to the batch of encoded codes.
     * @param dists Output array for computed distances.
     */
    inline void
    ScanBatchDists(uint64_t count, const uint8_t* codes, float* dists) {
        quantizer_->ScanBatchDists(*this, count, codes, dists);
    }

    /**
     * @brief Computes distances for four encoded codes in batch.
     *
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
    ComputeDistsBatch4(const uint8_t* codes1,
                       const uint8_t* codes2,
                       const uint8_t* codes3,
                       const uint8_t* codes4,
                       float& dists1,
                       float& dists2,
                       float& dists3,
                       float& dists4) {
        quantizer_->ComputeDistsBatch4(
            *this, codes1, codes2, codes3, codes4, dists1, dists2, dists3, dists4);
    }

public:
    Allocator* const allocator_{nullptr};  /// Memory allocator
    const T* quantizer_{nullptr};          /// Pointer to the quantizer
    uint8_t* buf_{nullptr};                /// Buffer for intermediate computations
    Vector<float> raw_query_;              /// Storage for the raw query vector
};

template <typename QuantImpl, MetricType metric>
class TransformQuantizer;

/**
 * @brief Specialized Computer for TransformQuantizer.
 *
 * This template specialization provides distance computation for
 * TransformQuantizer by wrapping an inner computer.
 *
 * @tparam QuantImpl The inner quantizer implementation type.
 * @tparam metric The metric type for distance computation.
 */
template <typename QuantImpl, MetricType metric>
class Computer<TransformQuantizer<QuantImpl, metric>> : public ComputerInterface {
public:
    /**
     * @brief Constructs a Computer for TransformQuantizer.
     *
     * @param quantizer Pointer to the TransformQuantizer.
     * @param allocator Pointer to the allocator for memory management.
     */
    explicit Computer(const TransformQuantizer<QuantImpl, metric>* quantizer, Allocator* allocator)
        : quantizer_(quantizer), allocator_(allocator) {
        inner_computer_ = new Computer<QuantImpl>(quantizer_->quantizer_.get(), allocator);
    }

    ~Computer() override {
        if (quantizer_) {
            quantizer_->ReleaseComputer(*this);
        }
        delete inner_computer_;
    }

    /**
     * @brief Sets the query vector for distance computation.
     *
     * @param query Pointer to the query vector data.
     */
    void
    SetQuery(const DataType* query) {
        quantizer_->ProcessQuery(query, *this);
    }

    /**
     * @brief Computes distance for a single encoded code.
     *
     * @param codes Pointer to the encoded code.
     * @param dists Output pointer for the computed distance.
     */
    inline void
    ComputeDist(const uint8_t* codes, float* dists) {
        quantizer_->ComputeDist(*this, codes, dists);
    }

    /**
     * @brief Computes distances for a batch of encoded codes.
     *
     * @param count Number of codes to process.
     * @param codes Pointer to the batch of encoded codes.
     * @param dists Output array for computed distances.
     */
    inline void
    ScanBatchDists(uint64_t count, const uint8_t* codes, float* dists) {
        quantizer_->ScanBatchDists(*this, count, codes, dists);
    }

    /**
     * @brief Computes distances for four encoded codes in batch.
     *
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
    ComputeDistsBatch4(const uint8_t* codes1,
                       const uint8_t* codes2,
                       const uint8_t* codes3,
                       const uint8_t* codes4,
                       float& dists1,
                       float& dists2,
                       float& dists3,
                       float& dists4) {
        quantizer_->ComputeDistsBatch4(
            *this, codes1, codes2, codes3, codes4, dists1, dists2, dists3, dists4);
    }

public:
    Allocator* const allocator_{nullptr};  /// Memory allocator
    const TransformQuantizer<QuantImpl, metric>* quantizer_{
        nullptr};                                   /// Pointer to TransformQuantizer
    uint8_t* buf_{nullptr};                         /// Buffer for intermediate computations
    Computer<QuantImpl>* inner_computer_{nullptr};  /// Inner computer for actual computation
};

}  // namespace vsag
