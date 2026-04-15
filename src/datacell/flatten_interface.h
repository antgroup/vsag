/**
 * @file flatten_interface.h
 * @brief Flatten interface for vector quantization and storage.
 *
 * This file defines the abstract interface for flatten data structures
 * that store quantized vectors in a contiguous format for efficient search.
 */

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

#include <shared_mutex>
#include <string>

#include "flatten_datacell_parameter.h"
#include "flatten_interface_parameter.h"
#include "impl/runtime_parameter.h"
#include "index_common_param.h"
#include "io/reader_io.h"
#include "quantization/computer.h"
#include "query_context.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "utils/pointer_define.h"
#include "vsag/constants.h"

namespace vsag {

DEFINE_POINTER(FlattenInterface);

/**
 * @brief Abstract interface for flatten vector storage with quantization.
 *
 * FlattenInterface provides operations for storing quantized vectors in a
 * contiguous (flattened) format. It supports training, insertion, querying,
 * and serialization of quantized vector data.
 */
class FlattenInterface {
public:
    FlattenInterface() = default;

    /**
     * @brief Creates a FlattenInterface instance based on parameters.
     *
     * @param param Flatten interface configuration parameters.
     * @param common_param Common index parameters.
     * @return Shared pointer to the created FlattenInterface instance.
     */
    static FlattenInterfacePtr
    MakeInstance(const FlattenInterfaceParamPtr& param, const IndexCommonParam& common_param);

public:
    /**
     * @brief Computes distances between a query and multiple vectors.
     *
     * @param result_dists Output array for distance results.
     * @param computer The computer interface for distance computation.
     * @param idx Array of internal IDs to query.
     * @param id_count Number of IDs in the array.
     * @param ctx Query context for optimization hints.
     */
    virtual void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          QueryContext* ctx = nullptr) = 0;

    /**
     * @brief Creates a computer for distance computation.
     *
     * @param query The query vector data.
     * @return Shared pointer to the computer interface.
     */
    virtual ComputerInterfacePtr
    FactoryComputer(const void* query) = 0;

    /**
     * @brief Trains the quantizer with sample data.
     *
     * @param data Pointer to training data.
     * @param count Number of training vectors.
     */
    virtual void
    Train(const void* data, uint64_t count) = 0;

    /**
     * @brief Inserts a single vector into the flatten storage.
     *
     * @param vector Pointer to the vector data.
     * @param idx Internal ID for the vector (default: max value for auto-assign).
     */
    virtual void
    InsertVector(const void* vector, InnerIdType idx = std::numeric_limits<InnerIdType>::max()) = 0;

    /**
     * @brief Updates a vector at the given index.
     *
     * @param vector Pointer to the new vector data.
     * @param idx Internal ID of the vector to update.
     * @return True if update succeeded.
     */
    virtual bool
    UpdateVector(const void* vector, InnerIdType idx = std::numeric_limits<InnerIdType>::max()) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "UpdateVector not implemented in FlattenInterface");
    };

    /**
     * @brief Inserts multiple vectors in batch.
     *
     * @param vectors Pointer to the vector array.
     * @param count Number of vectors to insert.
     * @param idx_vec Optional array of internal IDs (nullptr for auto-assign).
     */
    virtual void
    BatchInsertVector(const void* vectors, InnerIdType count, InnerIdType* idx_vec = nullptr) = 0;

    /**
     * @brief Computes distance between two stored vectors.
     *
     * @param id1 Internal ID of the first vector.
     * @param id2 Internal ID of the second vector.
     * @return Computed distance value.
     */
    virtual float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) = 0;

    /**
     * @brief Compares two stored vectors for equality.
     *
     * @param id1 Internal ID of the first vector.
     * @param id2 Internal ID of the second vector.
     * @return True if vectors are equal.
     */
    bool
    CompareVectors(InnerIdType id1, InnerIdType id2) {
        bool release1, release2;
        const auto* codes1 = this->GetCodesById(id1, release1);
        const auto* codes2 = this->GetCodesById(id2, release2);
        bool result = (std::memcmp(codes1, codes2, this->code_size_) == 0);
        if (release1) {
            this->Release(codes1);
        }
        if (release2) {
            this->Release(codes2);
        }
        return result;
    }

    /**
     * @brief Prefetches data for a given ID for cache optimization.
     *
     * @param id The internal ID to prefetch.
     */
    virtual void
    Prefetch(InnerIdType id) = 0;

    /**
     * @brief Gets the name of the quantizer used.
     *
     * @return Quantizer name string.
     */
    [[nodiscard]] virtual std::string
    GetQuantizerName() = 0;

    /**
     * @brief Gets the metric type used for distance computation.
     *
     * @return The metric type.
     */
    [[nodiscard]] virtual MetricType
    GetMetricType() = 0;

    /**
     * @brief Resizes the storage to a new capacity.
     *
     * @param capacity The new capacity.
     */
    virtual void
    Resize(InnerIdType capacity) = 0;

    /**
     * @brief Exports the quantization model to another instance.
     *
     * @param other The target flatten interface to export to.
     */
    virtual void
    ExportModel(const FlattenInterfacePtr& other) const = 0;

    /**
     * @brief Initializes IO resources for disk-based storage.
     *
     * @param io_param IO configuration parameters.
     */
    virtual void
    InitIO(const IOParamPtr& io_param) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "InitIO not implemented in FlattenInterface");
    }

    /**
     * @brief Gets the memory usage of the flatten storage.
     *
     * @return Memory usage in bytes.
     */
    virtual int64_t
    GetMemoryUsage() const {
        return 0;
    }

    /**
     * @brief Exports common parameters used by this instance.
     *
     * @return The common parameters.
     */
    virtual IndexCommonParam
    ExportCommonParam() {
        throw VsagException(ErrorType::INTERNAL_ERROR, "ExportCommonParam is not implemented");
    }

public:
    /**
     * @brief Sets runtime parameters for optimization.
     *
     * @param new_params Map of parameter names to values.
     * @return True if any parameter was updated.
     */
    virtual bool
    SetRuntimeParameters(const UnorderedMap<std::string, float>& new_params) {
        bool ret = false;
        auto iter = new_params.find(PREFETCH_STRIDE_CODE);
        if (iter != new_params.end()) {
            prefetch_stride_code_ = static_cast<uint32_t>(iter->second);
            ret = true;
        }

        iter = new_params.find(PREFETCH_DEPTH_CODE);
        if (iter != new_params.end()) {
            prefetch_depth_code_ = static_cast<uint32_t>(iter->second);
            ret = true;
        }

        return ret;
    }

    /**
     * @brief Decodes quantized codes to a float vector.
     *
     * @param codes Pointer to the quantized codes.
     * @param vector Output array for the decoded vector.
     * @return True if decoding succeeded.
     */
    virtual bool
    Decode(const uint8_t* codes, DataType* vector) = 0;

    /**
     * @brief Encodes a float vector to quantized codes.
     *
     * @param vector Input vector to encode.
     * @param codes Output array for quantized codes.
     * @return True if encoding succeeded.
     */
    virtual bool
    Encode(const DataType* vector, uint8_t* codes) = 0;

    /**
     * @brief Gets quantized codes by internal ID.
     *
     * @param id The internal ID to query.
     * @param need_release Output flag indicating if Release() must be called.
     * @return Pointer to the quantized codes.
     */
    [[nodiscard]] virtual const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const = 0;

    /**
     * @brief Releases resources obtained from GetCodesById.
     *
     * @param data Pointer to the data to release.
     */
    virtual void
    Release(const uint8_t* data) const = 0;

    /**
     * @brief Copies quantized codes by internal ID.
     *
     * @param id The internal ID to query.
     * @param codes Output buffer for the quantized codes.
     * @return True if retrieval succeeded.
     */
    virtual bool
    GetCodesById(InnerIdType id, uint8_t* codes) const = 0;

    /**
     * @brief Gets the total count of stored vectors.
     *
     * @return Total count of vectors.
     */
    [[nodiscard]] virtual InnerIdType
    TotalCount() const {
        std::shared_lock lock(mutex_);
        return this->total_count_;
    }

    /**
     * @brief Serializes the flatten storage to a stream writer.
     *
     * @param writer The stream writer for output.
     */
    virtual void
    Serialize(StreamWriter& writer) {
        StreamWriter::WriteObj(writer, this->total_count_);
        StreamWriter::WriteObj(writer, this->max_capacity_);
        StreamWriter::WriteObj(writer, this->code_size_);
    }

    /**
     * @brief Deserializes the flatten storage from a stream reader.
     *
     * @param reader The stream reader for input.
     */
    virtual void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) {
        StreamReader::ReadObj(reader, this->total_count_);
        StreamReader::ReadObj(reader, this->max_capacity_);
        StreamReader::ReadObj(reader, this->code_size_);
    }

    /**
     * @brief Calculates the serialized size of the flatten storage.
     *
     * @return Size in bytes needed for serialization.
     */
    uint64_t
    CalcSerializeSize() {
        auto calSizeFunc = [](uint64_t cursor, uint64_t size, void* buf) { return; };
        WriteFuncStreamWriter writer(calSizeFunc, 0);
        this->Serialize(writer);
        return writer.cursor_;
    }

    /**
     * @brief Checks if the data is stored in memory.
     *
     * @return True if in-memory, false otherwise.
     */
    [[nodiscard]] virtual bool
    InMemory() const {
        return true;
    }

    /**
     * @brief Checks if the instance holds quantization molds (codebooks).
     *
     * @return True if molds are held, false otherwise.
     */
    [[nodiscard]] virtual bool
    HoldMolds() const {
        return false;
    }

    /**
     * @brief Merges another flatten storage into this one.
     *
     * @param other The flatten storage to merge.
     * @param bias ID offset for the merged vectors.
     */
    virtual void
    MergeOther(const FlattenInterfacePtr& other, InnerIdType bias) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "MergeOther not implemented");
    }

public:
    /// Mutex for thread-safe access
    mutable std::shared_mutex mutex_;

    /// Total count of stored vectors
    InnerIdType total_count_{0};
    /// Maximum capacity of the storage
    InnerIdType max_capacity_{800};
    /// Size of each quantized code in bytes
    uint32_t code_size_{0};
    /// Stride for prefetching codes
    uint32_t prefetch_stride_code_{1};
    /// Depth for prefetching codes
    uint32_t prefetch_depth_code_{1};
};

}  // namespace vsag