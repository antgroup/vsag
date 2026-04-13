// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with License.
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
 * @file sparse_vector_datacell.h
 * @brief Sparse vector data cell for storing and querying sparse vectors.
 */

#pragma once

#include "algorithm/inner_index_interface.h"
#include "flatten_interface.h"
#include "io/basic_io.h"
#include "io/memory_block_io.h"
#include "vsag/dataset.h"

namespace vsag {

/**
 * @brief Sparse vector data cell implementation for sparse vector storage and retrieval.
 *
 * This class provides functionality for storing sparse vectors with quantization support.
 * It inherits from FlattenInterface and implements all required interfaces for vector
 * operations including insertion, query, serialization, and deserialization.
 *
 * @tparam QuantTmpl Quantizer type for sparse vector quantization
 * @tparam IOTmpl IO type for data storage
 */
template <typename QuantTmpl, typename IOTmpl>
class SparseVectorDataCell : public FlattenInterface {
public:
    SparseVectorDataCell() = default;

    /**
     * @brief Constructs a sparse vector data cell with parameters.
     *
     * @param quantization_param Quantizer parameters for sparse vector quantization
     * @param io_param IO parameters for data storage
     * @param common_param Common index parameters
     */
    SparseVectorDataCell(const QuantizerParamPtr& quantization_param,
                         const IOParamPtr& io_param,
                         const IndexCommonParam& common_param);

    /**
     * @brief Performs query operation to compute distances.
     *
     * @param result_dists Output array for distance results
     * @param computer Computer interface for distance computation
     * @param idx Array of inner IDs to query
     * @param id_count Number of IDs in the array
     * @param ctx Query context (optional)
     */
    void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          QueryContext* ctx = nullptr) override {
        auto comp = std::static_pointer_cast<Computer<QuantTmpl>>(computer);
        this->query(result_dists, comp, idx, id_count);
    }

    /**
     * @brief Creates a computer for distance computation.
     *
     * @param query Query vector data
     * @return Computer interface pointer for distance computation
     */
    ComputerInterfacePtr
    FactoryComputer(const void* query) override {
        return this->factory_computer(static_cast<const float*>(query));
    }

    /**
     * @brief Decodes quantized codes to original vector data.
     *
     * @param codes Quantized code data
     * @param vector Output vector data
     * @return True if decoding succeeds
     * @throws VsagException Always thrown as not implemented for sparse vectors
     */
    bool
    Decode(const uint8_t* codes, DataType* vector) override {
        // TODO(inabao): Implement the decode function
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Decode function is not implemented for SparseVectorDataCell");
    }

    /**
     * @brief Encodes vector data to quantized codes.
     *
     * @param vector Input vector data
     * @param codes Output quantized code data
     * @return True if encoding succeeds
     * @throws VsagException Always thrown as not implemented for sparse vectors
     */
    bool
    Encode(const DataType* vector, uint8_t* codes) override {
        // TODO(inabao): Implement the decode function
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Encode function is not implemented for SparseVectorDataCell");
    }

    /**
     * @brief Computes distance between two vectors by their IDs.
     *
     * @param id1 First vector's inner ID
     * @param id2 Second vector's inner ID
     * @return Computed distance between the two vectors
     */
    float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) override;

    /**
     * @brief Trains the quantizer with provided data.
     *
     * @param data Training data
     * @param count Number of training vectors
     */
    void
    Train(const void* data, uint64_t count) override;

    /**
     * @brief Inserts a single vector into the data cell.
     *
     * @param vector Vector data to insert
     * @param idx Inner ID for the inserted vector
     */
    void
    InsertVector(const void* vector, InnerIdType idx) override;

    /**
     * @brief Inserts multiple vectors in batch.
     *
     * @param vectors Array of vectors to insert
     * @param count Number of vectors
     * @param idx_vec Array of inner IDs for the vectors
     */
    void
    BatchInsertVector(const void* vectors, InnerIdType count, InnerIdType* idx_vec) override;

    /**
     * @brief Resizes the data cell to a new capacity.
     *
     * @param new_capacity New maximum capacity for the data cell
     */
    void
    Resize(InnerIdType new_capacity) override {
        if (new_capacity <= this->max_capacity_) {
            return;
        }
        uint64_t io_size = (new_capacity - total_count_) * max_code_size_ + current_offset_;
        this->io_->Resize(io_size);
        this->offset_io_->Resize(static_cast<uint64_t>(new_capacity) * sizeof(uint32_t));
        this->max_capacity_ = new_capacity;
    }

    /**
     * @brief Prefetches data for the given ID.
     *
     * @param id Inner ID to prefetch
     */
    void
    Prefetch(InnerIdType id) override{};

    /**
     * @brief Exports quantizer model to another data cell.
     *
     * @param other Target data cell to export model to
     * @throws VsagException if other is not a valid SparseVectorDataCell
     */
    void
    ExportModel(const FlattenInterfacePtr& other) const override {
        std::stringstream ss;
        IOStreamWriter writer(ss);
        this->quantizer_->Serialize(writer);
        ss.seekg(0, std::ios::beg);
        IOStreamReader reader(ss);
        auto ptr = std::dynamic_pointer_cast<FlattenDataCell<QuantTmpl, IOTmpl>>(other);
        if (ptr == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "Export model's sparse flatten datacell failed");
        }
        ptr->quantizer_->Deserialize(reader);
    }

    /**
     * @brief Gets the quantizer name.
     *
     * @return Name of the quantizer
     */
    [[nodiscard]] std::string
    GetQuantizerName() override;

    /**
     * @brief Gets the metric type used for distance computation.
     *
     * @return Metric type
     */
    [[nodiscard]] MetricType
    GetMetricType() override;

    /**
     * @brief Gets the codes for a given ID.
     *
     * @param id Inner ID to retrieve codes for
     * @param need_release Output flag indicating if the returned data needs release
     * @return Pointer to the codes data
     */
    [[nodiscard]] const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const override;

    /**
     * @brief Releases previously obtained codes data.
     *
     * @param data Pointer to data to release
     */
    void
    Release(const uint8_t* data) const override;

    /**
     * @brief Checks if the data cell is stored in memory.
     *
     * @return True if data is stored in memory
     */
    [[nodiscard]] bool
    InMemory() const override;

    /**
     * @brief Gets codes for a given ID and copies to output buffer.
     *
     * @param id Inner ID to retrieve codes for
     * @param codes Output buffer to copy codes to
     * @return True if retrieval succeeds
     */
    bool
    GetCodesById(InnerIdType id, uint8_t* codes) const override;

    /**
     * @brief Serializes the data cell to a stream.
     *
     * @param writer Stream writer for serialization
     */
    void
    Serialize(StreamWriter& writer) override;

    /**
     * @brief Deserializes the data cell from a stream.
     *
     * @param reader Stream reader for deserialization
     */
    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override;

    /**
     * @brief Sets the quantizer for the data cell.
     *
     * @param quantizer Quantizer to set
     */
    inline void
    SetQuantizer(std::shared_ptr<Quantizer<QuantTmpl>> quantizer) {
        this->quantizer_ = quantizer;
    }

    /**
     * @brief Sets the IO for the data cell.
     *
     * @param io IO to set
     */
    inline void
    SetIO(std::shared_ptr<BasicIO<IOTmpl>> io) {
        this->io_ = io;
    }

    /**
     * @brief Gets the memory usage of the data cell.
     *
     * @return Memory usage in bytes
     */
    int64_t
    GetMemoryUsage() const override;

private:
    /**
     * @brief Internal query implementation.
     *
     * @param result_dists Output array for distance results
     * @param computer Computer for distance computation
     * @param idx Array of inner IDs to query
     * @param id_count Number of IDs in the array
     */
    inline void
    query(float* result_dists,
          const std::shared_ptr<Computer<QuantTmpl>>& computer,
          const InnerIdType* idx,
          InnerIdType id_count);

    /**
     * @brief Creates a computer for the query.
     *
     * @param query Query vector data
     * @return Computer interface pointer
     */
    ComputerInterfacePtr
    factory_computer(const float* query) {
        auto computer = this->quantizer_->FactoryComputer();
        computer->SetQuery(query);
        return computer;
    }

private:
    std::shared_ptr<Quantizer<QuantTmpl>> quantizer_{nullptr};  ///< Quantizer for sparse vectors
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};              ///< IO for data storage

    Allocator* const allocator_{nullptr};                ///< Memory allocator
    std::shared_ptr<MemoryBlockIO> offset_io_{nullptr};  ///< IO for offset storage
    uint32_t current_offset_{0};                         ///< Current write offset
    uint64_t max_code_size_{0};                          ///< Maximum code size per vector
    std::mutex current_offset_mutex_;                    ///< Mutex for offset operations
};

}  // namespace vsag

#include "sparse_vector_datacell.inl"