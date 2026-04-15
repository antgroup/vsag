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
 * @file sparse_term_datacell.h
 * @brief Sparse term data cell for managing term-based inverted index structure.
 */

#pragma once

#include "algorithm/sindi/sindi_parameter.h"
#include "impl/searcher/basic_searcher.h"
#include "quantization/sparse_quantization//sparse_term_computer.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/pointer_define.h"
#include "vsag/allocator.h"
#include "vsag/dataset.h"

namespace vsag {

DEFINE_POINTER(SparseTermDataCell);

/**
 * @brief Sparse term data cell for term-based inverted index management.
 *
 * This class manages the inverted index structure for sparse vectors,
 * supporting term-based query operations, quantization, and memory-efficient
 * storage of term lists with associated document IDs and data.
 */
class SparseTermDataCell {
public:
    SparseTermDataCell() = default;

    /**
     * @brief Constructs a sparse term data cell with parameters.
     *
     * @param doc_retain_ratio Ratio of documents to retain during pruning
     * @param term_id_limit Maximum term ID limit
     * @param allocator Memory allocator
     * @param use_quantization Whether to use quantization for term data
     * @param quantization_params Quantization parameters
     */
    SparseTermDataCell(float doc_retain_ratio,
                       uint32_t term_id_limit,
                       Allocator* allocator,
                       bool use_quantization,
                       std::shared_ptr<QuantizationParams> quantization_params)
        : doc_retain_ratio_(doc_retain_ratio),
          term_id_limit_(term_id_limit),
          allocator_(allocator),
          term_ids_(allocator),
          term_datas_(allocator),
          term_sizes_(allocator),
          use_quantization_(use_quantization),
          quantization_params_(std::move(quantization_params)) {
    }

    /**
     * @brief Performs query operation using term computer.
     *
     * @param global_dists Output array for global distances
     * @param computer Sparse term computer for query processing
     */
    void
    Query(float* global_dists, const SparseTermComputerPtr& computer) const;

    /**
     * @brief Inserts candidates into heap by iterating through term lists.
     *
     * @param dists Pre-allocated distance array (will be modified during processing)
     * @param computer SparseTermComputer for iterating through terms
     * @param heap MaxHeap to store candidate results
     * @param param Inner search parameters
     * @param offset_id Offset to add to inner IDs when inserting into heap
     */
    template <InnerSearchMode mode = InnerSearchMode::KNN_SEARCH,
              InnerSearchType type = InnerSearchType::PURE>
    void
    InsertHeapByTermLists(float* dists,
                          const SparseTermComputerPtr& computer,
                          MaxHeap& heap,
                          const InnerSearchParam& param,
                          uint32_t offset_id) const;

    /**
     * @brief Inserts candidates into heap directly from precomputed distance array.
     *
     * @param dists Precomputed distance array (will be modified during processing)
     * @param dists_size Size of the distance array
     * @param heap MaxHeap to store candidate results
     * @param param Inner search parameters
     * @param offset_id Offset to add to inner IDs when inserting into heap
     */
    template <InnerSearchMode mode = InnerSearchMode::KNN_SEARCH,
              InnerSearchType type = InnerSearchType::PURE>
    void
    InsertHeapByDists(float* dists,
                      uint32_t dists_size,
                      MaxHeap& heap,
                      const InnerSearchParam& param,
                      uint32_t offset_id) const;

    /**
     * @brief Prunes documents based on sorted base list.
     *
     * @param sorted_base Sorted list of document ID and value pairs
     */
    void
    DocPrune(Vector<std::pair<uint32_t, float>>& sorted_base) const;

    /**
     * @brief Inserts a sparse vector into the term data cell.
     *
     * @param sparse_base Sparse vector to insert
     * @param base_id Document ID for the vector
     */
    void
    InsertVector(const SparseVector& sparse_base, uint16_t base_id);

    /**
     * @brief Resizes the term list to a new capacity.
     *
     * @param new_term_capacity New capacity for term lists
     */
    void
    ResizeTermList(InnerIdType new_term_capacity);

    /**
     * @brief Serializes the term data cell to a stream.
     *
     * @param writer Stream writer for serialization
     */
    void
    Serialize(StreamWriter& writer) const;

    /**
     * @brief Deserializes the term data cell from a stream.
     *
     * @param reader Stream reader for deserialization
     */
    void
    Deserialize(StreamReader& reader);

    /**
     * @brief Calculates distance by inner ID using term computer.
     *
     * @param computer Sparse term computer for distance computation
     * @param base_id Document ID to compute distance for
     * @return Computed distance value
     */
    float
    CalcDistanceByInnerId(const SparseTermComputerPtr& computer, uint16_t base_id);

    /**
     * @brief Encodes a float value to quantized bytes.
     *
     * @param val Value to encode
     * @param dst Destination buffer for encoded bytes
     */
    void
    Encode(float val, uint8_t* dst) const;

    /**
     * @brief Decodes quantized bytes to float values.
     *
     * @param src Source buffer with encoded bytes
     * @param size Number of values to decode
     * @param dst Destination buffer for decoded floats
     */
    void
    Decode(const uint8_t* src, size_t size, float* dst) const;

    /**
     * @brief Retrieves a sparse vector by document ID.
     *
     * @param base_id Document ID to retrieve
     * @param data Output sparse vector
     * @param specified_allocator Allocator to use for the output vector
     */
    void
    GetSparseVector(uint32_t base_id, SparseVector* data, Allocator* specified_allocator);

    /**
     * @brief Gets the memory usage of the term data cell.
     *
     * @return Memory usage in bytes
     */
    [[nodiscard]] int64_t
    GetMemoryUsage() const;

private:
    /**
     * @brief Inserts a candidate into the heap with filtering.
     *
     * @param id Candidate ID
     * @param dist Candidate distance (modified during processing)
     * @param cur_heap_top Current heap top distance (modified during processing)
     * @param heap MaxHeap to insert into
     * @param offset_id Offset to add to the ID
     * @param radius Radius threshold for range search
     * @param filter Filter for candidate acceptance
     */
    template <InnerSearchMode mode, InnerSearchType type>
    void
    insert_candidate_into_heap(uint32_t id,
                               float& dist,
                               float& cur_heap_top,
                               MaxHeap& heap,
                               uint32_t offset_id,
                               float radius,
                               const FilterPtr& filter) const;

    /**
     * @brief Fills the heap with initial candidates.
     *
     * @param id Candidate ID
     * @param dist Candidate distance (modified during processing)
     * @param cur_heap_top Current heap top distance (modified during processing)
     * @param heap MaxHeap to fill
     * @param offset_id Offset to add to the ID
     * @param n_candidate Number of candidates to fill
     * @param filter Filter for candidate acceptance
     * @return True if heap was filled successfully
     */
    template <InnerSearchType type>
    bool
    fill_heap_initial(uint32_t id,
                      float& dist,
                      float& cur_heap_top,
                      MaxHeap& heap,
                      uint32_t offset_id,
                      uint32_t n_candidate,
                      const FilterPtr& filter) const;

public:
    uint32_t term_id_limit_{0};  ///< Maximum term ID limit

    float doc_retain_ratio_{0};  ///< Document retention ratio for pruning

    uint32_t term_capacity_{0};  ///< Current capacity for term lists

    /// Term IDs for each term (inverted index)
    Vector<std::unique_ptr<Vector<uint16_t>>> term_ids_;

    /// Term data (values) for each term
    Vector<std::unique_ptr<Vector<uint8_t>>> term_datas_;

    /// Size of each term list
    Vector<uint32_t> term_sizes_;

    Allocator* const allocator_{nullptr};  ///< Memory allocator

    bool use_quantization_{false};  ///< Whether quantization is enabled

    int64_t total_count_{0};  ///< Total count of inserted vectors

    std::shared_ptr<QuantizationParams> quantization_params_;  ///< Quantization parameters
};
}  // namespace vsag