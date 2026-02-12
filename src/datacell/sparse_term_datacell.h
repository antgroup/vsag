
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
class SparseTermDataCell {
public:
    SparseTermDataCell() = default;

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

    void
    Query(float* global_dists, const SparseTermComputerPtr& computer) const;

    /**
     * @brief Insert candidates into heap by iterating through term lists
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
     * @brief Insert candidates into heap directly from precomputed distance array
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

    void
    DocPrune(Vector<std::pair<uint32_t, float>>& sorted_base) const;

    void
    InsertVector(const SparseVector& sparse_base, uint16_t base_id);

    void
    ResizeTermList(InnerIdType new_term_capacity);

    void
    Serialize(StreamWriter& writer) const;

    void
    Deserialize(StreamReader& reader);

    float
    CalcDistanceByInnerId(const SparseTermComputerPtr& computer, uint16_t base_id);

    void
    Encode(float val, uint8_t* dst) const;

    void
    Decode(const uint8_t* src, size_t size, float* dst) const;

    void
    GetSparseVector(uint32_t base_id, SparseVector* data, Allocator* specified_allocator);

    [[nodiscard]] int64_t
    GetMemoryUsage() const;

private:
    template <InnerSearchMode mode, InnerSearchType type>
    void
    insert_candidate_into_heap(uint32_t id,
                               float& dist,
                               float& cur_heap_top,
                               MaxHeap& heap,
                               uint32_t offset_id,
                               float radius,
                               const FilterPtr& filter) const;

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
    uint32_t term_id_limit_{0};

    float doc_retain_ratio_{0};

    uint32_t term_capacity_{0};

    Vector<std::unique_ptr<Vector<uint16_t>>> term_ids_;

    Vector<std::unique_ptr<Vector<uint8_t>>> term_datas_;

    Vector<uint32_t> term_sizes_;

    Allocator* const allocator_{nullptr};

    bool use_quantization_{false};

    int64_t total_count_{0};

    std::shared_ptr<QuantizationParams> quantization_params_;
};
}  // namespace vsag
