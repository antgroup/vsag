
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

#include <algorithm>
#include <limits>

#include "algorithm/sindi/sindi_parameter.h"
#include "impl/searcher/basic_searcher.h"
#include "quantization/sparse_quantization//sparse_term_computer.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/pointer_define.h"
#include "vsag/allocator.h"
#include "vsag/dataset.h"

namespace vsag {

class SparseEvaluationTracker {
public:
    SparseEvaluationTracker(uint32_t capacity, Allocator* allocator)
        : generations_(capacity, 0, allocator) {
    }

    void
    BeginWindow() {
        if (generation_ == std::numeric_limits<uint16_t>::max()) {
            std::fill(generations_.begin(), generations_.end(), 0);
            generation_ = 1;
        } else {
            ++generation_;
        }
        evaluated_ = 0;
    }

    void
    Mark(const uint16_t* ids, uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) {
            auto id = ids[i];
            if (generations_[id] != generation_) {
                generations_[id] = generation_;
                ++evaluated_;
            }
        }
    }

    [[nodiscard]] uint64_t
    Count() const {
        return evaluated_;
    }

private:
    Vector<uint16_t> generations_;
    uint16_t generation_{0};
    uint64_t evaluated_{0};
};

DEFINE_POINTER(SparseTermDataCell);
class SparseTermDataCell {
public:
    SparseTermDataCell() = default;

    SparseTermDataCell(float doc_retain_ratio,
                       uint32_t term_id_limit,
                       Allocator* allocator,
                       SparseValueQuantizationType sparse_value_quant_type,
                       std::shared_ptr<QuantizationParams> quantization_params)
        : doc_retain_ratio_(doc_retain_ratio),
          term_id_limit_(term_id_limit),
          allocator_(allocator),
          term_ids_(allocator),
          term_datas_(allocator),
          term_sizes_(allocator),
          sparse_value_quant_type_(sparse_value_quant_type),
          quantization_params_(std::move(quantization_params)) {
    }

    void
    Query(float* global_dists, const SparseTermComputerPtr& computer) const;

    uint64_t
    Query(float* global_dists,
          const SparseTermComputerPtr& computer,
          SparseEvaluationTracker& evaluation_tracker) const;

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
    bool
    InsertHeapByTermLists(float* dists,
                          const SparseTermComputerPtr& computer,
                          MaxHeap& heap,
                          const InnerSearchParam& param,
                          uint32_t offset_id,
                          const uint64_t* filter_callback_limit = nullptr) const;

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
    bool
    InsertHeapByDists(float* dists,
                      uint32_t dists_size,
                      MaxHeap& heap,
                      const InnerSearchParam& param,
                      uint32_t offset_id,
                      const uint64_t* filter_callback_limit = nullptr) const;

    void
    DocPrune(Vector<std::pair<uint32_t, float>>& sorted_base) const;

    void
    InsertVector(const SparseVector& sparse_base, uint16_t base_id);

    void
    SortByValue();

    static void
    SortPostingListByValue(uint16_t* ids,
                           uint8_t* data,
                           uint32_t posting_count,
                           SparseValueQuantizationType quantization_type,
                           Vector<uint32_t>& order,
                           Vector<uint16_t>& sorted_ids,
                           Vector<uint8_t>& sorted_data);

    void
    ResizeTermList(InnerIdType new_term_capacity);

    void
    Compact();

    void
    Serialize(StreamWriter& writer) const;

    void
    Deserialize(StreamReader& reader, bool postings_sorted = false);

    float
    CalcDistanceByInnerId(const SparseTermComputerPtr& computer, uint16_t base_id);

    void
    Encode(float val, uint8_t* dst) const;

    void
    Decode(const uint8_t* src, size_t size, float* dst) const;

    static uint32_t
    GetSparseValueCodeSize(SparseValueQuantizationType type);

    void
    GetSparseVector(uint32_t base_id, SparseVector* data, Allocator* specified_allocator);

    [[nodiscard]] uint64_t
    GetMemoryUsage() const;

private:
    void
    query_impl(float* global_dists,
               const SparseTermComputerPtr& computer,
               SparseEvaluationTracker* evaluation_tracker) const;

    template <InnerSearchMode mode, InnerSearchType type>
    void
    insert_candidate_into_heap(uint32_t id,
                               float& dist,
                               float& cur_heap_top,
                               MaxHeap& heap,
                               uint32_t offset_id,
                               uint32_t n_candidate,
                               float radius,
                               const FilterPtr& filter,
                               const std::optional<float>& threshold,
                               bool enable_reorder) const;

    template <InnerSearchType type>
    bool
    fill_heap_initial(uint32_t id,
                      float& dist,
                      float& cur_heap_top,
                      MaxHeap& heap,
                      uint32_t offset_id,
                      uint32_t n_candidate,
                      const FilterPtr& filter,
                      const std::optional<float>& threshold,
                      bool enable_reorder) const;

public:
    uint32_t term_id_limit_{0};

    float doc_retain_ratio_{0};

    uint32_t term_capacity_{0};

    Vector<std::unique_ptr<Vector<uint16_t>>> term_ids_;

    Vector<std::unique_ptr<Vector<uint8_t>>> term_datas_;

    Vector<uint32_t> term_sizes_;

    Allocator* const allocator_{nullptr};

    SparseValueQuantizationType sparse_value_quant_type_{SparseValueQuantizationType::FP32};

    int64_t total_count_{0};

    std::shared_ptr<QuantizationParams> quantization_params_;
};
}  // namespace vsag
