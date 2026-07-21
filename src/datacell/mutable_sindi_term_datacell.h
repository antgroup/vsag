
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
#include "datacell/sindi_datacell_utils.h"
#include "datacell/sindi_term_datacell.h"
#include "impl/searcher/basic_searcher.h"
#include "quantization/sparse_quantization/sparse_term_computer.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/pointer_define.h"
#include "vsag/allocator.h"
#include "vsag/dataset.h"

namespace vsag {

struct MutableSINDIWindow {
    explicit MutableSINDIWindow(Allocator* allocator = nullptr)
        : term_ids_(allocator), term_datas_(allocator), term_sizes_(allocator) {
    }

    uint32_t term_capacity_{0};
    Vector<std::unique_ptr<Vector<uint16_t>>> term_ids_;
    Vector<std::unique_ptr<Vector<uint8_t>>> term_datas_;
    Vector<uint32_t> term_sizes_;
};

DEFINE_POINTER(MutableSindiTermDataCell);
class MutableSindiTermDataCell : public SindiTermDataCell {
public:
    MutableSindiTermDataCell(float doc_retain_ratio,
                             uint32_t term_id_limit,
                             uint32_t window_size,
                             Allocator* allocator,
                             SparseValueQuantizationType sparse_value_quant_type,
                             std::shared_ptr<QuantizationParams> quantization_params)
        : term_id_limit_(term_id_limit),
          doc_retain_ratio_(doc_retain_ratio),
          allocator_(allocator),
          sparse_value_quant_type_(sparse_value_quant_type),
          quantization_params_(std::move(quantization_params)),
          window_size_(window_size),
          windows_(allocator) {
    }

    void
    Finalize();

    void
    SerializeWindows(StreamWriter& writer) const;

    void
    DeserializeWindows(StreamReader& reader, uint32_t window_count);

    void
    SerializeTermLayout(StreamWriter& writer, uint32_t term_dict_count) const override;

    void
    DeserializeTermLayout(StreamReader& reader, uint32_t window_count, uint64_t total_count);

    QueryTermBuffers
    LoadQueryTermBuffers(const Vector<uint32_t>& query_term_ids) const override {
        return QueryTermBuffers(allocator_);
    }

    [[nodiscard]] const MutableSINDIWindow&
    GetWindow(uint32_t window_id) const;

    [[nodiscard]] MutableSINDIWindow&
    GetWindow(uint32_t window_id);

    void
    QueryWindow(float* dists,
                uint32_t window_id,
                const SparseTermComputerPtr& computer,
                bool use_term_lists_heap_insert,
                SindiQueryContext& query_context) const override;

    void
    InsertHeapByWindow(float* dists,
                       uint32_t window_id,
                       const SparseTermComputerPtr& computer,
                       MaxHeap& heap,
                       const InnerSearchParam& param,
                       uint32_t offset_id,
                       InnerSearchMode mode,
                       bool with_filter,
                       const SindiQueryContext& query_context) const override;

    void
    InsertHeapByDists(float* dists,
                      uint32_t dists_size,
                      MaxHeap& heap,
                      const InnerSearchParam& param,
                      uint32_t offset_id,
                      InnerSearchMode mode,
                      bool with_filter) const override;

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

    // doc_id is interpreted in the coordinate space owned by this data cell.
    void
    InsertVector(const SparseVector& sparse_base, uint32_t doc_id);

    void
    ResizeTermList(InnerIdType new_term_capacity);

    void
    Compact();

    float
    CalcDistanceByInnerId(const SparseTermComputerPtr& computer,
                          uint32_t base_id,
                          const QueryTermBuffers& query_term_buffers) const override;

    float
    CalcDistanceByInnerId(const SparseTermComputerPtr& computer, uint16_t base_id) const;

    void
    Encode(float val, uint8_t* dst) const;

    void
    Decode(const uint8_t* src, uint64_t size, float* dst) const;

    void
    GetSparseVector(uint32_t base_id,
                    SparseVector* data,
                    Allocator* specified_allocator) const override;

    [[nodiscard]] uint64_t
    GetMemoryUsage() const override;

    [[nodiscard]] uint32_t
    GetWindowCount() const override {
        return static_cast<uint32_t>(windows_.size());
    }

    [[nodiscard]] uint32_t
    GetTermDictCount() const override;

private:
    [[nodiscard]] uint32_t
    GetTermValueCodeSize() const;

    [[nodiscard]] SindiTermPostingView
    GetTermPostingView(uint32_t term_id, uint32_t window_id) const;

    [[nodiscard]] std::vector<sindi_datacell_utils::TermPostingRecord>
    CollectTermPostings() const;

    void
    ResizeTermList(MutableSINDIWindow& window, InnerIdType new_term_capacity);

    void
    Compact(MutableSINDIWindow& window);

    void
    SerializeWindow(StreamWriter& writer, const MutableSINDIWindow& window) const;

    void
    DeserializeWindow(StreamReader& reader, MutableSINDIWindow& window);

    float
    CalcDistanceByInnerId(const MutableSINDIWindow& window,
                          const SparseTermComputerPtr& computer,
                          uint16_t base_id) const;

    void
    GetSparseVector(const MutableSINDIWindow& window,
                    uint32_t base_id,
                    SparseVector* data,
                    Allocator* specified_allocator) const;

    [[nodiscard]] uint64_t
    GetWindowMemoryUsage(const MutableSINDIWindow& window) const;

    template <InnerSearchMode mode, InnerSearchType type>
    void
    InsertHeapByTermLists(const MutableSINDIWindow& window,
                          float* dists,
                          const SparseTermComputerPtr& computer,
                          MaxHeap& heap,
                          const InnerSearchParam& param,
                          uint32_t offset_id) const;

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

    Allocator* const allocator_{nullptr};

    SparseValueQuantizationType sparse_value_quant_type_{SparseValueQuantizationType::FP32};

    int64_t total_count_{0};

    std::shared_ptr<QuantizationParams> quantization_params_;

    uint32_t window_size_{0};

    Vector<MutableSINDIWindow> windows_;
};
}  // namespace vsag
