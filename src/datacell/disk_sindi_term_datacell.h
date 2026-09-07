
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

#include <cmath>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <vector>

#include "container_types.h"
#include "datacell/sindi_search_term_datacell.h"
#include "impl/inner_search_param.h"
#include "index_common_param.h"
#include "io/common/io_parameter.h"
#include "quantization/sparse_quantization/sparse_term_computer.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/pointer_define.h"
#include "vsag/allocator.h"
#include "vsag/dataset.h"
#include "vsag/filter.h"
#include "vsag/readerset.h"

namespace vsag {

using TermBuffer = SindiTermBuffer;

class DiskSindiTermDataCellInterface : public SindiSearchTermDataCell {
public:
    static std::shared_ptr<DiskSindiTermDataCellInterface>
    MakeInstance(uint32_t term_id_limit,
                 Allocator* allocator,
                 SparseValueQuantizationType sparse_value_quant_type,
                 QuantizationParamsPtr quantization_params,
                 uint32_t window_size,
                 const IOParamPtr& io_param,
                 const IndexCommonParam& common_param);

    virtual void
    DeserializeTermLayout(StreamReader& reader, uint32_t window_count, uint64_t total_count) = 0;

    virtual void
    SetIO(const std::shared_ptr<Reader>& reader) = 0;
};

using DiskSindiTermDataCellInterfacePtr = std::shared_ptr<DiskSindiTermDataCellInterface>;

template <typename IOTmpl>
class DiskSindiTermDataCell : public DiskSindiTermDataCellInterface {
public:
    DiskSindiTermDataCell(uint32_t term_id_limit,
                          Allocator* allocator,
                          SparseValueQuantizationType sparse_value_quant_type,
                          QuantizationParamsPtr quantization_params,
                          uint32_t window_size,
                          IOParamPtr io_param,
                          IndexCommonParam common_param);

    void
    SerializeTermLayout(StreamWriter& writer, uint32_t term_dict_count) const override;

    void
    DeserializeTermLayout(StreamReader& reader,
                          uint32_t window_count,
                          uint64_t total_count) override;

    void
    SetIO(const std::shared_ptr<Reader>& reader) override;

    QueryTermBuffers
    LoadQueryTermBuffers(const Vector<uint32_t>& query_term_ids,
                         Allocator* query_allocator = nullptr) const override;

    uint32_t
    GetWindowCount() const override {
        return window_count_;
    }

    uint32_t
    GetTermDictCount() const override {
        return static_cast<uint32_t>(term_dict_.size());
    }

    uint64_t
    GetMemoryUsage() const override;

    void
    QueryWindow(float* dists,
                uint32_t window_id,
                const SparseTermComputerPtr& computer,
                bool use_term_lists_heap_insert,
                SindiQueryContext& query_context) const override;

    bool
    InsertHeapByWindow(float* dists,
                       uint32_t window_id,
                       const SparseTermComputerPtr& computer,
                       MaxHeap& heap,
                       const InnerSearchParam& param,
                       uint32_t offset_id,
                       InnerSearchMode mode,
                       bool with_filter,
                       const SindiQueryContext& query_context,
                       const uint64_t* filter_callback_remaining = nullptr) const override;

    bool
    InsertHeapByDists(float* dists,
                      uint32_t dists_size,
                      MaxHeap& heap,
                      const InnerSearchParam& param,
                      uint32_t offset_id,
                      InnerSearchMode mode,
                      bool with_filter,
                      const uint64_t* filter_callback_remaining = nullptr) const override;

    void
    GetSparseVector(uint32_t inner_id,
                    SparseVector* data,
                    Allocator* specified_allocator) const override;

    float
    CalcDistanceByInnerId(const SparseTermComputerPtr& computer,
                          uint32_t base_id,
                          const QueryTermBuffers& query_term_buffers) const override;

private:
    void
    InitIO(const IOParamPtr& io_param);

    const TermBuffer*
    GetTermBufferNoLock(uint32_t term_id, const QueryTermBuffers& query_term_buffers) const;

    bool
    InsertHeapByWindowKnn(float* dists,
                          uint32_t window_id,
                          const SparseTermComputerPtr& computer,
                          MaxHeap& heap,
                          const InnerSearchParam& param,
                          uint32_t offset_id,
                          bool with_filter,
                          const QueryTermBuffers& query_term_buffers,
                          const uint64_t* filter_callback_remaining) const;

    bool
    InsertHeapByDistsKnn(float* dists,
                         uint32_t dists_size,
                         MaxHeap& heap,
                         const InnerSearchParam& param,
                         uint32_t offset_id,
                         bool with_filter,
                         const uint64_t* filter_callback_remaining) const;

    template <InnerSearchMode mode, InnerSearchType type>
    bool
    InsertHeapByWindow(float* dists,
                       uint32_t window_id,
                       const SparseTermComputerPtr& computer,
                       MaxHeap& heap,
                       const InnerSearchParam& param,
                       uint32_t offset_id,
                       const QueryTermBuffers& query_term_buffers,
                       const uint64_t* filter_callback_remaining = nullptr) const;

    template <InnerSearchMode mode, InnerSearchType type>
    bool
    InsertHeapByDists(float* dists,
                      uint32_t dists_size,
                      MaxHeap& heap,
                      const InnerSearchParam& param,
                      uint32_t offset_id,
                      const uint64_t* filter_callback_remaining = nullptr) const;

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

    void
    ValidateBoundLayout(uint64_t payload_size) const;

    void
    ValidateBoundPayloads() const;

private:
    uint32_t term_id_limit_{0};
    Allocator* allocator_{nullptr};
    SparseValueQuantizationType sparse_value_quant_type_{SparseValueQuantizationType::FP32};
    QuantizationParamsPtr quantization_params_;
    uint32_t window_size_{0};
    IOParamPtr io_param_{nullptr};
    IndexCommonParam common_param_;
    std::shared_ptr<IOTmpl> io_{nullptr};

    mutable std::shared_mutex term_layout_mutex_;
    std::vector<DiskTermEntry> term_dict_;
    uint32_t window_count_{0};
    uint64_t total_count_{0};
    uint64_t payload_size_{0};
    bool payloads_validated_{false};
};

template <typename IOTmpl>
using DiskSindiTermDataCellPtr = std::shared_ptr<DiskSindiTermDataCell<IOTmpl>>;

}  // namespace vsag

#include "disk_sindi_term_datacell.inl"
