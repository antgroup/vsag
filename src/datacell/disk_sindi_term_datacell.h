
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

#include <mutex>
#include <shared_mutex>
#include <vector>

#include "container_types.h"
#include "datacell/sindi_term_datacell.h"
#include "impl/inner_search_param.h"
#include "index_common_param.h"
#include "io/basic_io.h"
#include "io/io_parameter.h"
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

class DiskSindiTermDataCellInterface : public SindiTermDataCell {
public:
    virtual ~DiskSindiTermDataCellInterface() = default;

    static std::shared_ptr<DiskSindiTermDataCellInterface>
    MakeInstance(float doc_retain_ratio,
                 uint32_t term_id_limit,
                 Allocator* allocator,
                 SparseValueQuantizationType sparse_value_quant_type,
                 QuantizationParamsPtr quantization_params,
                 uint32_t window_size,
                 const IOParamPtr& io_param,
                 const IndexCommonParam& common_param);

    static std::shared_ptr<DiskSindiTermDataCellInterface>
    MakeInstance(float doc_retain_ratio,
                 uint32_t term_id_limit,
                 Allocator* allocator,
                 bool use_quantization,
                 QuantizationParamsPtr quantization_params,
                 uint32_t window_size,
                 const IOParamPtr& io_param,
                 const IndexCommonParam& common_param) {
        auto quantization_type =
            use_quantization ? SparseValueQuantizationType::SQ8 : SparseValueQuantizationType::FP32;
        return MakeInstance(doc_retain_ratio,
                            term_id_limit,
                            allocator,
                            quantization_type,
                            std::move(quantization_params),
                            window_size,
                            io_param,
                            common_param);
    }

    virtual void
    DeserializeTermLayout(StreamReader& reader, uint32_t window_count, uint64_t total_count) = 0;

    virtual void
    InitIO(const IOParamPtr& io_param) = 0;

    virtual void
    SetIO(const std::shared_ptr<Reader>& reader) = 0;

    virtual QueryTermBuffers
    LoadQueryTermBuffers(const Vector<uint32_t>& query_term_ids) const override = 0;

    virtual uint32_t
    GetWindowCount() const override = 0;

    virtual uint32_t
    GetTermDictCount() const override = 0;

    virtual uint64_t
    GetMemoryUsage() const override = 0;

    virtual void
    QueryWindow(float* dists,
                uint32_t window_id,
                const SparseTermComputerPtr& computer,
                bool use_term_lists_heap_insert,
                const QueryTermBuffers& query_term_buffers) const override = 0;

    virtual void
    InsertHeapByWindowKnn(float* dists,
                          uint32_t window_id,
                          const SparseTermComputerPtr& computer,
                          MaxHeap& heap,
                          const InnerSearchParam& param,
                          uint32_t offset_id,
                          bool with_filter,
                          const QueryTermBuffers& query_term_buffers) const = 0;

    virtual void
    InsertHeapByDistsKnn(float* dists,
                         uint32_t dists_size,
                         MaxHeap& heap,
                         const InnerSearchParam& param,
                         uint32_t offset_id,
                         bool with_filter) const = 0;

    void
    InsertHeapByWindow(float* dists,
                       uint32_t window_id,
                       const SparseTermComputerPtr& computer,
                       MaxHeap& heap,
                       const InnerSearchParam& param,
                       uint32_t offset_id,
                       InnerSearchMode mode,
                       bool with_filter,
                       const QueryTermBuffers& query_term_buffers) const override = 0;

    void
    InsertHeapByDists(float* dists,
                      uint32_t dists_size,
                      MaxHeap& heap,
                      const InnerSearchParam& param,
                      uint32_t offset_id,
                      InnerSearchMode mode,
                      bool with_filter) const override = 0;

    virtual void
    GetSparseVector(uint32_t inner_id,
                    SparseVector* data,
                    Allocator* specified_allocator) const override = 0;

    virtual float
    CalcDistanceByInnerId(const SparseTermComputerPtr& computer,
                          uint32_t base_id,
                          const QueryTermBuffers& query_term_buffers) const override = 0;
};

using DiskSindiTermDataCellInterfacePtr = std::shared_ptr<DiskSindiTermDataCellInterface>;

template <typename IOTmpl>
class DiskSindiTermDataCell : public DiskSindiTermDataCellInterface {
public:
    DiskSindiTermDataCell(float doc_retain_ratio,
                          uint32_t term_id_limit,
                          Allocator* allocator,
                          SparseValueQuantizationType sparse_value_quant_type,
                          QuantizationParamsPtr quantization_params,
                          uint32_t window_size,
                          IOParamPtr io_param,
                          const IndexCommonParam& common_param);

    void
    SerializeTermLayout(StreamWriter& writer, uint32_t term_dict_count) const override;

    void
    DeserializeTermLayout(StreamReader& reader,
                          uint32_t window_count,
                          uint64_t total_count) override;

    void
    InitIO(const IOParamPtr& io_param) override;

    void
    SetIO(const std::shared_ptr<Reader>& reader) override;

    QueryTermBuffers
    LoadQueryTermBuffers(const Vector<uint32_t>& query_term_ids) const override;

    const TermBuffer*
    GetTermBufferNoLock(uint32_t term_id, const QueryTermBuffers& query_term_buffers) const;

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
                const QueryTermBuffers& query_term_buffers) const override;

    void
    InsertHeapByWindowKnn(float* dists,
                          uint32_t window_id,
                          const SparseTermComputerPtr& computer,
                          MaxHeap& heap,
                          const InnerSearchParam& param,
                          uint32_t offset_id,
                          bool with_filter,
                          const QueryTermBuffers& query_term_buffers) const override;

    void
    InsertHeapByDistsKnn(float* dists,
                         uint32_t dists_size,
                         MaxHeap& heap,
                         const InnerSearchParam& param,
                         uint32_t offset_id,
                         bool with_filter) const override;

    void
    InsertHeapByWindow(float* dists,
                       uint32_t window_id,
                       const SparseTermComputerPtr& computer,
                       MaxHeap& heap,
                       const InnerSearchParam& param,
                       uint32_t offset_id,
                       InnerSearchMode mode,
                       bool with_filter,
                       const QueryTermBuffers& query_term_buffers) const override;

    void
    InsertHeapByDists(float* dists,
                      uint32_t dists_size,
                      MaxHeap& heap,
                      const InnerSearchParam& param,
                      uint32_t offset_id,
                      InnerSearchMode mode,
                      bool with_filter) const override;

    template <InnerSearchMode mode, InnerSearchType type>
    void
    InsertHeapByWindow(float* dists,
                       uint32_t window_id,
                       const SparseTermComputerPtr& computer,
                       MaxHeap& heap,
                       const InnerSearchParam& param,
                       uint32_t offset_id,
                       const QueryTermBuffers& query_term_buffers) const;

    template <InnerSearchMode mode, InnerSearchType type>
    void
    InsertHeapByDists(float* dists,
                      uint32_t dists_size,
                      MaxHeap& heap,
                      const InnerSearchParam& param,
                      uint32_t offset_id) const;

    void
    GetSparseVector(uint32_t inner_id,
                    SparseVector* data,
                    Allocator* specified_allocator) const override;

    float
    CalcDistanceByInnerId(const SparseTermComputerPtr& computer,
                          uint32_t base_id,
                          const QueryTermBuffers& query_term_buffers) const override;

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

    void
    ValidateBoundLayout(uint64_t payload_size) const;

private:
    uint32_t term_id_limit_{0};
    Allocator* allocator_{nullptr};
    SparseValueQuantizationType sparse_value_quant_type_{SparseValueQuantizationType::FP32};
    QuantizationParamsPtr quantization_params_;
    uint32_t window_size_{0};
    IOParamPtr io_param_{nullptr};
    IndexCommonParam common_param_;
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};

    mutable QueryTermBuffers term_buffers_;
    mutable std::shared_mutex term_buffers_mutex_;
    std::vector<DiskTermEntry> term_dict_;
    uint32_t window_count_{0};
    uint64_t total_count_{0};
    uint64_t payload_size_{0};
};

template <typename IOTmpl>
using DiskSindiTermDataCellPtr = std::shared_ptr<DiskSindiTermDataCell<IOTmpl>>;

}  // namespace vsag

#include "disk_sindi_term_datacell.inl"
