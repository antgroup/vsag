
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

#include <map>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "container_types.h"
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

struct DiskSINDIManifest {
    uint64_t term_dict_offset{0};
    uint64_t term_dict_size{0};
    uint64_t posting_payload_offset{0};
    uint64_t posting_payload_size{0};
    uint64_t rerank_flat_offset{0};
    uint64_t rerank_flat_size{0};
    uint64_t label_table_offset{0};
    uint64_t label_table_size{0};
};

struct DiskTermEntry {
    uint64_t posting_payload_offset{0};
    uint32_t posting_payload_size{0};
    uint32_t term_num{0};
};

struct TermBuffer {
    std::vector<uint32_t> window_offsets;
    std::vector<uint16_t> ids;
    std::vector<uint8_t> values;
};

using QueryTermBuffers = std::unordered_map<uint32_t, TermBuffer>;

class DiskSparseTermListDataCellInterface {
public:
    virtual ~DiskSparseTermListDataCellInterface() = default;

    static std::shared_ptr<DiskSparseTermListDataCellInterface>
    MakeInstance(float doc_retain_ratio,
                 uint32_t term_id_limit,
                 Allocator* allocator,
                 bool use_quantization,
                 QuantizationParamsPtr quantization_params,
                 uint32_t window_size,
                 const IOParamPtr& io_param,
                 const IndexCommonParam& common_param);

    virtual void
    InsertVector(const SparseVector& vector, uint32_t inner_id) = 0;

    virtual void
    FinalizeTermBuffers(uint32_t window_count) = 0;

    virtual uint64_t
    ComputePayloadSize() const = 0;

    virtual void
    WriteTermDictAndPayload(StreamWriter& writer, uint64_t payload_base_offset) const = 0;

    virtual void
    Deserialize(StreamReader& reader, uint64_t term_dict_size, uint32_t window_count) = 0;

    virtual void
    InitIO(const IOParamPtr& io_param) = 0;

    virtual void
    WritePayloadToIO(uint64_t payload_base_offset) = 0;

    virtual void
    WritePayloadToIO(StreamReader& reader, uint64_t payload_offset, uint64_t payload_size) = 0;

    virtual void
    SetIO(const std::shared_ptr<Reader>& reader,
          uint64_t payload_offset,
          uint64_t payload_size) = 0;

    virtual void
    LoadQueryTerms(const Vector<uint32_t>& query_term_ids) = 0;

    virtual QueryTermBuffers
    LoadQueryTermBuffers(const Vector<uint32_t>& query_term_ids) const = 0;

    virtual uint32_t
    GetWindowCount() const = 0;

    virtual uint64_t
    GetMemoryUsage() const = 0;

    virtual void
    QueryWindow(float* dists,
                uint32_t window_id,
                const SparseTermComputerPtr& computer,
                const QueryTermBuffers& query_term_buffers) const = 0;

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

    virtual void
    GetSparseVector(uint32_t inner_id, SparseVector* data, Allocator* specified_allocator) = 0;

    virtual float
    CalcDistanceByInnerId(const SparseTermComputerPtr& computer,
                          uint32_t base_id,
                          const QueryTermBuffers& query_term_buffers) = 0;
};

using DiskSparseTermListDataCellInterfacePtr = std::shared_ptr<DiskSparseTermListDataCellInterface>;

template <typename IOTmpl>
class DiskSparseTermListDataCell : public DiskSparseTermListDataCellInterface {
public:
    DiskSparseTermListDataCell(float doc_retain_ratio,
                               uint32_t term_id_limit,
                               Allocator* allocator,
                               bool use_quantization,
                               QuantizationParamsPtr quantization_params,
                               uint32_t window_size,
                               IOParamPtr io_param,
                               const IndexCommonParam& common_param);

    void
    InsertVector(const SparseVector& vector, uint32_t inner_id) override;

    void
    FinalizeTermBuffers(uint32_t window_count) override;

    uint64_t
    ComputePayloadSize() const override;

    void
    BuildTermDict(std::vector<DiskTermEntry>& term_dict, uint64_t payload_base_offset) const;

    void
    WritePayload(StreamWriter& writer) const;

    void
    WriteTermDictAndPayload(StreamWriter& writer, uint64_t payload_base_offset) const override;

    void
    Deserialize(StreamReader& reader, uint64_t term_dict_size, uint32_t window_count) override;

    void
    InitIO(const IOParamPtr& io_param) override;

    void
    WritePayloadToIO(uint64_t payload_base_offset) override;

    void
    WritePayloadToIO(StreamReader& reader, uint64_t payload_offset, uint64_t payload_size) override;

    void
    SetIO(const std::shared_ptr<Reader>& reader,
          uint64_t payload_offset,
          uint64_t payload_size) override;

    void
    LoadQueryTerms(const Vector<uint32_t>& query_term_ids) override;

    QueryTermBuffers
    LoadQueryTermBuffers(const Vector<uint32_t>& query_term_ids) const override;

    const TermBuffer*
    GetTermBuffer(uint32_t term_id) const;

    const TermBuffer*
    GetTermBuffer(uint32_t term_id, const QueryTermBuffers& query_term_buffers) const;

    uint32_t
    GetWindowCount() const override {
        return window_count_;
    }

    uint64_t
    GetMemoryUsage() const override;

    void
    QueryWindow(float* dists,
                uint32_t window_id,
                const SparseTermComputerPtr& computer,
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
    GetSparseVector(uint32_t inner_id, SparseVector* data, Allocator* specified_allocator) override;

    float
    CalcDistanceByInnerId(const SparseTermComputerPtr& computer,
                          uint32_t base_id,
                          const QueryTermBuffers& query_term_buffers) override;

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
    DocPrune(Vector<std::pair<uint32_t, float>>& sorted_base) const;

    void
    Encode(float val, uint8_t* dst) const;

    void
    Decode(const uint8_t* src, size_t size, float* dst) const;

private:
    float doc_retain_ratio_{0};
    uint32_t term_id_limit_{0};
    Allocator* allocator_{nullptr};
    bool use_quantization_{false};
    QuantizationParamsPtr quantization_params_;
    uint32_t window_size_{0};
    IOParamPtr io_param_{nullptr};
    IndexCommonParam common_param_;
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};

    std::unordered_map<uint32_t, TermBuffer> term_buffers_;
    mutable std::shared_mutex term_buffers_mutex_;
    std::map<uint32_t, std::vector<std::pair<uint32_t, float>>> build_buffers_;
    std::vector<DiskTermEntry> term_dict_;
    uint32_t window_count_{0};
    uint64_t total_count_{0};
};

template <typename IOTmpl>
using DiskSparseTermListDataCellPtr = std::shared_ptr<DiskSparseTermListDataCell<IOTmpl>>;

}  // namespace vsag

#include "disk_sparse_term_list_datacell.inl"
