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

#include <optional>

#include "datacell/mutable_sindi_term_datacell.h"
#include "datacell/sindi_term_datacell.h"

namespace vsag {

struct ImmutableSINDIWindow {
    explicit ImmutableSINDIWindow(Allocator* allocator)
        : sorted_global_terms(allocator),
          offsets(allocator),
          id_payloads(allocator),
          value_payloads(allocator) {
    }

    Vector<uint32_t> sorted_global_terms;
    Vector<uint32_t> offsets;
    Vector<uint16_t> id_payloads;
    Vector<uint8_t> value_payloads;
};

/** Compact, read-only SINDI term index. */
class ImmutableSindiTermDataCell : public SindiTermDataCell {
public:
    using MappedQueryTerms = Vector<std::pair<uint32_t, uint32_t>>;

    ImmutableSindiTermDataCell(uint32_t term_id_limit,
                               uint32_t window_size,
                               bool remap_term_ids,
                               SparseValueQuantizationType sparse_value_quant_type,
                               QuantizationParamsPtr quantization_params,
                               Allocator* allocator);

    void
    Reserve(uint32_t window_count);

    void
    AppendWindow(const MutableSINDIWindow& window);

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

    void
    SerializeWindow(StreamWriter& writer, const ImmutableSINDIWindow& window) const;

    void
    DeserializeWindow(StreamReader& reader, ImmutableSINDIWindow& window) const;

    void
    QueryWindow(float* dists,
                uint32_t window_id,
                const SparseTermComputerPtr& computer,
                bool use_term_lists_heap_insert,
                const QueryTermBuffers& query_term_buffers) const override;

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

    float
    CalcDistanceByInnerId(const SparseTermComputerPtr& computer,
                          uint32_t base_id,
                          const QueryTermBuffers& query_term_buffers) const override;

    void
    GetSparseVector(uint32_t inner_id,
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

    [[nodiscard]] const Vector<ImmutableSINDIWindow>&
    GetWindows() const {
        return windows_;
    }

private:
    [[nodiscard]] SindiTermPostingView
    GetTermPostingView(uint32_t term_id, uint32_t window_id) const;

    [[nodiscard]] std::vector<sindi_datacell_utils::TermPostingRecord>
    CollectTermPostings() const;

    std::optional<uint32_t>
    get_local_term(const ImmutableSINDIWindow& window, uint32_t term) const;

    void
    map_query_terms(const ImmutableSINDIWindow& window,
                    const SparseTermComputerPtr& computer,
                    MappedQueryTerms& mapped_terms) const;

    template <InnerSearchMode mode, InnerSearchType type>
    void
    insert_heap_by_terms(float* dists,
                         const ImmutableSINDIWindow& window,
                         const SparseTermComputerPtr& computer,
                         const MappedQueryTerms& mapped_terms,
                         MaxHeap& heap,
                         const InnerSearchParam& param,
                         uint32_t offset_id) const;

    template <InnerSearchMode mode, InnerSearchType type>
    void
    insert_heap_by_dists(float* dists,
                         uint32_t dists_size,
                         MaxHeap& heap,
                         const InnerSearchParam& param,
                         uint32_t offset_id) const;

    uint32_t term_id_limit_{0};
    uint32_t window_size_{0};
    bool remap_term_ids_{false};
    SparseValueQuantizationType sparse_value_quant_type_{SparseValueQuantizationType::FP32};
    QuantizationParamsPtr quantization_params_;
    Allocator* allocator_{nullptr};
    Vector<ImmutableSINDIWindow> windows_;
};

DEFINE_POINTER(ImmutableSindiTermDataCell);

}  // namespace vsag
