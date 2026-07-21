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

#include <cassert>
#include <memory>
#include <utility>
#include <vector>

#include "container_types.h"
#include "hash_types.h"
#include "impl/inner_search_param.h"
#include "quantization/sparse_quantization/sparse_term_computer.h"
#include "storage/stream_writer.h"
#include "utils/pointer_define.h"
#include "vsag/allocator.h"
#include "vsag/dataset.h"

namespace vsag {

struct SindiTermBuffer {
    explicit SindiTermBuffer(Allocator* allocator = nullptr)
        : window_offsets(allocator), ids(allocator), values(allocator) {
    }

    [[nodiscard]] std::pair<uint32_t, uint32_t>
    GetPostingRange(uint32_t window_id) const {
        assert(not window_offsets.empty() && window_id < window_offsets.size() - 1);
        const auto begin = window_offsets[window_id];
        return {begin, window_offsets[window_id + 1] - begin};
    }

    [[nodiscard]] const uint16_t*
    IdsData() const {
        return external_ids != nullptr ? external_ids : ids.data();
    }

    [[nodiscard]] const uint8_t*
    ValuesData() const {
        return external_values != nullptr ? external_values : values.data();
    }

    [[nodiscard]] uint64_t
    ValuesSize() const {
        return external_values != nullptr ? external_values_size : values.size();
    }

    const uint16_t* external_ids{nullptr};
    const uint8_t* external_values{nullptr};
    uint64_t external_values_size{0};

    Vector<uint32_t> window_offsets;
    Vector<uint16_t> ids;
    Vector<uint8_t> values;
};

struct DiskTermEntry {
    uint64_t posting_payload_offset{0};
    uint32_t posting_payload_size{0};
    uint32_t posting_count{0};
};

struct TermWindowMeta {
    uint32_t window_id{0};
    uint32_t posting_count{0};
};

struct SindiTermPostingView {
    const uint16_t* ids{nullptr};
    const uint8_t* values{nullptr};
    uint32_t count{0};
};

using QueryTermBuffers = UnorderedMap<uint32_t, SindiTermBuffer>;
using MappedQueryTerms = Vector<std::pair<uint32_t, uint32_t>>;

struct SindiQueryContext {
    explicit SindiQueryContext(Allocator* allocator)
        : query_term_buffers(allocator), mapped_query_terms(allocator) {
    }

    QueryTermBuffers query_term_buffers;
    MappedQueryTerms mapped_query_terms;
};

/**
 * Search and term-first serialization contract shared by mutable, immutable and disk SINDI term
 * data cells.
 *
 * Build and window-first serialization deliberately live on concrete data cells.
 * QueryTermBuffers is empty for memory data cells and query-scoped for disk data cells.
 */
class SindiTermDataCell {
public:
    virtual ~SindiTermDataCell() = default;

    virtual QueryTermBuffers
    LoadQueryTermBuffers(const Vector<uint32_t>& query_term_ids) const = 0;

    virtual void
    QueryWindow(float* dists,
                uint32_t window_id,
                const SparseTermComputerPtr& computer,
                bool use_term_lists_heap_insert,
                SindiQueryContext& query_context) const = 0;

    virtual void
    InsertHeapByWindow(float* dists,
                       uint32_t window_id,
                       const SparseTermComputerPtr& computer,
                       MaxHeap& heap,
                       const InnerSearchParam& param,
                       uint32_t offset_id,
                       InnerSearchMode mode,
                       bool with_filter,
                       const SindiQueryContext& query_context) const = 0;

    virtual void
    InsertHeapByDists(float* dists,
                      uint32_t dists_size,
                      MaxHeap& heap,
                      const InnerSearchParam& param,
                      uint32_t offset_id,
                      InnerSearchMode mode,
                      bool with_filter) const = 0;

    virtual float
    CalcDistanceByInnerId(const SparseTermComputerPtr& computer,
                          uint32_t base_id,
                          const QueryTermBuffers& query_term_buffers) const = 0;

    virtual void
    GetSparseVector(uint32_t inner_id,
                    SparseVector* data,
                    Allocator* specified_allocator) const = 0;

    [[nodiscard]] virtual uint64_t
    GetMemoryUsage() const = 0;

    [[nodiscard]] virtual uint32_t
    GetWindowCount() const = 0;

    [[nodiscard]] virtual uint32_t
    GetTermDictCount() const = 0;

    virtual void
    SerializeTermLayout(StreamWriter& writer, uint32_t term_dict_count) const = 0;
};

DEFINE_POINTER(SindiTermDataCell);

}  // namespace vsag
