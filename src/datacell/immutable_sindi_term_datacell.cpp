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

#include "immutable_sindi_term_datacell.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "datacell/sindi_datacell_utils.h"
#include "simd/fp16_simd.h"
#include "vsag_exception.h"

namespace vsag {
namespace {

template <InnerSearchMode mode, InnerSearchType type>
void
insert_candidate(uint32_t id,
                 float& dist,
                 float& heap_top,
                 MaxHeap& heap,
                 uint32_t offset_id,
                 float radius,
                 int range_search_limit_size,
                 const FilterPtr& filter) {
    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        if (range_search_limit_size == 0) {
            dist = 0;
            return;
        }
    }
    if constexpr (type == InnerSearchType::WITH_FILTER) {
        if (__builtin_expect(dist > heap_top || not filter->CheckValid(id + offset_id), 1)) {
            dist = 0;
            return;
        }
    } else {
        if (__builtin_expect(dist > heap_top, 1)) {
            dist = 0;
            return;
        }
    }
    heap.emplace(dist, id + offset_id);
    if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
        heap.pop();
        heap_top = heap.top().first;
    } else {
        if (range_search_limit_size > 0 &&
            heap.size() > static_cast<uint32_t>(range_search_limit_size)) {
            heap.pop();
        }
        heap_top = range_search_limit_size > 0 &&
                           heap.size() == static_cast<uint32_t>(range_search_limit_size)
                       ? heap.top().first
                       : radius - 1;
    }
    dist = 0;
}

template <InnerSearchType type>
bool
fill_initial(uint32_t id,
             float& dist,
             float& heap_top,
             MaxHeap& heap,
             uint32_t offset_id,
             uint32_t n_candidate,
             const FilterPtr& filter) {
    if (dist >= 0) {
        return false;
    }
    if constexpr (type == InnerSearchType::WITH_FILTER) {
        if (not filter->CheckValid(id + offset_id)) {
            dist = 0;
            return false;
        }
    }
    heap.emplace(dist, id + offset_id);
    heap_top = heap.top().first;
    dist = 0;
    return heap.size() == n_candidate;
}

}  // namespace

ImmutableSindiTermDataCell::ImmutableSindiTermDataCell(
    uint32_t term_id_limit,
    uint32_t window_size,
    bool remap_term_ids,
    SparseValueQuantizationType sparse_value_quant_type,
    QuantizationParamsPtr quantization_params,
    Allocator* allocator)
    : term_id_limit_(term_id_limit),
      window_size_(window_size),
      remap_term_ids_(remap_term_ids),
      sparse_value_quant_type_(sparse_value_quant_type),
      value_code_size_(sindi_datacell_utils::GetValueCodeSize(sparse_value_quant_type)),
      quantization_params_(std::move(quantization_params)),
      allocator_(allocator),
      windows_(allocator) {
}

void
ImmutableSindiTermDataCell::Reserve(uint32_t window_count) {
    windows_.reserve(window_count);
}

void
ImmutableSindiTermDataCell::SerializeTermLayout(StreamWriter& writer,
                                                uint32_t term_dict_count) const {
    const auto value_code_size = value_code_size_;
    const auto layout = sindi_datacell_utils::BuildTermLayout(
        term_dict_count, GetWindowCount(), value_code_size, this->CollectTermPostings());
    sindi_datacell_utils::SerializeTermLayout(writer, layout, value_code_size);
}

SindiTermPostingView
ImmutableSindiTermDataCell::GetTermPostingView(uint32_t term_id, uint32_t window_id) const {
    if (window_id >= windows_.size()) {
        return {};
    }
    const auto& window = windows_[window_id];
    const auto local_term = this->get_local_term(window, term_id);
    if (not local_term.has_value()) {
        return {};
    }
    const auto begin = window.offsets[local_term.value()];
    const auto end = window.offsets[local_term.value() + 1];
    if (begin == end) {
        return {};
    }
    return {window.id_payloads.data() + begin,
            window.value_payloads.data() + static_cast<uint64_t>(begin) * value_code_size_,
            end - begin};
}

std::vector<sindi_datacell_utils::TermPostingRecord>
ImmutableSindiTermDataCell::CollectTermPostings() const {
    std::vector<sindi_datacell_utils::TermPostingRecord> postings;
    for (uint32_t window_id = 0; window_id < windows_.size(); ++window_id) {
        const auto& window = windows_[window_id];
        const auto term_count = static_cast<uint32_t>(window.offsets.size() - 1);
        for (uint32_t local_term = 0; local_term < term_count; ++local_term) {
            if (window.offsets[local_term] == window.offsets[local_term + 1]) {
                continue;
            }
            const auto term_id =
                remap_term_ids_ ? window.sorted_global_terms[local_term] : local_term;
            postings.push_back({term_id, window_id, this->GetTermPostingView(term_id, window_id)});
        }
    }
    return postings;
}

uint32_t
ImmutableSindiTermDataCell::GetTermDictCount() const {
    uint32_t term_dict_count = 0;
    for (const auto& window : windows_) {
        if (remap_term_ids_) {
            if (not window.sorted_global_terms.empty()) {
                term_dict_count = std::max(term_dict_count, window.sorted_global_terms.back() + 1);
            }
            continue;
        }
        if (window.offsets.empty()) {
            continue;
        }
        for (uint32_t term = static_cast<uint32_t>(window.offsets.size() - 1);
             term > term_dict_count;
             --term) {
            if (window.offsets[term - 1] != window.offsets[term]) {
                term_dict_count = term;
                break;
            }
        }
    }
    return term_dict_count;
}

void
ImmutableSindiTermDataCell::AppendWindow(const MutableSINDIWindow& term_list) {
    ImmutableSINDIWindow window(allocator_);
    uint64_t posting_count = 0;
    if (not remap_term_ids_) {
        window.offsets.reserve(static_cast<uint64_t>(term_list.term_capacity_) + 1);
    }
    window.offsets.push_back(0);
    for (uint32_t term = 0; term < term_list.term_capacity_; ++term) {
        const auto count = term_list.term_sizes_[term];
        if (count == 0) {
            if (not remap_term_ids_) {
                window.offsets.push_back(static_cast<uint32_t>(posting_count));
            }
            continue;
        }
        CHECK_ARGUMENT(posting_count <= std::numeric_limits<uint32_t>::max() - count,
                       "immutable SINDI posting offset overflows uint32_t");
        if (remap_term_ids_) {
            window.sorted_global_terms.push_back(term);
        }
        window.id_payloads.insert(window.id_payloads.end(),
                                  term_list.term_ids_[term]->begin(),
                                  term_list.term_ids_[term]->end());
        window.value_payloads.insert(window.value_payloads.end(),
                                     term_list.term_datas_[term]->begin(),
                                     term_list.term_datas_[term]->end());
        posting_count += count;
        window.offsets.push_back(static_cast<uint32_t>(posting_count));
    }
    windows_.emplace_back(std::move(window));
}

std::optional<uint32_t>
ImmutableSindiTermDataCell::get_local_term(const ImmutableSINDIWindow& window,
                                           uint32_t term) const {
    if (remap_term_ids_) {
        const auto it = std::lower_bound(
            window.sorted_global_terms.begin(), window.sorted_global_terms.end(), term);
        if (it == window.sorted_global_terms.end() || *it != term) {
            return std::nullopt;
        }
        return static_cast<uint32_t>(it - window.sorted_global_terms.begin());
    }
    if (term + 1 >= window.offsets.size()) {
        return std::nullopt;
    }
    return term;
}

void
ImmutableSindiTermDataCell::map_query_terms(const ImmutableSINDIWindow& window,
                                            const SparseTermComputerPtr& computer,
                                            MappedQueryTerms& mapped_terms) const {
    mapped_terms.clear();
    mapped_terms.reserve(computer->pruned_len_);
    for (uint32_t it = 0; it < computer->pruned_len_; ++it) {
        const auto local_term = this->get_local_term(window, computer->GetTerm(it));
        if (local_term.has_value() &&
            window.offsets[local_term.value()] != window.offsets[local_term.value() + 1]) {
            mapped_terms.emplace_back(local_term.value(), it);
        }
    }
}

void
ImmutableSindiTermDataCell::QueryWindow(float* dists,
                                        uint32_t window_id,
                                        const SparseTermComputerPtr& computer,
                                        bool use_term_lists_heap_insert,
                                        SindiQueryContext& query_context) const {
    CHECK_ARGUMENT(window_id < windows_.size(), "immutable SINDI window id out of range");
    const auto& window = windows_[window_id];
    auto& mapped_terms = query_context.mapped_query_terms;
    this->map_query_terms(window, computer, mapped_terms);
    for (uint32_t pos = 0; pos < mapped_terms.size(); ++pos) {
        const auto local_term = mapped_terms[pos].first;
        const auto query_term = mapped_terms[pos].second;
        if (pos + 1 < mapped_terms.size()) {
            const auto next_begin = window.offsets[mapped_terms[pos + 1].first];
            __builtin_prefetch(window.id_payloads.data() + next_begin, 0, 3);
            __builtin_prefetch(
                window.value_payloads.data() + static_cast<uint64_t>(next_begin) * value_code_size_,
                0,
                3);
        }
        const auto begin = window.offsets[local_term];
        const auto count =
            static_cast<uint32_t>(static_cast<float>(window.offsets[local_term + 1] - begin) *
                                  computer->term_retain_ratio_);
        const auto* ids = window.id_payloads.data() + begin;
        const auto* values =
            window.value_payloads.data() + static_cast<uint64_t>(begin) * value_code_size_;
        if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
            computer->ScanForAccumulateSQ8(query_term, ids, values, count, dists);
        } else if (sparse_value_quant_type_ == SparseValueQuantizationType::FP16) {
            computer->ScanForAccumulateFP16Bytes(query_term, ids, values, count, dists);
        } else {
            computer->ScanForAccumulateFloatBytes(query_term, ids, values, count, dists);
        }
    }
    computer->ResetTerm();
}

template <InnerSearchMode mode, InnerSearchType type>
void
ImmutableSindiTermDataCell::insert_heap_by_terms(float* dists,
                                                 const ImmutableSINDIWindow& window,
                                                 const SparseTermComputerPtr& computer,
                                                 const MappedQueryTerms& mapped_terms,
                                                 MaxHeap& heap,
                                                 const InnerSearchParam& param,
                                                 uint32_t offset_id) const {
    const auto n_candidate = static_cast<uint32_t>(param.ef);
    const auto radius = param.radius;
    const auto range_search_limit_size = param.range_search_limit_size;
    const auto& filter = param.is_inner_id_allowed;
    float heap_top =
        mode == InnerSearchMode::RANGE_SEARCH ? radius - 1 : std::numeric_limits<float>::max();
    for (const auto& mapped_term : mapped_terms) {
        const auto begin = window.offsets[mapped_term.first];
        const auto count = static_cast<uint32_t>(
            static_cast<float>(window.offsets[mapped_term.first + 1] - begin) *
            computer->term_retain_ratio_);
        const auto* ids = window.id_payloads.data() + begin;
        uint32_t pos = 0;
        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            while (pos < count && heap.size() < n_candidate) {
                const auto id = ids[pos++];
                if (fill_initial<type>(
                        id, dists[id], heap_top, heap, offset_id, n_candidate, filter)) {
                    break;
                }
            }
        }
        for (; pos < count; ++pos) {
            const auto id = ids[pos];
            insert_candidate<mode, type>(
                id, dists[id], heap_top, heap, offset_id, radius, range_search_limit_size, filter);
        }
    }
}

template <InnerSearchMode mode, InnerSearchType type>
void
ImmutableSindiTermDataCell::insert_heap_by_dists(float* dists,
                                                 uint32_t dists_size,
                                                 MaxHeap& heap,
                                                 const InnerSearchParam& param,
                                                 uint32_t offset_id) const {
    const auto n_candidate = static_cast<uint32_t>(param.ef);
    const auto radius = param.radius;
    const auto range_search_limit_size = param.range_search_limit_size;
    const auto& filter = param.is_inner_id_allowed;
    float heap_top =
        mode == InnerSearchMode::RANGE_SEARCH ? radius - 1 : std::numeric_limits<float>::max();
    uint32_t id = 0;
    if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
        while (id < dists_size && heap.size() < n_candidate) {
            if (fill_initial<type>(id, dists[id], heap_top, heap, offset_id, n_candidate, filter)) {
                ++id;
                break;
            }
            ++id;
        }
    }
    for (; id < dists_size; ++id) {
        insert_candidate<mode, type>(
            id, dists[id], heap_top, heap, offset_id, radius, range_search_limit_size, filter);
    }
}

void
ImmutableSindiTermDataCell::InsertHeapByWindow(float* dists,
                                               uint32_t window_id,
                                               const SparseTermComputerPtr& computer,
                                               MaxHeap& heap,
                                               const InnerSearchParam& param,
                                               uint32_t offset_id,
                                               InnerSearchMode mode,
                                               bool with_filter,
                                               const SindiQueryContext& query_context) const {
    CHECK_ARGUMENT(window_id < windows_.size(), "immutable SINDI window id out of range");
    const auto& window = windows_[window_id];
    const auto& mapped_terms = query_context.mapped_query_terms;
    if (mode == InnerSearchMode::KNN_SEARCH && with_filter) {
        this->insert_heap_by_terms<InnerSearchMode::KNN_SEARCH, InnerSearchType::WITH_FILTER>(
            dists, window, computer, mapped_terms, heap, param, offset_id);
    } else if (mode == InnerSearchMode::KNN_SEARCH) {
        this->insert_heap_by_terms<InnerSearchMode::KNN_SEARCH, InnerSearchType::PURE>(
            dists, window, computer, mapped_terms, heap, param, offset_id);
    } else if (with_filter) {
        this->insert_heap_by_terms<InnerSearchMode::RANGE_SEARCH, InnerSearchType::WITH_FILTER>(
            dists, window, computer, mapped_terms, heap, param, offset_id);
    } else {
        this->insert_heap_by_terms<InnerSearchMode::RANGE_SEARCH, InnerSearchType::PURE>(
            dists, window, computer, mapped_terms, heap, param, offset_id);
    }
}

void
ImmutableSindiTermDataCell::InsertHeapByDists(float* dists,
                                              uint32_t dists_size,
                                              MaxHeap& heap,
                                              const InnerSearchParam& param,
                                              uint32_t offset_id,
                                              InnerSearchMode mode,
                                              bool with_filter) const {
    if (mode == InnerSearchMode::KNN_SEARCH && with_filter) {
        this->insert_heap_by_dists<InnerSearchMode::KNN_SEARCH, InnerSearchType::WITH_FILTER>(
            dists, dists_size, heap, param, offset_id);
    } else if (mode == InnerSearchMode::KNN_SEARCH) {
        this->insert_heap_by_dists<InnerSearchMode::KNN_SEARCH, InnerSearchType::PURE>(
            dists, dists_size, heap, param, offset_id);
    } else if (with_filter) {
        this->insert_heap_by_dists<InnerSearchMode::RANGE_SEARCH, InnerSearchType::WITH_FILTER>(
            dists, dists_size, heap, param, offset_id);
    } else {
        this->insert_heap_by_dists<InnerSearchMode::RANGE_SEARCH, InnerSearchType::PURE>(
            dists, dists_size, heap, param, offset_id);
    }
}

float
ImmutableSindiTermDataCell::CalcDistanceByInnerId(
    const SparseTermComputerPtr& computer,
    uint32_t base_id,
    const QueryTermBuffers& query_term_buffers) const {
    const auto window_id = base_id / window_size_;
    const auto local_id = static_cast<uint16_t>(base_id % window_size_);
    CHECK_ARGUMENT(window_id < windows_.size(), "immutable SINDI inner id out of range");
    const auto& window = windows_[window_id];
    float ip = 0.0F;
    while (computer->HasNextTerm()) {
        const auto query_term = computer->NextTermIter();
        const auto local_term = this->get_local_term(window, computer->GetTerm(query_term));
        if (not local_term.has_value()) {
            continue;
        }
        const auto begin = window.offsets[local_term.value()];
        const auto end = window.offsets[local_term.value() + 1];
        const auto found = std::lower_bound(
            window.id_payloads.begin() + begin, window.id_payloads.begin() + end, local_id);
        if (found == window.id_payloads.begin() + end || *found != local_id) {
            continue;
        }
        const auto posting = static_cast<uint64_t>(found - window.id_payloads.begin());
        const auto* encoded = window.value_payloads.data() + posting * value_code_size_;
        const auto value = sindi_datacell_utils::DecodeValue(
            encoded, sparse_value_quant_type_, quantization_params_.get());
        ip += computer->sorted_query_[query_term].second * value;
    }
    computer->ResetTerm();
    return 1.0F + ip;
}

void
ImmutableSindiTermDataCell::GetSparseVector(uint32_t inner_id,
                                            SparseVector* data,
                                            Allocator* specified_allocator) const {
    const auto window_id = inner_id / window_size_;
    const auto local_id = static_cast<uint16_t>(inner_id % window_size_);
    CHECK_ARGUMENT(window_id < windows_.size(), "immutable SINDI inner id out of range");
    const auto& window = windows_[window_id];
    auto* output_allocator = specified_allocator == nullptr ? allocator_ : specified_allocator;
    Vector<uint32_t> ids(output_allocator);
    Vector<float> values(output_allocator);
    const auto term_count = static_cast<uint32_t>(window.offsets.size() - 1);
    for (uint32_t local_term = 0; local_term < term_count; ++local_term) {
        const auto begin = window.offsets[local_term];
        const auto end = window.offsets[local_term + 1];
        const auto found = std::lower_bound(
            window.id_payloads.begin() + begin, window.id_payloads.begin() + end, local_id);
        if (found == window.id_payloads.begin() + end || *found != local_id) {
            continue;
        }
        ids.push_back(remap_term_ids_ ? window.sorted_global_terms[local_term] : local_term);
        const auto posting = static_cast<uint64_t>(found - window.id_payloads.begin());
        const auto* encoded = window.value_payloads.data() + posting * value_code_size_;
        const auto value = sindi_datacell_utils::DecodeValue(
            encoded, sparse_value_quant_type_, quantization_params_.get());
        values.push_back(value);
    }
    data->len_ = static_cast<uint32_t>(ids.size());
    data->ids_ = static_cast<uint32_t*>(output_allocator->Allocate(ids.size() * sizeof(uint32_t)));
    data->vals_ = static_cast<float*>(output_allocator->Allocate(values.size() * sizeof(float)));
    std::copy(ids.begin(), ids.end(), data->ids_);
    std::copy(values.begin(), values.end(), data->vals_);
}

uint64_t
ImmutableSindiTermDataCell::GetMemoryUsage() const {
    uint64_t memory = sizeof(*this) + windows_.capacity() * sizeof(ImmutableSINDIWindow);
    for (const auto& window : windows_) {
        memory += window.sorted_global_terms.capacity() * sizeof(uint32_t);
        memory += window.offsets.capacity() * sizeof(uint32_t);
        memory += window.id_payloads.capacity() * sizeof(uint16_t);
        memory += window.value_payloads.capacity();
    }
    return memory;
}

void
ImmutableSindiTermDataCell::SerializeWindow(StreamWriter& writer,
                                            const ImmutableSINDIWindow& window) const {
    StreamWriter::WriteVector(writer, window.sorted_global_terms);
    StreamWriter::WriteVector(writer, window.offsets);
    StreamWriter::WriteVector(writer, window.id_payloads);
    StreamWriter::WriteVector(writer, window.value_payloads);
}

void
ImmutableSindiTermDataCell::SerializeWindows(StreamWriter& writer) const {
    for (const auto& window : windows_) {
        this->SerializeWindow(writer, window);
    }
}

void
ImmutableSindiTermDataCell::DeserializeWindow(StreamReader& reader,
                                              ImmutableSINDIWindow& window) const {
    const auto read_vector = [&reader](auto& values, uint64_t max_size, const char* message) {
        uint64_t size = 0;
        StreamReader::ReadObj(reader, size);
        CHECK_ARGUMENT(size <= max_size && size <= static_cast<uint64_t>(values.max_size()),
                       message);
        values.resize(size);
        reader.Read(reinterpret_cast<char*>(values.data()),
                    size * sizeof(typename std::decay_t<decltype(values)>::value_type));
    };
    read_vector(window.sorted_global_terms,
                remap_term_ids_ ? term_id_limit_ : 0,
                "immutable SINDI remapped term count exceeds term_id_limit");
    read_vector(window.offsets,
                remap_term_ids_ ? static_cast<uint64_t>(term_id_limit_) + 1
                                : static_cast<uint64_t>(term_id_limit_) * 2 + 1,
                "immutable SINDI offset count exceeds capacity bound");
    CHECK_ARGUMENT(not window.offsets.empty() && window.offsets.front() == 0,
                   "immutable SINDI offsets must start at zero");
    CHECK_ARGUMENT(std::is_sorted(window.offsets.begin(), window.offsets.end()),
                   "immutable SINDI offsets must be sorted");
    if (remap_term_ids_) {
        CHECK_ARGUMENT(window.offsets.size() == window.sorted_global_terms.size() + 1,
                       "immutable SINDI term and offset counts do not match");
        CHECK_ARGUMENT(
            std::adjacent_find(window.sorted_global_terms.begin(),
                               window.sorted_global_terms.end(),
                               std::greater_equal<uint32_t>()) == window.sorted_global_terms.end(),
            "immutable SINDI remapped terms must be strictly increasing");
    } else {
        CHECK_ARGUMENT(window.sorted_global_terms.empty(),
                       "immutable SINDI dense window must not contain remapped terms");
    }
    for (uint64_t pos = 1; pos < window.offsets.size(); ++pos) {
        CHECK_ARGUMENT(window.offsets[pos] - window.offsets[pos - 1] <= window_size_,
                       "immutable SINDI posting count exceeds window size");
    }
    const auto posting_count = static_cast<uint64_t>(window.offsets.back());
    read_vector(
        window.id_payloads, posting_count, "immutable SINDI id payload count exceeds offsets");
    CHECK_ARGUMENT(window.id_payloads.size() == posting_count,
                   "immutable SINDI id payload size does not match offsets");
    CHECK_ARGUMENT(posting_count <= std::numeric_limits<uint64_t>::max() / value_code_size_,
                   "immutable SINDI value payload size overflows uint64_t");
    read_vector(window.value_payloads,
                posting_count * value_code_size_,
                "immutable SINDI value payload count exceeds offsets");
    CHECK_ARGUMENT(window.value_payloads.size() == posting_count * value_code_size_,
                   "immutable SINDI value payload size does not match offsets");
    for (const auto id : window.id_payloads) {
        CHECK_ARGUMENT(id < window_size_,
                       "immutable SINDI window-local doc id exceeds window size");
    }
}

void
ImmutableSindiTermDataCell::DeserializeWindows(StreamReader& reader, uint32_t window_count) {
    Vector<ImmutableSINDIWindow> loaded(allocator_);
    loaded.reserve(window_count);
    for (uint32_t i = 0; i < window_count; ++i) {
        loaded.emplace_back(allocator_);
        this->DeserializeWindow(reader, loaded.back());
    }
    windows_.swap(loaded);
}

void
ImmutableSindiTermDataCell::DeserializeTermLayout(StreamReader& reader,
                                                  uint32_t window_count,
                                                  uint64_t total_count) {
    const auto term_dict = sindi_datacell_utils::DeserializeTermDictionary(reader, term_id_limit_);
    uint64_t term_payload_size = 0;
    StreamReader::ReadObj(reader, term_payload_size);
    CHECK_ARGUMENT(reader.GetCursor() <= reader.Length() &&
                       term_payload_size <= reader.Length() - reader.GetCursor(),
                   "SINDI V2 term payload exceeds stream length");
    const auto payload_start = reader.GetCursor();
    sindi_datacell_utils::ValidateTermDict(term_dict, term_payload_size);
    const auto value_code_size = value_code_size_;

    Vector<uint64_t> window_term_counts(window_count, 0, allocator_);
    Vector<uint32_t> window_posting_counts(window_count, 0, allocator_);
    Vector<TermWindowMeta> metadata(allocator_);
    uint64_t posting_scratch_size = 0;
    for (const auto& entry : term_dict) {
        if (entry.posting_count == 0) {
            continue;
        }
        [[maybe_unused]] const auto layout =
            sindi_datacell_utils::ReadTermPayloadMetadata(reader,
                                                          payload_start,
                                                          term_payload_size,
                                                          entry,
                                                          window_count,
                                                          value_code_size,
                                                          metadata);
        if (metadata.size() > 1) {
            const auto posting_width = std::max<uint64_t>(sizeof(uint16_t), value_code_size);
            posting_scratch_size = std::max(
                posting_scratch_size, static_cast<uint64_t>(entry.posting_count) * posting_width);
        }
        for (const auto& meta : metadata) {
            CHECK_ARGUMENT(
                window_term_counts[meta.window_id] < std::numeric_limits<uint64_t>::max(),
                "immutable SINDI window term count overflows uint64_t");
            ++window_term_counts[meta.window_id];
            CHECK_ARGUMENT(window_posting_counts[meta.window_id] <=
                               std::numeric_limits<uint32_t>::max() - meta.posting_count,
                           "immutable SINDI posting offset overflows uint32_t");
            window_posting_counts[meta.window_id] += meta.posting_count;
        }
    }

    Vector<ImmutableSINDIWindow> loaded(allocator_);
    loaded.reserve(window_count);
    for (uint32_t window_id = 0; window_id < window_count; ++window_id) {
        loaded.emplace_back(allocator_);
        auto& window = loaded.back();
        const auto term_count = remap_term_ids_ ? window_term_counts[window_id] : term_dict.size();
        CHECK_ARGUMENT(term_count < std::numeric_limits<uint64_t>::max(),
                       "immutable SINDI offset count overflows uint64_t");
        CHECK_ARGUMENT(term_count <= window.offsets.max_size() - 1,
                       "immutable SINDI offset count exceeds vector capacity");
        if (remap_term_ids_) {
            CHECK_ARGUMENT(term_count <= window.sorted_global_terms.max_size(),
                           "immutable SINDI remapped term count exceeds vector capacity");
            window.sorted_global_terms.resize(term_count);
        }
        window.offsets.resize(term_count + 1);
        window.offsets[0] = 0;
        const auto posting_count = window_posting_counts[window_id];
        CHECK_ARGUMENT(posting_count <= window.id_payloads.max_size(),
                       "immutable SINDI id payload count exceeds vector capacity");
        window.id_payloads.resize(posting_count);
        CHECK_ARGUMENT(posting_count <= std::numeric_limits<uint64_t>::max() / value_code_size,
                       "immutable SINDI value payload size overflows uint64_t");
        const auto value_count = static_cast<uint64_t>(posting_count) * value_code_size;
        CHECK_ARGUMENT(value_count <= window.value_payloads.max_size(),
                       "immutable SINDI value payload count exceeds vector capacity");
        window.value_payloads.resize(value_count);
    }

    Vector<uint8_t> posting_scratch(posting_scratch_size, allocator_);
    Vector<uint64_t> window_term_cursors(window_count, 0, allocator_);
    Vector<uint32_t> window_posting_cursors(window_count, 0, allocator_);
    for (uint64_t term = 0; term < term_dict.size(); ++term) {
        const auto& entry = term_dict[term];
        if (entry.posting_count != 0) {
            const auto layout = sindi_datacell_utils::ReadTermPayloadMetadata(reader,
                                                                              payload_start,
                                                                              term_payload_size,
                                                                              entry,
                                                                              window_count,
                                                                              value_code_size,
                                                                              metadata);
            const auto validate_ids = [&](const TermWindowMeta& meta, uint32_t posting_offset) {
                const auto window_id = meta.window_id;
                const auto window_start = static_cast<uint64_t>(window_id) * window_size_;
                const auto window_document_count = static_cast<uint32_t>(std::min<uint64_t>(
                    window_size_, total_count > window_start ? total_count - window_start : 0));
                const auto& window = loaded[window_id];
                for (uint32_t posting = 0; posting < meta.posting_count; ++posting) {
                    CHECK_ARGUMENT(
                        window.id_payloads[posting_offset + posting] < window_document_count,
                        "SINDI V2 posting id exceeds its window document count");
                }
            };
            const auto complete_window = [&](const TermWindowMeta& meta) {
                const auto window_id = meta.window_id;
                auto& window = loaded[window_id];
                window_posting_cursors[window_id] += meta.posting_count;
                if (remap_term_ids_) {
                    const auto term_offset = window_term_cursors[window_id];
                    CHECK_ARGUMENT(term_offset < window.sorted_global_terms.size(),
                                   "immutable SINDI term fill exceeds allocated terms");
                    window.sorted_global_terms[term_offset] = static_cast<uint32_t>(term);
                    ++window_term_cursors[window_id];
                    window.offsets[term_offset + 1] = window_posting_cursors[window_id];
                }
            };

            uint32_t source_posting_offset = 0;
            if (metadata.size() == 1) {
                const auto& meta = metadata.front();
                const auto window_id = meta.window_id;
                auto& window = loaded[window_id];
                const auto posting_offset = window_posting_cursors[window_id];
                CHECK_ARGUMENT(posting_offset <= window.id_payloads.size() &&
                                   meta.posting_count <= window.id_payloads.size() - posting_offset,
                               "immutable SINDI posting fill exceeds allocated payload");
                sindi_datacell_utils::ReadTermPostingRange(
                    reader,
                    layout,
                    0,
                    meta.posting_count,
                    value_code_size,
                    window.id_payloads.data() + posting_offset,
                    window.value_payloads.data() +
                        static_cast<uint64_t>(posting_offset) * value_code_size);
                validate_ids(meta, posting_offset);
                complete_window(meta);
                source_posting_offset = meta.posting_count;
            } else {
                const auto ids_size = static_cast<uint64_t>(entry.posting_count) * sizeof(uint16_t);
                CHECK_ARGUMENT(ids_size <= posting_scratch.size(),
                               "immutable SINDI id scratch buffer is too small");
                reader.PushSeek(layout.ids_offset);
                reader.Read(reinterpret_cast<char*>(posting_scratch.data()), ids_size);
                reader.PopSeek();
                for (const auto& meta : metadata) {
                    const auto window_id = meta.window_id;
                    auto& window = loaded[window_id];
                    const auto posting_offset = window_posting_cursors[window_id];
                    CHECK_ARGUMENT(
                        posting_offset <= window.id_payloads.size() &&
                            meta.posting_count <= window.id_payloads.size() - posting_offset,
                        "immutable SINDI posting fill exceeds allocated payload");
                    std::memcpy(window.id_payloads.data() + posting_offset,
                                posting_scratch.data() +
                                    static_cast<uint64_t>(source_posting_offset) * sizeof(uint16_t),
                                static_cast<uint64_t>(meta.posting_count) * sizeof(uint16_t));
                    validate_ids(meta, posting_offset);
                    source_posting_offset += meta.posting_count;
                }

                const auto values_size =
                    static_cast<uint64_t>(entry.posting_count) * value_code_size;
                CHECK_ARGUMENT(values_size <= posting_scratch.size(),
                               "immutable SINDI value scratch buffer is too small");
                reader.PushSeek(layout.values_offset);
                reader.Read(reinterpret_cast<char*>(posting_scratch.data()), values_size);
                reader.PopSeek();
                source_posting_offset = 0;
                for (const auto& meta : metadata) {
                    const auto window_id = meta.window_id;
                    auto& window = loaded[window_id];
                    const auto posting_offset = window_posting_cursors[window_id];
                    std::memcpy(window.value_payloads.data() +
                                    static_cast<uint64_t>(posting_offset) * value_code_size,
                                posting_scratch.data() +
                                    static_cast<uint64_t>(source_posting_offset) * value_code_size,
                                static_cast<uint64_t>(meta.posting_count) * value_code_size);
                    complete_window(meta);
                    source_posting_offset += meta.posting_count;
                }
            }
            CHECK_ARGUMENT(source_posting_offset == entry.posting_count,
                           "SINDI V2 term posting count does not match term dictionary");
        }
        if (not remap_term_ids_) {
            for (uint32_t window_id = 0; window_id < window_count; ++window_id) {
                loaded[window_id].offsets[term + 1] = window_posting_cursors[window_id];
            }
        }
    }
    for (uint32_t window_id = 0; window_id < window_count; ++window_id) {
        CHECK_ARGUMENT(window_posting_cursors[window_id] == window_posting_counts[window_id],
                       "immutable SINDI posting fill does not match allocated payload");
        if (remap_term_ids_) {
            CHECK_ARGUMENT(window_term_cursors[window_id] == window_term_counts[window_id],
                           "immutable SINDI term fill does not match allocated terms");
        }
    }
    windows_.swap(loaded);
    reader.Seek(payload_start + term_payload_size);
}

}  // namespace vsag
