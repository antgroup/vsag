
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

#include "disk_sparse_term_list_datacell.h"

#include <type_traits>

#include "io/io_headers.h"
#include "io/reader_io_parameter.h"
#include "utils/byte_buffer.h"
#include "utils/sparse_vector_transform.h"
#include "utils/util_functions.h"
#include "vsag_exception.h"

namespace vsag {

namespace {

template <typename IOTmpl>
DiskSparseTermListDataCellInterfacePtr
make_disk_sparse_term_list_datacell(float doc_retain_ratio,
                                    uint32_t term_id_limit,
                                    Allocator* allocator,
                                    bool use_quantization,
                                    QuantizationParamsPtr quantization_params,
                                    uint32_t window_size,
                                    const IOParamPtr& io_param,
                                    const IndexCommonParam& common_param) {
    return std::make_shared<DiskSparseTermListDataCell<IOTmpl>>(doc_retain_ratio,
                                                                term_id_limit,
                                                                allocator,
                                                                use_quantization,
                                                                std::move(quantization_params),
                                                                window_size,
                                                                io_param,
                                                                common_param);
}

}  // namespace

DiskSparseTermListDataCellInterfacePtr
DiskSparseTermListDataCellInterface::MakeInstance(float doc_retain_ratio,
                                                  uint32_t term_id_limit,
                                                  Allocator* allocator,
                                                  bool use_quantization,
                                                  QuantizationParamsPtr quantization_params,
                                                  uint32_t window_size,
                                                  const IOParamPtr& io_param,
                                                  const IndexCommonParam& common_param) {
    auto io_type_name = io_param->GetTypeName();
    if (io_type_name == IO_TYPE_VALUE_MMAP_IO) {
        return make_disk_sparse_term_list_datacell<MMapIO>(doc_retain_ratio,
                                                           term_id_limit,
                                                           allocator,
                                                           use_quantization,
                                                           std::move(quantization_params),
                                                           window_size,
                                                           io_param,
                                                           common_param);
    }
    if (io_type_name == IO_TYPE_VALUE_BUFFER_IO) {
        return make_disk_sparse_term_list_datacell<BufferIO>(doc_retain_ratio,
                                                             term_id_limit,
                                                             allocator,
                                                             use_quantization,
                                                             std::move(quantization_params),
                                                             window_size,
                                                             io_param,
                                                             common_param);
    }
    if (io_type_name == IO_TYPE_VALUE_ASYNC_IO) {
#if HAVE_LIBAIO
        return make_disk_sparse_term_list_datacell<AsyncIO>(doc_retain_ratio,
                                                            term_id_limit,
                                                            allocator,
                                                            use_quantization,
                                                            std::move(quantization_params),
                                                            window_size,
                                                            io_param,
                                                            common_param);
#else
        return make_disk_sparse_term_list_datacell<BufferIO>(doc_retain_ratio,
                                                             term_id_limit,
                                                             allocator,
                                                             use_quantization,
                                                             std::move(quantization_params),
                                                             window_size,
                                                             io_param,
                                                             common_param);
#endif
    }
    if (io_type_name == IO_TYPE_VALUE_READER_IO) {
        return make_disk_sparse_term_list_datacell<ReaderIO>(doc_retain_ratio,
                                                             term_id_limit,
                                                             allocator,
                                                             use_quantization,
                                                             std::move(quantization_params),
                                                             window_size,
                                                             io_param,
                                                             common_param);
    }
    throw VsagException(ErrorType::INVALID_ARGUMENT,
                        fmt::format("unsupported DiskSINDI term io type: {}", io_type_name));
}

template <typename IOTmpl>
DiskSparseTermListDataCell<IOTmpl>::DiskSparseTermListDataCell(
    float doc_retain_ratio,
    uint32_t term_id_limit,
    Allocator* allocator,
    bool use_quantization,
    QuantizationParamsPtr quantization_params,
    uint32_t window_size,
    IOParamPtr io_param,
    const IndexCommonParam& common_param)
    : doc_retain_ratio_(doc_retain_ratio),
      term_id_limit_(term_id_limit),
      allocator_(allocator),
      use_quantization_(use_quantization),
      quantization_params_(std::move(quantization_params)),
      window_size_(window_size),
      io_param_(std::move(io_param)),
      common_param_(common_param) {
}

template <typename IOTmpl>
void
DiskSparseTermListDataCell<IOTmpl>::InsertVector(const SparseVector& vector, uint32_t inner_id) {
    Vector<std::pair<uint32_t, float>> sorted_base(allocator_);
    sort_sparse_vector(vector, sorted_base);

    if (sorted_base.size() <= 1 || doc_retain_ratio_ == 1.0F) {
        // keep all
    } else {
        float total_mass = 0.0F;
        for (const auto& pair : sorted_base) {
            total_mass += pair.second;
        }
        float part_mass = total_mass * doc_retain_ratio_;
        float temp_mass = 0.0F;
        int pruned_doc_len = 0;
        while (temp_mass < part_mass && pruned_doc_len < static_cast<int>(sorted_base.size())) {
            temp_mass += sorted_base[pruned_doc_len++].second;
        }
        sorted_base.resize(pruned_doc_len);
    }

    for (const auto& pair : sorted_base) {
        auto term_id = pair.first;
        if (term_id > term_id_limit_) {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
                fmt::format("term id {} exceeds term_id_limit {}", term_id, term_id_limit_));
        }
        build_buffers_[term_id].push_back({inner_id, pair.second});
    }
    total_count_ = std::max(total_count_, static_cast<uint64_t>(inner_id) + 1);
}

template <typename IOTmpl>
void
DiskSparseTermListDataCell<IOTmpl>::FinalizeTermBuffers(uint32_t window_count) {
    std::unique_lock lock(term_buffers_mutex_);
    window_count_ = window_count;
    term_buffers_.clear();
    term_dict_.resize(term_id_limit_ + 1);

    for (auto& kv : build_buffers_) {
        uint32_t term_id = kv.first;
        auto& postings = kv.second;
        if (postings.empty()) {
            continue;
        }

        // Sort postings by inner_id for deterministic layout
        std::sort(postings.begin(), postings.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        TermBuffer tb;
        tb.window_offsets.resize(window_count + 1, 0);

        uint32_t prev_window = 0;
        uint32_t count = 0;
        for (const auto& posting : postings) {
            uint32_t doc_id = posting.first;
            uint32_t window_id = doc_id / window_size_;
            while (prev_window < window_id) {
                tb.window_offsets[prev_window + 1] = count;
                prev_window++;
            }
            tb.ids.push_back(static_cast<uint16_t>(doc_id % window_size_));
            count++;
        }
        while (prev_window < window_count) {
            tb.window_offsets[prev_window + 1] = count;
            prev_window++;
        }
        tb.window_offsets[window_count] = count;

        if (use_quantization_) {
            tb.values.resize(count);
            size_t idx = 0;
            for (const auto& posting : postings) {
                float x = (posting.second - quantization_params_->min_val) /
                          quantization_params_->diff * 255.0F;
                tb.values[idx++] = static_cast<uint8_t>(std::clamp(x, 0.0F, 255.0F));
            }
        } else {
            tb.values.resize(count * sizeof(float));
            size_t idx = 0;
            for (const auto& posting : postings) {
                std::memcpy(tb.values.data() + idx * sizeof(float), &posting.second, sizeof(float));
                idx++;
            }
        }

        term_buffers_[term_id] = std::move(tb);
    }

    build_buffers_.clear();
    this->BuildTermDict(term_dict_, 0);
    if constexpr (!std::is_same_v<IOTmpl, ReaderIO>) {
        this->WritePayloadToIO(0);
    }
}

template <typename IOTmpl>
uint64_t
DiskSparseTermListDataCell<IOTmpl>::ComputePayloadSize() const {
    uint64_t size = 0;
    if (term_buffers_.empty() && !term_dict_.empty()) {
        for (const auto& entry : term_dict_) {
            size += entry.posting_payload_size;
        }
        return size;
    }
    for (uint32_t term_id = 0; term_id <= term_id_limit_; ++term_id) {
        auto it = term_buffers_.find(term_id);
        if (it == term_buffers_.end() || it->second.ids.empty()) {
            continue;
        }
        const auto& tb = it->second;
        size += tb.window_offsets.size() * sizeof(uint32_t);
        size += tb.ids.size() * sizeof(uint16_t);
        size += (4 - (tb.ids.size() * sizeof(uint16_t)) % 4) % 4;
        size += tb.values.size();
    }
    return size;
}

template <typename IOTmpl>
void
DiskSparseTermListDataCell<IOTmpl>::BuildTermDict(std::vector<DiskTermEntry>& term_dict,
                                                  uint64_t payload_base_offset) const {
    term_dict.resize(term_id_limit_ + 1);
    uint64_t current_offset = payload_base_offset;

    for (uint32_t term_id = 0; term_id <= term_id_limit_; ++term_id) {
        auto it = term_buffers_.find(term_id);
        if (it == term_buffers_.end() || it->second.ids.empty()) {
            term_dict[term_id] = DiskTermEntry{0, 0, 0};
            continue;
        }

        const auto& tb = it->second;
        DiskTermEntry entry;
        entry.posting_payload_offset = current_offset;
        entry.term_num = static_cast<uint32_t>(tb.ids.size());

        current_offset += tb.window_offsets.size() * sizeof(uint32_t);
        current_offset += tb.ids.size() * sizeof(uint16_t);
        size_t ids_padding = (4 - (tb.ids.size() * sizeof(uint16_t)) % 4) % 4;
        current_offset += ids_padding;
        current_offset += tb.values.size();

        entry.posting_payload_size =
            static_cast<uint32_t>(current_offset - entry.posting_payload_offset);
        term_dict[term_id] = entry;
    }
}

template <typename IOTmpl>
void
DiskSparseTermListDataCell<IOTmpl>::WritePayload(StreamWriter& writer) const {
    for (uint32_t term_id = 0; term_id <= term_id_limit_; ++term_id) {
        auto it = term_buffers_.find(term_id);
        if (it == term_buffers_.end() || it->second.ids.empty()) {
            continue;
        }

        const auto& tb = it->second;

        // Write window_offsets
        writer.Write(reinterpret_cast<const char*>(tb.window_offsets.data()),
                     tb.window_offsets.size() * sizeof(uint32_t));

        // Write ids
        writer.Write(reinterpret_cast<const char*>(tb.ids.data()),
                     tb.ids.size() * sizeof(uint16_t));

        // Pad ids to 4-byte boundary
        size_t ids_padding = (4 - (tb.ids.size() * sizeof(uint16_t)) % 4) % 4;
        if (ids_padding > 0) {
            char pad[2] = {0, 0};
            writer.Write(pad, ids_padding);
        }

        // Write values
        writer.Write(reinterpret_cast<const char*>(tb.values.data()), tb.values.size());
    }
}

template <typename IOTmpl>
void
DiskSparseTermListDataCell<IOTmpl>::WriteTermDictAndPayload(StreamWriter& writer,
                                                            uint64_t payload_base_offset) const {
    (void)payload_base_offset;
    std::vector<DiskTermEntry> term_dict;
    if (term_buffers_.empty() && !term_dict_.empty()) {
        term_dict = term_dict_;
    } else {
        this->BuildTermDict(term_dict, 0);
    }
    for (uint32_t i = 0; i <= term_id_limit_; ++i) {
        StreamWriter::WriteObj(writer, term_dict[i]);
    }

    auto payload_size = this->ComputePayloadSize();
    if (payload_size == 0) {
        return;
    }
    if (io_ == nullptr) {
        this->WritePayload(writer);
        return;
    }

    constexpr uint64_t serialize_buffer_size = uint64_t{2} * 1024 * 1024;
    ByteBuffer buffer(serialize_buffer_size, allocator_);
    uint64_t offset = 0;
    while (offset < payload_size) {
        auto current_size = std::min(serialize_buffer_size, payload_size - offset);
        io_->Read(current_size, offset, buffer.data);
        writer.Write(reinterpret_cast<const char*>(buffer.data), current_size);
        offset += current_size;
    }
}

template <typename IOTmpl>
void
DiskSparseTermListDataCell<IOTmpl>::Deserialize(StreamReader& reader,
                                                uint64_t term_dict_size,
                                                uint32_t window_count) {
    std::unique_lock lock(term_buffers_mutex_);
    window_count_ = window_count;
    CHECK_ARGUMENT(term_dict_size % sizeof(DiskTermEntry) == 0,
                   fmt::format("invalid DiskSINDI term_dict_size {}", term_dict_size));
    auto term_dict_count = static_cast<uint32_t>(term_dict_size / sizeof(DiskTermEntry));
    std::vector<DiskTermEntry> term_dict(term_dict_count);
    reader.Read(reinterpret_cast<char*>(term_dict.data()), term_dict_size);
    term_dict_ = term_dict;
    term_buffers_.clear();  // Posting payload is loaded lazily for pruned query terms.
}

template <typename IOTmpl>
void
DiskSparseTermListDataCell<IOTmpl>::InitIO(const IOParamPtr& io_param) {
    if (io_param != nullptr) {
        io_param_ = io_param;
    }
    if (io_ == nullptr && io_param_ != nullptr) {
        io_ = std::make_shared<IOTmpl>(io_param_, common_param_);
    }
    if (io_ == nullptr) {
        return;
    }
    io_->InitIO(io_param_);
}

template <typename IOTmpl>
void
DiskSparseTermListDataCell<IOTmpl>::WritePayloadToIO(uint64_t payload_base_offset) {
    (void)payload_base_offset;
    if (io_ == nullptr) {
        this->InitIO(io_param_);
    }
    if (io_ == nullptr) {
        return;
    }

    io_->Resize(this->ComputePayloadSize());
    std::vector<DiskTermEntry> term_dict;
    this->BuildTermDict(term_dict, 0);
    for (uint32_t term_id = 0; term_id <= term_id_limit_; ++term_id) {
        auto it = term_buffers_.find(term_id);
        if (it == term_buffers_.end() || it->second.ids.empty()) {
            continue;
        }

        const auto& tb = it->second;
        auto offset = term_dict[term_id].posting_payload_offset;
        auto window_offsets_size = tb.window_offsets.size() * sizeof(uint32_t);
        io_->Write(reinterpret_cast<const uint8_t*>(tb.window_offsets.data()),
                   window_offsets_size,
                   offset);
        offset += window_offsets_size;

        auto ids_size = tb.ids.size() * sizeof(uint16_t);
        io_->Write(reinterpret_cast<const uint8_t*>(tb.ids.data()), ids_size, offset);
        offset += ids_size;

        size_t ids_padding = (4 - ids_size % 4) % 4;
        if (ids_padding > 0) {
            uint8_t pad[2] = {0, 0};
            io_->Write(pad, ids_padding, offset);
            offset += ids_padding;
        }

        io_->Write(tb.values.data(), tb.values.size(), offset);
    }
}

template <typename IOTmpl>
void
DiskSparseTermListDataCell<IOTmpl>::WritePayloadToIO(StreamReader& reader,
                                                     uint64_t payload_offset,
                                                     uint64_t payload_size) {
    if (payload_size == 0) {
        return;
    }
    if (io_ == nullptr) {
        this->InitIO(io_param_);
    }
    if (io_ == nullptr) {
        return;
    }

    io_->Resize(payload_size);
    constexpr uint64_t serialize_buffer_size = uint64_t{2} * 1024 * 1024;
    ByteBuffer buffer(serialize_buffer_size, allocator_);
    uint64_t copied_size = 0;
    reader.PushSeek(payload_offset);
    while (copied_size < payload_size) {
        auto current_size = std::min(serialize_buffer_size, payload_size - copied_size);
        reader.Read(reinterpret_cast<char*>(buffer.data), current_size);
        io_->Write(buffer.data, current_size, copied_size);
        copied_size += current_size;
    }
    reader.PopSeek();
}

template <typename IOTmpl>
void
DiskSparseTermListDataCell<IOTmpl>::SetIO(const std::shared_ptr<Reader>& reader,
                                          uint64_t payload_offset,
                                          uint64_t payload_size) {
    if constexpr (std::is_same_v<IOTmpl, ReaderIO>) {
        auto reader_param = std::make_shared<ReaderIOParameter>();
        reader_param->reader = reader;
        io_param_ = reader_param;
        if (io_ == nullptr) {
            io_ = std::make_shared<ReaderIO>(allocator_);
        }
        io_->InitIO(reader_param);
        io_->start_ = payload_offset;
        io_->size_ = payload_size;
    }
}

template <typename IOTmpl>
void
DiskSparseTermListDataCell<IOTmpl>::LoadQueryTerms(const Vector<uint32_t>& query_term_ids) {
    if (io_ == nullptr) {
        return;
    }
    std::unique_lock lock(term_buffers_mutex_);
    for (uint32_t term_id : query_term_ids) {
        if (term_id >= term_dict_.size()) {
            logger::warn("term_id {} out of range in LoadQueryTerms", term_id);
            continue;
        }

        const auto& entry = term_dict_[term_id];
        if (entry.term_num == 0) {
            continue;
        }

        // Skip if already loaded
        if (term_buffers_.find(term_id) != term_buffers_.end()) {
            continue;
        }

        TermBuffer tb;
        tb.window_offsets.resize(window_count_ + 1);

        if (use_quantization_) {
            tb.values.resize(entry.term_num);
        } else {
            tb.values.resize(entry.term_num * sizeof(float));
        }
        tb.ids.resize(entry.term_num);

        // Read window_offsets
        io_->Read(tb.window_offsets.size() * sizeof(uint32_t),
                  entry.posting_payload_offset,
                  reinterpret_cast<uint8_t*>(tb.window_offsets.data()));

        // Read ids
        io_->Read(tb.ids.size() * sizeof(uint16_t),
                  entry.posting_payload_offset + tb.window_offsets.size() * sizeof(uint32_t),
                  reinterpret_cast<uint8_t*>(tb.ids.data()));

        // Skip padding
        size_t ids_padding = (4 - (tb.ids.size() * sizeof(uint16_t)) % 4) % 4;
        uint64_t values_offset = entry.posting_payload_offset +
                                 tb.window_offsets.size() * sizeof(uint32_t) +
                                 tb.ids.size() * sizeof(uint16_t) + ids_padding;

        // Read values
        io_->Read(tb.values.size(), values_offset, tb.values.data());

        bool valid_payload =
            tb.window_offsets.front() == 0 && tb.window_offsets.back() == entry.term_num;
        for (uint64_t i = 1; valid_payload && i < tb.window_offsets.size(); ++i) {
            valid_payload = tb.window_offsets[i - 1] <= tb.window_offsets[i] &&
                            tb.window_offsets[i] <= entry.term_num;
        }
        for (uint64_t i = 0; valid_payload && i < tb.ids.size(); ++i) {
            valid_payload = tb.ids[i] < window_size_;
        }
        if (not valid_payload) {
            throw VsagException(
                ErrorType::INTERNAL_ERROR,
                fmt::format("invalid DiskSINDI term payload for term {} from io", term_id));
        }

        term_buffers_[term_id] = std::move(tb);
    }
}

template <typename IOTmpl>
QueryTermBuffers
DiskSparseTermListDataCell<IOTmpl>::LoadQueryTermBuffers(
    const Vector<uint32_t>& query_term_ids) const {
    QueryTermBuffers query_term_buffers;
    if (io_ == nullptr) {
        return query_term_buffers;
    }

    std::shared_lock lock(term_buffers_mutex_);
    query_term_buffers.reserve(query_term_ids.size());
    for (uint32_t term_id : query_term_ids) {
        if (term_id >= term_dict_.size()) {
            logger::warn("term_id {} out of range in LoadQueryTermBuffers", term_id);
            continue;
        }

        const auto& entry = term_dict_[term_id];
        if (entry.term_num == 0) {
            continue;
        }

        TermBuffer tb;
        tb.window_offsets.resize(window_count_ + 1);
        tb.ids.resize(entry.term_num);
        if (use_quantization_) {
            tb.values.resize(entry.term_num);
        } else {
            tb.values.resize(entry.term_num * sizeof(float));
        }

        io_->Read(tb.window_offsets.size() * sizeof(uint32_t),
                  entry.posting_payload_offset,
                  reinterpret_cast<uint8_t*>(tb.window_offsets.data()));

        auto ids_offset =
            entry.posting_payload_offset + tb.window_offsets.size() * sizeof(uint32_t);
        io_->Read(tb.ids.size() * sizeof(uint16_t),
                  ids_offset,
                  reinterpret_cast<uint8_t*>(tb.ids.data()));

        size_t ids_padding = (4 - (tb.ids.size() * sizeof(uint16_t)) % 4) % 4;
        uint64_t values_offset = ids_offset + tb.ids.size() * sizeof(uint16_t) + ids_padding;
        io_->Read(tb.values.size(), values_offset, tb.values.data());

        bool valid_payload =
            tb.window_offsets.front() == 0 && tb.window_offsets.back() == entry.term_num;
        for (uint64_t i = 1; valid_payload && i < tb.window_offsets.size(); ++i) {
            valid_payload = tb.window_offsets[i - 1] <= tb.window_offsets[i] &&
                            tb.window_offsets[i] <= entry.term_num;
        }
        for (uint64_t i = 0; valid_payload && i < tb.ids.size(); ++i) {
            valid_payload = tb.ids[i] < window_size_;
        }
        if (not valid_payload) {
            throw VsagException(
                ErrorType::INTERNAL_ERROR,
                fmt::format("invalid DiskSINDI term payload for term {} from io", term_id));
        }

        query_term_buffers.emplace(term_id, std::move(tb));
    }

    return query_term_buffers;
}

template <typename IOTmpl>
void
DiskSparseTermListDataCell<IOTmpl>::InsertHeapByWindowKnn(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    bool with_filter,
    const QueryTermBuffers& query_term_buffers) const {
    if (with_filter) {
        this->template InsertHeapByWindow<InnerSearchMode::KNN_SEARCH,
                                          InnerSearchType::WITH_FILTER>(
            dists, window_id, computer, heap, param, offset_id, query_term_buffers);
    } else {
        this->template InsertHeapByWindow<InnerSearchMode::KNN_SEARCH, InnerSearchType::PURE>(
            dists, window_id, computer, heap, param, offset_id, query_term_buffers);
    }
}

template <typename IOTmpl>
void
DiskSparseTermListDataCell<IOTmpl>::InsertHeapByDistsKnn(float* dists,
                                                         uint32_t dists_size,
                                                         MaxHeap& heap,
                                                         const InnerSearchParam& param,
                                                         uint32_t offset_id,
                                                         bool with_filter) const {
    if (with_filter) {
        this->template InsertHeapByDists<InnerSearchMode::KNN_SEARCH, InnerSearchType::WITH_FILTER>(
            dists, dists_size, heap, param, offset_id);
    } else {
        this->template InsertHeapByDists<InnerSearchMode::KNN_SEARCH, InnerSearchType::PURE>(
            dists, dists_size, heap, param, offset_id);
    }
}

template <typename IOTmpl>
const TermBuffer*
DiskSparseTermListDataCell<IOTmpl>::GetTermBuffer(uint32_t term_id) const {
    std::shared_lock lock(term_buffers_mutex_);
    auto it = term_buffers_.find(term_id);
    if (it == term_buffers_.end()) {
        return nullptr;
    }
    return &it->second;
}

template <typename IOTmpl>
const TermBuffer*
DiskSparseTermListDataCell<IOTmpl>::GetTermBuffer(
    uint32_t term_id, const QueryTermBuffers& query_term_buffers) const {
    auto query_it = query_term_buffers.find(term_id);
    if (query_it != query_term_buffers.end()) {
        return &query_it->second;
    }
    auto it = term_buffers_.find(term_id);
    if (it == term_buffers_.end()) {
        return nullptr;
    }
    return &it->second;
}

template <typename IOTmpl>
uint64_t
DiskSparseTermListDataCell<IOTmpl>::GetMemoryUsage() const {
    uint64_t memory = sizeof(DiskSparseTermListDataCell<IOTmpl>);
    // Only count term_dict_ and metadata, not loaded posting payloads
    memory += term_dict_.size() * sizeof(DiskTermEntry);
    return memory;
}

template <typename IOTmpl>
void
DiskSparseTermListDataCell<IOTmpl>::QueryWindow(float* dists,
                                                uint32_t window_id,
                                                const SparseTermComputerPtr& computer,
                                                const QueryTermBuffers& query_term_buffers) const {
    std::shared_lock lock(term_buffers_mutex_);
    while (computer->HasNextTerm()) {
        auto it = computer->NextTermIter();
        auto term = computer->GetTerm(it);
        auto* tb = this->GetTermBuffer(term, query_term_buffers);
        if (tb == nullptr) {
            continue;
        }
        if (window_id >= window_count_) {
            continue;
        }
        uint32_t start = tb->window_offsets[window_id];
        uint32_t end = tb->window_offsets[window_id + 1];
        uint32_t count = end - start;
        if (count == 0) {
            continue;
        }

        if (use_quantization_) {
            computer->ScanForAccumulate(
                it, tb->ids.data() + start, tb->values.data() + start, count, dists);
        } else {
            computer->ScanForAccumulate(
                it,
                tb->ids.data() + start,
                reinterpret_cast<const float*>(tb->values.data() + start * sizeof(float)),
                count,
                dists);
        }
    }
    computer->ResetTerm();
}

template <typename IOTmpl>
void
DiskSparseTermListDataCell<IOTmpl>::GetSparseVector(uint32_t inner_id,
                                                    SparseVector* data,
                                                    Allocator* specified_allocator) {
    std::shared_lock lock(term_buffers_mutex_);
    Allocator* allocator = specified_allocator != nullptr ? specified_allocator : allocator_;
    Vector<uint32_t> ids(allocator);
    Vector<float> vals(allocator);

    for (const auto& kv : term_buffers_) {
        uint32_t term_id = kv.first;
        const auto& tb = kv.second;
        uint32_t window_id = inner_id / window_size_;
        uint32_t local_id = inner_id % window_size_;
        if (window_id >= window_count_) {
            continue;
        }
        uint32_t start = tb.window_offsets[window_id];
        uint32_t end = tb.window_offsets[window_id + 1];
        for (uint32_t i = start; i < end; i++) {
            if (tb.ids[i] == local_id) {
                ids.push_back(term_id);
                float v;
                if (use_quantization_) {
                    v = static_cast<float>(tb.values[i]) / 255.0F * quantization_params_->diff +
                        quantization_params_->min_val;
                } else {
                    std::memcpy(&v, tb.values.data() + i * sizeof(float), sizeof(float));
                }
                vals.push_back(v);
                break;
            }
        }
    }

    data->len_ = ids.size();
    if (data->len_ == 0) {
        data->ids_ = nullptr;
        data->vals_ = nullptr;
        return;
    }
    data->ids_ = static_cast<uint32_t*>(allocator->Allocate(sizeof(uint32_t) * data->len_));
    data->vals_ = static_cast<float*>(allocator->Allocate(sizeof(float) * data->len_));
    memcpy(data->ids_, ids.data(), data->len_ * sizeof(uint32_t));
    memcpy(data->vals_, vals.data(), data->len_ * sizeof(float));
}

template <typename IOTmpl>
float
DiskSparseTermListDataCell<IOTmpl>::CalcDistanceByInnerId(
    const SparseTermComputerPtr& computer,
    uint32_t base_id,
    const QueryTermBuffers& query_term_buffers) {
    std::shared_lock lock(term_buffers_mutex_);
    float ip = 0;
    while (computer->HasNextTerm()) {
        auto it = computer->NextTermIter();
        auto term = computer->GetTerm(it);
        auto* tb = this->GetTermBuffer(term, query_term_buffers);
        if (tb == nullptr) {
            continue;
        }
        uint32_t window_id = base_id / window_size_;
        uint32_t local_id = static_cast<uint16_t>(base_id % window_size_);
        if (window_id >= window_count_) {
            continue;
        }
        uint32_t start = tb->window_offsets[window_id];
        uint32_t end = tb->window_offsets[window_id + 1];
        for (uint32_t i = start; i < end; i++) {
            if (tb->ids[i] == local_id) {
                float val;
                if (use_quantization_) {
                    val = static_cast<float>(tb->values[i]) / 255.0F * quantization_params_->diff +
                          quantization_params_->min_val;
                } else {
                    std::memcpy(&val, tb->values.data() + i * sizeof(float), sizeof(float));
                }
                ip += computer->sorted_query_[it].second * val * -1.0F;
                break;
            }
        }
    }
    computer->ResetTerm();
    return 1.0F + ip;
}

template <typename IOTmpl>
void
DiskSparseTermListDataCell<IOTmpl>::Encode(float val, uint8_t* dst) const {
    if (use_quantization_) {
        float x = (val - quantization_params_->min_val) / quantization_params_->diff * 255.0F;
        *dst = static_cast<uint8_t>(std::clamp(x, 0.0F, 255.0F));
    } else {
        std::memcpy(dst, &val, sizeof(float));
    }
}

template <typename IOTmpl>
void
DiskSparseTermListDataCell<IOTmpl>::Decode(const uint8_t* src, size_t size, float* dst) const {
    if (use_quantization_) {
        float x = static_cast<float>(*src) / 255.0F * quantization_params_->diff +
                  quantization_params_->min_val;
        *dst = x;
    } else {
        std::memcpy(dst, src, sizeof(float));
    }
}

// Explicit instantiations
template class DiskSparseTermListDataCell<MMapIO>;
template class DiskSparseTermListDataCell<BufferIO>;
template class DiskSparseTermListDataCell<ReaderIO>;
#if HAVE_LIBAIO
template class DiskSparseTermListDataCell<AsyncIO>;
#endif

template void
DiskSparseTermListDataCell<MMapIO>::InsertHeapByWindow<InnerSearchMode::KNN_SEARCH,
                                                       InnerSearchType::PURE>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers) const;

template void
DiskSparseTermListDataCell<MMapIO>::InsertHeapByWindow<InnerSearchMode::KNN_SEARCH,
                                                       InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers) const;

template void
DiskSparseTermListDataCell<MMapIO>::InsertHeapByWindow<InnerSearchMode::RANGE_SEARCH,
                                                       InnerSearchType::PURE>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers) const;

template void
DiskSparseTermListDataCell<MMapIO>::InsertHeapByWindow<InnerSearchMode::RANGE_SEARCH,
                                                       InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers) const;

template void
DiskSparseTermListDataCell<MMapIO>::InsertHeapByDists<InnerSearchMode::KNN_SEARCH,
                                                      InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
DiskSparseTermListDataCell<MMapIO>::InsertHeapByDists<InnerSearchMode::KNN_SEARCH,
                                                      InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
DiskSparseTermListDataCell<MMapIO>::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH,
                                                      InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
DiskSparseTermListDataCell<MMapIO>::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH,
                                                      InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
DiskSparseTermListDataCell<BufferIO>::InsertHeapByWindow<InnerSearchMode::KNN_SEARCH,
                                                         InnerSearchType::PURE>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers) const;

template void
DiskSparseTermListDataCell<BufferIO>::InsertHeapByWindow<InnerSearchMode::KNN_SEARCH,
                                                         InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers) const;

template void
DiskSparseTermListDataCell<BufferIO>::InsertHeapByWindow<InnerSearchMode::RANGE_SEARCH,
                                                         InnerSearchType::PURE>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers) const;

template void
DiskSparseTermListDataCell<BufferIO>::InsertHeapByWindow<InnerSearchMode::RANGE_SEARCH,
                                                         InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers) const;

template void
DiskSparseTermListDataCell<BufferIO>::InsertHeapByDists<InnerSearchMode::KNN_SEARCH,
                                                        InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
DiskSparseTermListDataCell<BufferIO>::InsertHeapByDists<InnerSearchMode::KNN_SEARCH,
                                                        InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
DiskSparseTermListDataCell<BufferIO>::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH,
                                                        InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
DiskSparseTermListDataCell<BufferIO>::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH,
                                                        InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
DiskSparseTermListDataCell<ReaderIO>::InsertHeapByWindow<InnerSearchMode::KNN_SEARCH,
                                                         InnerSearchType::PURE>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers) const;

template void
DiskSparseTermListDataCell<ReaderIO>::InsertHeapByWindow<InnerSearchMode::KNN_SEARCH,
                                                         InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers) const;

template void
DiskSparseTermListDataCell<ReaderIO>::InsertHeapByWindow<InnerSearchMode::RANGE_SEARCH,
                                                         InnerSearchType::PURE>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers) const;

template void
DiskSparseTermListDataCell<ReaderIO>::InsertHeapByWindow<InnerSearchMode::RANGE_SEARCH,
                                                         InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers) const;

template void
DiskSparseTermListDataCell<ReaderIO>::InsertHeapByDists<InnerSearchMode::KNN_SEARCH,
                                                        InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
DiskSparseTermListDataCell<ReaderIO>::InsertHeapByDists<InnerSearchMode::KNN_SEARCH,
                                                        InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
DiskSparseTermListDataCell<ReaderIO>::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH,
                                                        InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
DiskSparseTermListDataCell<ReaderIO>::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH,
                                                        InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

}  // namespace vsag
