
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

#include "disk_sindi_term_datacell.h"

#include <fmt/format.h>

#include <limits>
#include <type_traits>

#include "datacell/sindi_datacell_utils.h"
#include "io/async_io/async_io_parameter.h"
#include "io/io_headers.h"
#include "io/reader_io/reader_io_parameter.h"
#include "simd/fp16_simd.h"
#include "utils/byte_buffer.h"
#include "utils/util_functions.h"
#include "vsag_exception.h"

namespace vsag {

namespace {

template <typename IOTmpl>
DiskSindiTermDataCellInterfacePtr
make_disk_sindi_term_datacell(uint32_t term_id_limit,
                              Allocator* allocator,
                              SparseValueQuantizationType sparse_value_quant_type,
                              QuantizationParamsPtr quantization_params,
                              uint32_t window_size,
                              const IOParamPtr& io_param,
                              const IndexCommonParam& common_param) {
    return std::make_shared<DiskSindiTermDataCell<IOTmpl>>(term_id_limit,
                                                           allocator,
                                                           sparse_value_quant_type,
                                                           std::move(quantization_params),
                                                           window_size,
                                                           io_param,
                                                           common_param);
}

IOParamPtr
make_buffer_io_param_from_async(const IOParamPtr& io_param) {
    auto async_param = std::dynamic_pointer_cast<AsyncIOParameter>(io_param);
    CHECK_ARGUMENT(async_param != nullptr, "async_io parameter is invalid");

    auto buffer_param = std::make_shared<BufferIOParameter>();
    buffer_param->path_ = async_param->path_;
    return buffer_param;
}

}  // namespace

DiskSindiTermDataCellInterfacePtr
DiskSindiTermDataCellInterface::MakeInstance(uint32_t term_id_limit,
                                             Allocator* allocator,
                                             SparseValueQuantizationType sparse_value_quant_type,
                                             QuantizationParamsPtr quantization_params,
                                             uint32_t window_size,
                                             const IOParamPtr& io_param,
                                             const IndexCommonParam& common_param) {
    CHECK_ARGUMENT(io_param != nullptr, "invalid term io parameter");
    auto io_type_name = io_param->GetTypeName();
    if (io_type_name == IO_TYPE_VALUE_MMAP_IO) {
        return make_disk_sindi_term_datacell<MMapIO>(term_id_limit,
                                                     allocator,
                                                     sparse_value_quant_type,
                                                     std::move(quantization_params),
                                                     window_size,
                                                     io_param,
                                                     common_param);
    }
    if (io_type_name == IO_TYPE_VALUE_BUFFER_IO) {
        return make_disk_sindi_term_datacell<BufferIO>(term_id_limit,
                                                       allocator,
                                                       sparse_value_quant_type,
                                                       std::move(quantization_params),
                                                       window_size,
                                                       io_param,
                                                       common_param);
    }
    if (io_type_name == IO_TYPE_VALUE_ASYNC_IO) {
#if HAVE_LIBAIO
        return make_disk_sindi_term_datacell<AsyncIO>(term_id_limit,
                                                      allocator,
                                                      sparse_value_quant_type,
                                                      std::move(quantization_params),
                                                      window_size,
                                                      io_param,
                                                      common_param);
#else
        return make_disk_sindi_term_datacell<BufferIO>(term_id_limit,
                                                       allocator,
                                                       sparse_value_quant_type,
                                                       std::move(quantization_params),
                                                       window_size,
                                                       make_buffer_io_param_from_async(io_param),
                                                       common_param);
#endif
    }
    if (io_type_name == IO_TYPE_VALUE_READER_IO) {
        return make_disk_sindi_term_datacell<ReaderIO>(term_id_limit,
                                                       allocator,
                                                       sparse_value_quant_type,
                                                       std::move(quantization_params),
                                                       window_size,
                                                       io_param,
                                                       common_param);
    }
    throw VsagException(ErrorType::INVALID_ARGUMENT,
                        fmt::format("unsupported SINDIV2 term io type: {}", io_type_name));
}

template <typename IOTmpl>
DiskSindiTermDataCell<IOTmpl>::DiskSindiTermDataCell(
    uint32_t term_id_limit,
    Allocator* allocator,
    SparseValueQuantizationType sparse_value_quant_type,
    QuantizationParamsPtr quantization_params,
    uint32_t window_size,
    IOParamPtr io_param,
    IndexCommonParam common_param)
    : term_id_limit_(term_id_limit),
      allocator_(allocator),
      sparse_value_quant_type_(sparse_value_quant_type),
      quantization_params_(std::move(quantization_params)),
      window_size_(window_size),
      io_param_(std::move(io_param)),
      common_param_(std::move(common_param)) {
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        window_size_ > 0 &&
            window_size_ <= static_cast<uint32_t>(std::numeric_limits<uint16_t>::max()) + 1,
        "window_size must be in (0, 65536]");
    if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
        CHECK_ARGUMENT(quantization_params_ != nullptr,
                       "quantization_params must not be null when quantization is enabled");
        CHECK_ARGUMENT(quantization_params_->diff != 0.0F,
                       "quantization diff must not be zero when quantization is enabled");
    }
}

template <typename IOTmpl>
void
DiskSindiTermDataCell<IOTmpl>::SerializeTermLayout(StreamWriter& writer,
                                                   uint32_t term_dict_count) const {
    CHECK_ARGUMENT(term_dict_count == term_dict_.size(),
                   "SINDI_V2 term dict count does not match disk data cell");
    StreamWriter::WriteVector(writer, term_dict_);
    if (not payload_layout_.IsBound()) {
        CHECK_ARGUMENT(term_dict_.empty(), "SINDIV2 term data cell is not bound to IO");
        StreamWriter::WriteObj(writer, uint64_t{0});
        return;
    }
    payload_layout_.Serialize(writer);
}

template <typename IOTmpl>
void
DiskSindiTermDataCell<IOTmpl>::DeserializeTermLayout(StreamReader& reader,
                                                     uint32_t window_count,
                                                     uint64_t total_count) {
    std::unique_lock lock(term_layout_mutex_);
    window_count_ = window_count;
    total_count_ = total_count;
    term_dict_ = sindi_datacell_utils::DeserializeTermDictionary(reader, term_id_limit_);
    this->InitIO(io_param_);
    CHECK_ARGUMENT(payload_layout_.IsBound(), "SINDIV2 term data cell has no IO");
    payload_layout_.Deserialize(reader);
    payload_size_ = payload_layout_.GetByteSize();
    this->ValidateBoundLayout(payload_size_);
    payloads_validated_ = false;
    if constexpr (not std::is_same_v<IOTmpl, ReaderIO>) {
        this->ValidateBoundPayloads();
        payloads_validated_ = true;
    }
}

template <typename IOTmpl>
void
DiskSindiTermDataCell<IOTmpl>::InitIO(const IOParamPtr& io_param) {
    if (io_param != nullptr) {
        io_param_ = io_param;
    }
    if (not payload_layout_.IsBound() && io_param_ != nullptr) {
        auto io = std::make_shared<IOTmpl>(io_param_, common_param_);
        if constexpr (std::is_same_v<IOTmpl, ReaderIO>) {
            const auto reader_param = std::dynamic_pointer_cast<ReaderIOParameter>(io_param_);
            if (reader_param != nullptr && reader_param->reader != nullptr) {
                io->InitIO(io_param_);
            }
        } else {
            io->InitIO(io_param_);
        }
        payload_layout_.SetIO(std::move(io));
        return;
    }
    if constexpr (std::is_same_v<IOTmpl, ReaderIO>) {
        const auto reader_param = std::dynamic_pointer_cast<ReaderIOParameter>(io_param_);
        if (reader_param == nullptr || reader_param->reader == nullptr) {
            return;
        }
    }
    if (payload_layout_.IsBound()) {
        payload_layout_.InitIO(io_param_);
    }
}

template <typename IOTmpl>
void
DiskSindiTermDataCell<IOTmpl>::SetIO(const std::shared_ptr<Reader>& reader) {
    if constexpr (std::is_same_v<IOTmpl, ReaderIO>) {
        std::unique_lock lock(term_layout_mutex_);
        auto reader_param = std::make_shared<ReaderIOParameter>();
        reader_param->reader = reader;
        io_param_ = reader_param;
        if (not payload_layout_.IsBound()) {
            auto io = std::make_shared<ReaderIO>(allocator_);
            io->InitIO(reader_param);
            payload_layout_.SetIO(std::move(io));
        } else {
            payload_layout_.InitIO(reader_param);
        }
        payload_size_ = payload_layout_.GetByteSize();
        this->ValidateBoundLayout(payload_size_);
        this->ValidateBoundPayloads();
        payloads_validated_ = true;
    }
}

template <typename IOTmpl>
void
DiskSindiTermDataCell<IOTmpl>::ValidateBoundLayout(uint64_t payload_size) const {
    sindi_datacell_utils::ValidateTermDict(term_dict_, payload_size);
}

template <typename IOTmpl>
void
DiskSindiTermDataCell<IOTmpl>::ValidateBoundPayloads() const {
    CHECK_ARGUMENT(payload_layout_.IsBound(), "SINDI_V2 term payload IO is not bound");
    const auto value_code_size = sindi_datacell_utils::GetValueCodeSize(sparse_value_quant_type_);
    Vector<uint8_t> payload(allocator_);
    for (const auto& entry : term_dict_) {
        if (entry.posting_count == 0) {
            continue;
        }

        const uint8_t* payload_data = nullptr;
        bool need_release = false;
        if constexpr (std::is_same_v<IOTmpl, MMapIO>) {
            payload_data = payload_layout_.Read(
                entry.posting_payload_offset, entry.posting_payload_size, need_release);
            CHECK_ARGUMENT(payload_data != nullptr,
                           "failed to access mmap SINDI_V2 term payload during validation");
        } else {
            payload.resize(entry.posting_payload_size);
            const bool read_succeeded = payload_layout_.Read(
                entry.posting_payload_offset, entry.posting_payload_size, payload.data());
            CHECK_ARGUMENT(read_succeeded,
                           "failed to read SINDI_V2 term payload during validation");
            payload_data = payload.data();
        }

        try {
            (void)sindi_datacell_utils::ViewTermPayload(payload_data,
                                                        entry.posting_payload_size,
                                                        entry,
                                                        window_count_,
                                                        window_size_,
                                                        total_count_,
                                                        value_code_size,
                                                        allocator_);
        } catch (...) {
            if (need_release) {
                payload_layout_.Release(payload_data);
            }
            throw;
        }
        if (need_release) {
            payload_layout_.Release(payload_data);
        }
    }
}

template <typename IOTmpl>
QueryTermBuffers
DiskSindiTermDataCell<IOTmpl>::LoadQueryTermBuffers(const Vector<uint32_t>& query_term_ids,
                                                    Allocator* query_allocator) const {
    auto* allocator = query_allocator == nullptr ? allocator_ : query_allocator;
    QueryTermBuffers query_term_buffers(allocator);
    if (not payload_layout_.IsBound()) {
        return query_term_buffers;
    }

    struct QueryTermReadPlan {
        uint32_t term_id{0};
        DiskTermEntry entry{};
    };

    Vector<QueryTermReadPlan> read_plans(allocator);
    UnorderedSet<uint32_t> planned_terms(allocator);
    read_plans.reserve(query_term_ids.size());
    planned_terms.reserve(query_term_ids.size());
    query_term_buffers.reserve(query_term_ids.size());
    uint32_t window_count = 0;
    uint32_t window_size = 0;
    uint64_t total_count = 0;
    uint64_t payload_size = 0;
    SparseValueQuantizationType sparse_value_quant_type = SparseValueQuantizationType::FP32;
    {
        std::shared_lock lock(term_layout_mutex_);
        window_count = window_count_;
        window_size = window_size_;
        total_count = total_count_;
        payload_size = payload_size_;
        CHECK_ARGUMENT(payloads_validated_, "SINDI_V2 term payloads have not been validated");
        sparse_value_quant_type = sparse_value_quant_type_;
        for (uint32_t term_id : query_term_ids) {
            if (term_id >= term_dict_.size() || planned_terms.contains(term_id)) {
                continue;
            }

            const auto entry = term_dict_[term_id];
            if (entry.posting_count == 0) {
                continue;
            }
            planned_terms.insert(term_id);
            read_plans.push_back({term_id, entry});
        }
    }

    const auto validate_entry = [payload_size](const QueryTermReadPlan& plan) {
        const auto& entry = plan.entry;
        const bool valid_entry =
            entry.posting_payload_offset <= payload_size &&
            entry.posting_payload_size <= payload_size - entry.posting_payload_offset;
        CHECK_ARGUMENT(
            valid_entry,
            fmt::format("invalid SINDI_V2 term dictionary entry for term {}", plan.term_id));
    };
    const auto value_code_size = sindi_datacell_utils::GetValueCodeSize(sparse_value_quant_type);

    if constexpr (not std::is_same_v<IOTmpl, MMapIO>) {
        Vector<uint64_t> sizes(allocator);
        Vector<uint64_t> offsets(allocator);
        Vector<uint64_t> buffer_offsets(allocator);
        sizes.reserve(read_plans.size());
        offsets.reserve(read_plans.size());
        buffer_offsets.reserve(read_plans.size());

        uint64_t total_payload_size = 0;
        for (const auto& plan : read_plans) {
            CHECK_ARGUMENT(plan.entry.posting_payload_size <=
                               std::numeric_limits<uint64_t>::max() - total_payload_size,
                           "SINDI_V2 query term payload size overflow");
            buffer_offsets.push_back(total_payload_size);
            sizes.push_back(plan.entry.posting_payload_size);
            offsets.push_back(plan.entry.posting_payload_offset);
            total_payload_size += plan.entry.posting_payload_size;
        }

        if (not read_plans.empty()) {
            Vector<uint8_t> payloads(total_payload_size, 0, allocator);
            const bool read_succeeded = payload_layout_.MultiRead(
                offsets.data(), sizes.data(), read_plans.size(), payloads.data());
            if (not read_succeeded) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "failed to batch read SINDI_V2 term payloads");
            }

            for (uint64_t i = 0; i < read_plans.size(); ++i) {
                const auto& plan = read_plans[i];
                auto term_buffer = sindi_datacell_utils::ParseTrustedTermPayload(
                    payloads.data() + buffer_offsets[i],
                    plan.entry.posting_payload_size,
                    plan.entry,
                    window_count,
                    window_size,
                    total_count,
                    value_code_size,
                    allocator);
                query_term_buffers.emplace(plan.term_id, std::move(term_buffer));
            }
        }
        return query_term_buffers;
    }

    for (const auto& plan : read_plans) {
        const auto term_id = plan.term_id;
        const auto& entry = plan.entry;
        bool need_release = false;
        const auto* payload_data = payload_layout_.Read(
            entry.posting_payload_offset, entry.posting_payload_size, need_release);
        CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
            payload_data != nullptr && not need_release,
            "failed to access mmap SINDI_V2 term payload");
        auto term_buffer = sindi_datacell_utils::ViewTrustedTermPayload(payload_data,
                                                                        entry.posting_payload_size,
                                                                        entry,
                                                                        window_count,
                                                                        window_size,
                                                                        total_count,
                                                                        value_code_size,
                                                                        allocator);

        query_term_buffers.emplace(term_id, std::move(term_buffer));
    }

    return query_term_buffers;
}

template <typename IOTmpl>
bool
DiskSindiTermDataCell<IOTmpl>::InsertHeapByWindowKnn(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    bool with_filter,
    const QueryTermBuffers& query_term_buffers,
    const uint64_t* filter_callback_remaining) const {
    if (with_filter) {
        if (filter_callback_remaining != nullptr) {
            return this->template InsertHeapByWindow<InnerSearchMode::KNN_SEARCH,
                                                     InnerSearchType::WITH_FILTER_LIMIT>(
                dists,
                window_id,
                computer,
                heap,
                param,
                offset_id,
                query_term_buffers,
                filter_callback_remaining);
        }
        return this->template InsertHeapByWindow<InnerSearchMode::KNN_SEARCH,
                                                 InnerSearchType::WITH_FILTER>(
            dists, window_id, computer, heap, param, offset_id, query_term_buffers);
    }
    return this->template InsertHeapByWindow<InnerSearchMode::KNN_SEARCH, InnerSearchType::PURE>(
        dists, window_id, computer, heap, param, offset_id, query_term_buffers);
}

template <typename IOTmpl>
bool
DiskSindiTermDataCell<IOTmpl>::InsertHeapByDistsKnn(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    bool with_filter,
    const uint64_t* filter_callback_remaining) const {
    if (with_filter) {
        if (filter_callback_remaining != nullptr) {
            return this->template InsertHeapByDists<InnerSearchMode::KNN_SEARCH,
                                                    InnerSearchType::WITH_FILTER_LIMIT>(
                dists, dists_size, heap, param, offset_id, filter_callback_remaining);
        }
        return this
            ->template InsertHeapByDists<InnerSearchMode::KNN_SEARCH, InnerSearchType::WITH_FILTER>(
                dists, dists_size, heap, param, offset_id);
    }
    return this->template InsertHeapByDists<InnerSearchMode::KNN_SEARCH, InnerSearchType::PURE>(
        dists, dists_size, heap, param, offset_id);
}

template <typename IOTmpl>
bool
DiskSindiTermDataCell<IOTmpl>::InsertHeapByWindow(float* dists,
                                                  uint32_t window_id,
                                                  const SparseTermComputerPtr& computer,
                                                  MaxHeap& heap,
                                                  const InnerSearchParam& param,
                                                  uint32_t offset_id,
                                                  InnerSearchMode mode,
                                                  bool with_filter,
                                                  const SindiQueryContext& query_context,
                                                  const uint64_t* filter_callback_remaining) const {
    const auto& query_term_buffers = query_context.query_term_buffers;
    if (mode == InnerSearchMode::KNN_SEARCH) {
        return this->InsertHeapByWindowKnn(dists,
                                           window_id,
                                           computer,
                                           heap,
                                           param,
                                           offset_id,
                                           with_filter,
                                           query_term_buffers,
                                           filter_callback_remaining);
    }
    if (with_filter) {
        if (filter_callback_remaining != nullptr) {
            return this->template InsertHeapByWindow<InnerSearchMode::RANGE_SEARCH,
                                                     InnerSearchType::WITH_FILTER_LIMIT>(
                dists,
                window_id,
                computer,
                heap,
                param,
                offset_id,
                query_term_buffers,
                filter_callback_remaining);
        }
        return this->template InsertHeapByWindow<InnerSearchMode::RANGE_SEARCH,
                                                 InnerSearchType::WITH_FILTER>(
            dists, window_id, computer, heap, param, offset_id, query_term_buffers);
    }
    return this->template InsertHeapByWindow<InnerSearchMode::RANGE_SEARCH, InnerSearchType::PURE>(
        dists, window_id, computer, heap, param, offset_id, query_term_buffers);
}

template <typename IOTmpl>
bool
DiskSindiTermDataCell<IOTmpl>::InsertHeapByDists(float* dists,
                                                 uint32_t dists_size,
                                                 MaxHeap& heap,
                                                 const InnerSearchParam& param,
                                                 uint32_t offset_id,
                                                 InnerSearchMode mode,
                                                 bool with_filter,
                                                 const uint64_t* filter_callback_remaining) const {
    if (mode == InnerSearchMode::KNN_SEARCH) {
        return this->InsertHeapByDistsKnn(
            dists, dists_size, heap, param, offset_id, with_filter, filter_callback_remaining);
    }
    if (with_filter) {
        if (filter_callback_remaining != nullptr) {
            return this->template InsertHeapByDists<InnerSearchMode::RANGE_SEARCH,
                                                    InnerSearchType::WITH_FILTER_LIMIT>(
                dists, dists_size, heap, param, offset_id, filter_callback_remaining);
        }
        return this->template InsertHeapByDists<InnerSearchMode::RANGE_SEARCH,
                                                InnerSearchType::WITH_FILTER>(
            dists, dists_size, heap, param, offset_id);
    }
    return this->template InsertHeapByDists<InnerSearchMode::RANGE_SEARCH, InnerSearchType::PURE>(
        dists, dists_size, heap, param, offset_id);
}

template <typename IOTmpl>
const TermBuffer*
DiskSindiTermDataCell<IOTmpl>::GetTermBufferNoLock(
    uint32_t term_id, const QueryTermBuffers& query_term_buffers) const {
    auto query_it = query_term_buffers.find(term_id);
    if (query_it != query_term_buffers.end()) {
        return &query_it->second;
    }
    return nullptr;
}

template <typename IOTmpl>
uint64_t
DiskSindiTermDataCell<IOTmpl>::GetMemoryUsage() const {
    uint64_t memory = sizeof(DiskSindiTermDataCell<IOTmpl>);
    std::shared_lock lock(term_layout_mutex_);
    memory += term_dict_.size() * sizeof(DiskTermEntry);
    return memory;
}

template <typename IOTmpl>
void
DiskSindiTermDataCell<IOTmpl>::QueryWindow(float* dists,
                                           uint32_t window_id,
                                           const SparseTermComputerPtr& computer,
                                           bool use_term_lists_heap_insert,
                                           SindiQueryContext& query_context) const {
    (void)use_term_lists_heap_insert;
    const auto& query_term_buffers = query_context.query_term_buffers;
    query_context.evaluation_tracker.BeginWindow(window_size_);
    std::shared_lock lock(term_layout_mutex_);
    while (computer->HasNextTerm()) {
        auto it = computer->NextTermIter();
        auto term = computer->GetTerm(it);
        auto* tb = this->GetTermBufferNoLock(term, query_term_buffers);
        if (tb == nullptr) {
            continue;
        }
        if (window_id >= window_count_) {
            continue;
        }
        auto [start, count] = tb->GetPostingRange(window_id);
        count = computer->GetTermScanCount(count);
        if (count == 0) {
            continue;
        }
        query_context.evaluation_tracker.Mark(tb->IdsData() + start, count);

        if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
            computer->ScanForAccumulateSQ8(
                it, tb->IdsData() + start, tb->ValuesData() + start, count, dists);
        } else if (sparse_value_quant_type_ == SparseValueQuantizationType::FP16) {
            computer->ScanForAccumulateFP16Bytes(
                it,
                tb->IdsData() + start,
                tb->ValuesData() + static_cast<uint64_t>(start) * sizeof(uint16_t),
                count,
                dists);
        } else {
            computer->ScanForAccumulateFloatBytes(
                it,
                tb->IdsData() + start,
                tb->ValuesData() + static_cast<uint64_t>(start) * sizeof(float),
                count,
                dists);
        }
    }
    computer->ResetTerm();
}

template <typename IOTmpl>
void
DiskSindiTermDataCell<IOTmpl>::GetSparseVector(uint32_t inner_id,
                                               SparseVector* data,
                                               Allocator* specified_allocator) const {
    Allocator* allocator = specified_allocator != nullptr ? specified_allocator : allocator_;
    Vector<uint32_t> term_ids(allocator);
    {
        std::shared_lock lock(term_layout_mutex_);
        term_ids.reserve(term_dict_.size());
        for (uint32_t term = 0; term < term_dict_.size(); ++term) {
            if (term_dict_[term].posting_count != 0) {
                term_ids.push_back(term);
            }
        }
    }
    auto query_term_buffers = this->LoadQueryTermBuffers(term_ids, allocator);
    std::shared_lock lock(term_layout_mutex_);
    Vector<uint32_t> ids(allocator);
    Vector<float> vals(allocator);

    for (uint32_t term_id : term_ids) {
        const auto buffer_iter = query_term_buffers.find(term_id);
        if (buffer_iter == query_term_buffers.end()) {
            continue;
        }
        const auto& tb = buffer_iter->second;
        uint32_t window_id = inner_id / window_size_;
        uint32_t local_id = inner_id % window_size_;
        if (window_id >= window_count_) {
            continue;
        }
        const auto [start, count] = tb.GetPostingRange(window_id);
        const auto end = start + count;
        for (uint32_t i = start; i < end; i++) {
            if (tb.IdsData()[i] == local_id) {
                ids.push_back(term_id);
                vals.push_back(sindi_datacell_utils::DecodeValue(
                    tb.ValuesData() +
                        static_cast<uint64_t>(i) *
                            sindi_datacell_utils::GetValueCodeSize(sparse_value_quant_type_),
                    sparse_value_quant_type_,
                    quantization_params_.get()));
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
DiskSindiTermDataCell<IOTmpl>::CalcDistanceByInnerId(
    const SparseTermComputerPtr& computer,
    uint32_t base_id,
    const QueryTermBuffers& query_term_buffers) const {
    std::shared_lock lock(term_layout_mutex_);
    float ip = 0;
    while (computer->HasNextTerm()) {
        auto it = computer->NextTermIter();
        auto term = computer->GetTerm(it);
        auto* tb = this->GetTermBufferNoLock(term, query_term_buffers);
        if (tb == nullptr) {
            continue;
        }
        uint32_t window_id = base_id / window_size_;
        uint32_t local_id = static_cast<uint16_t>(base_id % window_size_);
        if (window_id >= window_count_) {
            continue;
        }
        const auto [start, count] = tb->GetPostingRange(window_id);
        const auto end = start + count;
        for (uint32_t i = start; i < end; i++) {
            if (tb->IdsData()[i] == local_id) {
                const auto val = sindi_datacell_utils::DecodeValue(
                    tb->ValuesData() +
                        static_cast<uint64_t>(i) *
                            sindi_datacell_utils::GetValueCodeSize(sparse_value_quant_type_),
                    sparse_value_quant_type_,
                    quantization_params_.get());
                ip += computer->sorted_query_[it].second * val;
                break;
            }
        }
    }
    computer->ResetTerm();
    return 1.0F + ip;
}

// Explicit instantiations
template class DiskSindiTermDataCell<MMapIO>;
template class DiskSindiTermDataCell<BufferIO>;
template class DiskSindiTermDataCell<ReaderIO>;
#if HAVE_LIBAIO
template class DiskSindiTermDataCell<AsyncIO>;
#endif

template bool
DiskSindiTermDataCell<MMapIO>::InsertHeapByWindow<InnerSearchMode::KNN_SEARCH,
                                                  InnerSearchType::PURE>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<MMapIO>::InsertHeapByWindow<InnerSearchMode::KNN_SEARCH,
                                                  InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<MMapIO>::InsertHeapByWindow<InnerSearchMode::RANGE_SEARCH,
                                                  InnerSearchType::PURE>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<MMapIO>::InsertHeapByWindow<InnerSearchMode::RANGE_SEARCH,
                                                  InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<MMapIO>::InsertHeapByDists<InnerSearchMode::KNN_SEARCH,
                                                 InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<MMapIO>::InsertHeapByDists<InnerSearchMode::KNN_SEARCH,
                                                 InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<MMapIO>::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH,
                                                 InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<MMapIO>::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH,
                                                 InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<BufferIO>::InsertHeapByWindow<InnerSearchMode::KNN_SEARCH,
                                                    InnerSearchType::PURE>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<BufferIO>::InsertHeapByWindow<InnerSearchMode::KNN_SEARCH,
                                                    InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<BufferIO>::InsertHeapByWindow<InnerSearchMode::RANGE_SEARCH,
                                                    InnerSearchType::PURE>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<BufferIO>::InsertHeapByWindow<InnerSearchMode::RANGE_SEARCH,
                                                    InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<BufferIO>::InsertHeapByDists<InnerSearchMode::KNN_SEARCH,
                                                   InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<BufferIO>::InsertHeapByDists<InnerSearchMode::KNN_SEARCH,
                                                   InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<BufferIO>::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH,
                                                   InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<BufferIO>::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH,
                                                   InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<ReaderIO>::InsertHeapByWindow<InnerSearchMode::KNN_SEARCH,
                                                    InnerSearchType::PURE>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<ReaderIO>::InsertHeapByWindow<InnerSearchMode::KNN_SEARCH,
                                                    InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<ReaderIO>::InsertHeapByWindow<InnerSearchMode::RANGE_SEARCH,
                                                    InnerSearchType::PURE>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<ReaderIO>::InsertHeapByWindow<InnerSearchMode::RANGE_SEARCH,
                                                    InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t window_id,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const QueryTermBuffers& query_term_buffers,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<ReaderIO>::InsertHeapByDists<InnerSearchMode::KNN_SEARCH,
                                                   InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<ReaderIO>::InsertHeapByDists<InnerSearchMode::KNN_SEARCH,
                                                   InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<ReaderIO>::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH,
                                                   InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_remaining) const;

template bool
DiskSindiTermDataCell<ReaderIO>::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH,
                                                   InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id,
    const uint64_t* filter_callback_remaining) const;

}  // namespace vsag
