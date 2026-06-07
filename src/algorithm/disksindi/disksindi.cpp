
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

#include "disksindi.h"

#include <algorithm>
#include <cstring>
#include <istream>
#include <mutex>
#include <sstream>
#include <vector>

#include "algorithm/sparse_distance.h"
#include "datacell/sparse_vector_datacell_parameter.h"
#include "impl/filter/inner_id_wrapper_filter.h"
#include "impl/filter/white_list_filter.h"
#include "impl/heap/standard_heap.h"
#include "index_feature_list.h"
#include "inner_string_params.h"
#include "io/reader_io_parameter.h"
#include "quantization/sparse_quantization/sparse_quantizer.h"
#include "quantization/sparse_quantization/sparse_quantizer_parameter.h"
#include "storage/empty_index_binary_set.h"
#include "storage/serialization.h"
#include "utils/util_functions.h"
#include "vsag/allocator.h"
#include "vsag/options.h"
#include "vsag_exception.h"

namespace vsag {

namespace {

constexpr const char* DISKSINDI_RERANK_FLAT_FORMAT_KEY = "disksindi_rerank_flat_format";
constexpr int64_t DISKSINDI_RERANK_FLAT_FORMAT_DATACELL = 2;

class BinaryReader : public Reader {
public:
    explicit BinaryReader(Binary binary) : binary_(std::move(binary)) {
    }

    void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        std::memcpy(dest, binary_.data.get() + offset, len);
    }

    void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, CallBack callback) override {
        Read(offset, len, dest);
        callback(IOErrorCode::IO_SUCCESS, "success");
    }

    uint64_t
    Size() const override {
        return binary_.size;
    }

private:
    Binary binary_;
};

class StreamBackedReader : public Reader {
public:
    explicit StreamBackedReader(std::istream& stream) : stream_(stream) {
        auto cursor = stream_.tellg();
        stream_.seekg(0, std::ios::end);
        size_ = static_cast<uint64_t>(stream_.tellg());
        stream_.seekg(cursor, std::ios::beg);
    }

    void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        std::lock_guard<std::mutex> lock(mutex_);
        stream_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        stream_.read(static_cast<char*>(dest), static_cast<std::streamsize>(len));
    }

    void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, CallBack callback) override {
        Read(offset, len, dest);
        callback(IOErrorCode::IO_SUCCESS, "success");
    }

    uint64_t
    Size() const override {
        return size_;
    }

private:
    std::istream& stream_;
    uint64_t size_{0};
    std::mutex mutex_;
};

float
compute_distance_from_codes(const uint8_t* codes,
                            const Vector<uint32_t>& sorted_ids,
                            const Vector<float>& sorted_vals) {
    auto len = *reinterpret_cast<const uint32_t*>(codes);
    const auto* entries = reinterpret_cast<const BufferEntry*>(codes + sizeof(uint32_t));
    float sum = 0.0F;
    uint32_t i = 0;
    uint32_t j = 0;
    while (i < sorted_ids.size() && j < len) {
        if (sorted_ids[i] < entries[j].id) {
            i++;
        } else if (sorted_ids[i] > entries[j].id) {
            j++;
        } else {
            sum += sorted_vals[i] * entries[j].val;
            i++;
            j++;
        }
    }
    return 1 - sum;
}

float
cal_distance_by_id_unsafe(const FlattenInterfacePtr& flat,
                          const Vector<uint32_t>& sorted_ids,
                          const Vector<float>& sorted_vals,
                          uint32_t inner_id) {
    bool need_release{false};
    const auto* codes = flat->GetCodesById(inner_id, need_release);
    float distance = compute_distance_from_codes(codes, sorted_ids, sorted_vals);
    if (need_release) {
        flat->Release(codes);
    }
    return distance;
}

DatasetPtr
collect_results(const DistHeapPtr& results, Allocator* allocator) {
    auto [result, dists, ids] =
        create_fast_dataset(static_cast<int64_t>(results->Size()), allocator);
    if (results->Empty()) {
        result->Dim(0)->NumElements(1);
        return result;
    }

    for (auto j = static_cast<int64_t>(results->Size() - 1); j >= 0; --j) {
        dists[j] = results->Top().first;
        ids[j] = results->Top().second;
        results->Pop();
    }
    return result;
}

Vector<uint32_t>
collect_query_term_ids(const SparseTermComputerPtr& computer, Allocator* allocator) {
    Vector<uint32_t> query_term_ids(allocator);
    while (computer->HasNextTerm()) {
        auto it = computer->NextTermIter();
        query_term_ids.push_back(computer->GetTerm(it));
    }
    computer->ResetTerm();
    return query_term_ids;
}

uint64_t
block_memory_ceil(uint64_t memory) {
    const auto block_size = Options::Instance().block_size_limit();
    return ((memory + block_size - 1) / block_size) * block_size;
}

FlattenInterfacePtr
create_rerank_flat(const IndexCommonParam& common_param, const IOParamPtr& io_param) {
    auto rerank_param = std::make_shared<SparseVectorDataCellParameter>();
    rerank_param->io_parameter = io_param;
    rerank_param->quantizer_parameter = std::make_shared<SparseQuantizerParameter>();
    return FlattenInterface::MakeInstance(rerank_param, common_param);
}

void
deserialize_legacy_rerank_flat(StreamReader& reader,
                               const FlattenInterfacePtr& flat,
                               Allocator* allocator) {
    int64_t cur_element_count = 0;
    StreamReader::ReadObj(reader, cur_element_count);
    flat->Resize(cur_element_count);
    std::vector<uint32_t> ids;
    std::vector<float> vals;
    for (int64_t i = 0; i < cur_element_count; ++i) {
        uint32_t len = 0;
        StreamReader::ReadObj(reader, len);
        ids.resize(len);
        vals.resize(len);
        reader.Read(reinterpret_cast<char*>(ids.data()),
                    static_cast<uint64_t>(len) * sizeof(uint32_t));
        reader.Read(reinterpret_cast<char*>(vals.data()),
                    static_cast<uint64_t>(len) * sizeof(float));
        SparseVector vector;
        vector.len_ = len;
        vector.ids_ = ids.data();
        vector.vals_ = vals.data();
        flat->InsertVector(&vector, i);
    }
    LabelTable legacy_label_table(allocator);
    legacy_label_table.Deserialize(reader);
}

void
deserialize_rerank_flat(StreamReader& reader,
                        const FlattenInterfacePtr& flat,
                        Allocator* allocator,
                        bool has_datacell_format) {
    if (has_datacell_format) {
        flat->Deserialize(reader);
        return;
    }
    deserialize_legacy_rerank_flat(reader, flat, allocator);
}

}  // namespace

ParamPtr
DiskSINDI::CheckAndMappingExternalParam(const JsonType& external_param,
                                        const IndexCommonParam& common_param) {
    auto ptr = std::make_shared<DiskSINDIParameter>();
    ptr->FromJson(external_param);
    return ptr;
}

DiskSINDI::DiskSINDI(const DiskSINDIParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param),
      use_reorder_(param->use_reorder),
      use_quantization_(param->use_quantization),
      term_id_limit_(param->term_id_limit),
      window_size_(param->window_size),
      doc_retain_ratio_(1.0F - param->doc_prune_ratio),
      deserialize_without_footer_(param->deserialize_without_footer),
      deserialize_without_buffer_(param->deserialize_without_buffer),
      quantization_params_(std::make_shared<QuantizationParams>()),
      avg_doc_term_length_(param->avg_doc_term_length),
      remap_term_ids_(param->remap_term_ids),
      param_(param) {
    CHECK_ARGUMENT(window_size_ > 0 && window_size_ <= 65536, "window_size must be in (0, 65536]");
    CHECK_ARGUMENT(term_id_limit_ > 0, "term_id_limit must be > 0");
    if (remap_term_ids_) {
        term_id_mapper_ =
            std::make_shared<TermIdMapper>(term_id_limit_, common_param.allocator_.get());
    }
    if (use_reorder_) {
        rerank_flat_ = create_rerank_flat(common_param, param->rerank_io_parameter);
    }
    term_datacell_ = DiskSparseTermListDataCellInterface::MakeInstance(doc_retain_ratio_,
                                                                       term_id_limit_,
                                                                       allocator_,
                                                                       use_quantization_,
                                                                       quantization_params_,
                                                                       window_size_,
                                                                       param->term_io_parameter,
                                                                       common_param);
}

std::string
DiskSINDI::GetStats() const {
    return "";
}

std::vector<int64_t>
DiskSINDI::Add(const DatasetPtr& base, AddMode mode) {
    std::scoped_lock wlock(this->global_mutex_);

    if (is_deserialized_) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "DiskSINDI does not support Add after Deserialize");
    }

    std::vector<int64_t> failed_ids;

    auto data_num = base->GetNumElements();
    CHECK_ARGUMENT(data_num > 0, "data_num is zero when add vectors");

    const auto* sparse_vectors = base->GetSparseVectors();
    const auto* ids = base->GetIds();
    const auto* extra_info = base->GetExtraInfos();
    const auto extra_info_size = base->GetExtraInfoSize();

    if (use_quantization_ && cur_element_count_ == 0) {
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        for (int64_t i = 0; i < data_num; ++i) {
            const auto& vec = sparse_vectors[i];
            for (int j = 0; j < vec.len_; ++j) {
                float val = vec.vals_[j];
                if (val < min_val) {
                    min_val = val;
                }
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
        quantization_params_->min_val = min_val;
        quantization_params_->max_val = max_val;
        quantization_params_->diff = max_val - min_val;
        if (quantization_params_->diff < 1e-6) {
            quantization_params_->diff = 1.0F;
        }
    }

    Vector<uint32_t> tmp_ids(allocator_);
    for (uint32_t i = 0; i < data_num; ++i) {
        const auto& sparse_vector = sparse_vectors[i];
        if (label_table_->CheckLabel(ids[i])) {
            failed_ids.push_back(ids[i]);
            logger::warn("id ({}) already exists", ids[i]);
            continue;
        }
        if (sparse_vector.len_ <= 0) {
            failed_ids.push_back(ids[i]);
            logger::warn(
                "sparse_vector.len_ ({}) is invalid for id ({})", sparse_vector.len_, ids[i]);
            continue;
        }

        try {
            if (remap_term_ids_) {
                auto remapped = remap_sparse_vector_for_build(sparse_vector, tmp_ids);
                term_datacell_->InsertVector(remapped, static_cast<uint32_t>(cur_element_count_));
            } else {
                term_datacell_->InsertVector(sparse_vector,
                                             static_cast<uint32_t>(cur_element_count_));
            }
        } catch (const std::runtime_error& e) {
            failed_ids.push_back(ids[i]);
            logger::warn("runtime error: {}", e.what());
            continue;
        } catch (const VsagException& e) {
            failed_ids.push_back(ids[i]);
            logger::warn("vsag exception: {}", e.what());
            continue;
        } catch (const std::bad_alloc& e) {
            failed_ids.push_back(ids[i]);
            logger::warn("memory allocation failed: {}", e.what());
            continue;
        }

        label_table_->Insert(cur_element_count_, ids[i]);

        if (extra_info_size > 0) {
            extra_infos_->InsertExtraInfo(extra_info + i * extra_info_size, cur_element_count_);
        }

        if (use_reorder_) {
            rerank_flat_->InsertVector(sparse_vectors + i, cur_element_count_);
        }

        cur_element_count_++;
    }

    this->cal_memory_usage();
    return failed_ids;
}

std::vector<int64_t>
DiskSINDI::Build(const DatasetPtr& base) {
    auto failed_ids = this->Add(base);

    uint32_t window_count =
        static_cast<uint32_t>(align_up(cur_element_count_, window_size_) / window_size_);
    term_datacell_->FinalizeTermBuffers(window_count);
    this->cal_memory_usage();
    return failed_ids;
}

bool
DiskSINDI::UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update) {
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        "DiskSINDI does not support UpdateVector");
}

DatasetPtr
DiskSINDI::KnnSearch(const DatasetPtr& query,
                     int64_t k,
                     const std::string& parameters,
                     const FilterPtr& filter) const {
    return KnnSearch(query, k, parameters, filter, allocator_);
}

DatasetPtr
DiskSINDI::KnnSearch(const DatasetPtr& query,
                     int64_t k,
                     const std::string& parameters,
                     const FilterPtr& filter,
                     vsag::Allocator* allocator) const {
    std::shared_lock rlock(this->global_mutex_);

    const auto* sparse_vectors = query->GetSparseVectors();
    CHECK_ARGUMENT(query->GetNumElements() == 1, "num of query should be 1");
    CHECK_ARGUMENT(k > 0, "k must be greater than 0");
    auto sparse_query = sparse_vectors[0];
    CHECK_ARGUMENT(
        sparse_query.len_ > 0,
        fmt::format("query->GetSparseVectors()->len_ ({}) is invalid", sparse_query.len_));

    DiskSINDISearchParameter search_param;
    search_param.FromJson(JsonType::Parse(parameters));
    CHECK_ARGUMENT(search_param.n_candidate <= SPARSE_AMPLIFICATION_FACTOR * k,
                   fmt::format("n_candidate ({}) should be less than {} * k ({})",
                               search_param.n_candidate,
                               SPARSE_AMPLIFICATION_FACTOR,
                               k));
    InnerSearchParam inner_param;
    inner_param.ef = std::max(static_cast<int64_t>(search_param.n_candidate), k);
    inner_param.topk = k;

    FilterPtr ft = nullptr;
    if (filter != nullptr) {
        ft = std::make_shared<InnerIdWrapperFilter>(filter, *this->label_table_);
    }
    inner_param.is_inner_id_allowed = ft;

    SparseVector effective_query = sparse_query;
    Vector<uint32_t> tmp_ids(allocator_);
    Vector<float> tmp_vals(allocator_);
    if (remap_term_ids_) {
        effective_query = remap_sparse_vector_for_query(sparse_query, tmp_ids, tmp_vals);
        if (effective_query.len_ == 0) {
            auto [results, ret_dists, ret_ids] = create_fast_dataset(0, allocator);
            return results;
        }
    }

    auto computer = std::make_shared<SparseTermComputer>(effective_query, search_param, allocator);
    const SparseVector* rerank_query = (remap_term_ids_ && use_reorder_) ? &sparse_query : nullptr;

    // Collect query term ids for lazy loading
    auto query_term_ids = collect_query_term_ids(computer, allocator);
    auto query_term_buffers = term_datacell_->LoadQueryTermBuffers(query_term_ids);

    return search_impl<KNN_SEARCH>(computer,
                                   inner_param,
                                   allocator,
                                   search_param.use_term_lists_heap_insert,
                                   query_term_buffers,
                                   rerank_query);
}

template <InnerSearchMode mode>
DatasetPtr
DiskSINDI::search_impl(const SparseTermComputerPtr& computer,
                       const InnerSearchParam& inner_param,
                       Allocator* allocator,
                       bool use_term_lists_heap_insert,
                       const QueryTermBuffers& query_term_buffers,
                       const SparseVector* original_query) const {
    MaxHeap heap(allocator);
    int64_t k = 0;

    if constexpr (mode == KNN_SEARCH) {
        k = inner_param.topk;
    }

    Vector<float> dists(window_size_, 0.0, allocator);
    auto filter = inner_param.is_inner_id_allowed;
    const auto [min_window_id, max_window_id] = this->get_min_max_window_id(filter);

    for (auto cur = min_window_id; cur <= max_window_id; cur++) {
        auto window_start_id = static_cast<uint32_t>(cur) * window_size_;

        term_datacell_->QueryWindow(
            dists.data(), static_cast<uint32_t>(cur), computer, query_term_buffers);

        if (use_term_lists_heap_insert) {
            term_datacell_->InsertHeapByWindowKnn(dists.data(),
                                                  static_cast<uint32_t>(cur),
                                                  computer,
                                                  heap,
                                                  inner_param,
                                                  window_start_id,
                                                  inner_param.is_inner_id_allowed != nullptr,
                                                  query_term_buffers);
        } else {
            term_datacell_->InsertHeapByDistsKnn(dists.data(),
                                                 dists.size(),
                                                 heap,
                                                 inner_param,
                                                 window_start_id,
                                                 inner_param.is_inner_id_allowed != nullptr);
        }
    }

    // rerank
    if (use_reorder_) {
        float cur_heap_top = std::numeric_limits<float>::max();
        auto candidate_size = heap.size();
        auto high_precise_heap = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
        auto [sorted_ids, sorted_vals] =
            sort_sparse_vector(original_query ? *original_query : computer->raw_query_, allocator_);

        // Phase B: collect all candidate inner ids, sort them by inner_id
        // (ascending), then issue a single batched IO. Sorting makes disk
        // offsets monotonically increasing, which enables
        // GetCodesByIdsBatch to merge adjacent IO requests and reduces
        // syscall count for DirectIO.
        Vector<InnerIdType> cand_ids(allocator_);
        cand_ids.resize(candidate_size);
        for (int64_t i = static_cast<int64_t>(candidate_size) - 1; i >= 0; --i) {
            cand_ids[i] = heap.top().second;
            heap.pop();
        }
        std::sort(cand_ids.begin(), cand_ids.end());
        auto batch = rerank_flat_->GetCodesByIdsBatch(
            cand_ids.data(), static_cast<InnerIdType>(candidate_size), allocator_);

        for (uint64_t i = 0; i < candidate_size; i++) {
            auto inner_id = cand_ids[i];
            const uint8_t* codes = batch.buffer.data() + batch.in_buffer_offsets[i];
            auto high_precise_distance =
                compute_distance_from_codes(codes, sorted_ids, sorted_vals);
            auto label = label_table_->GetLabelById(inner_id);
            if constexpr (mode == KNN_SEARCH) {
                if (high_precise_distance < cur_heap_top or
                    high_precise_heap->Size() < static_cast<uint64_t>(k)) {
                    high_precise_heap->Push(high_precise_distance, label);
                }
                if (high_precise_heap->Size() > static_cast<uint64_t>(k)) {
                    high_precise_heap->Pop();
                }
                cur_heap_top = high_precise_heap->Top().first;
            }
            if constexpr (mode == RANGE_SEARCH) {
                if (high_precise_distance <= inner_param.radius) {
                    high_precise_heap->Push(high_precise_distance, label);
                }
                if (inner_param.range_search_limit_size != -1 and
                    high_precise_heap->Size() >
                        static_cast<uint64_t>(inner_param.range_search_limit_size)) {
                    high_precise_heap->Pop();
                }
            }
        }

        return collect_results(high_precise_heap, allocator_);
    }

    // low precision
    if constexpr (mode == RANGE_SEARCH) {
        k = static_cast<int64_t>(heap.size());
        if (inner_param.range_search_limit_size != -1) {
            k = inner_param.range_search_limit_size;
        }
    }

    int64_t cur_size = std::min(static_cast<int64_t>(heap.size()), k);

    auto [results, ret_dists, ret_ids] = create_fast_dataset(cur_size, allocator_);
    if (cur_size == 0) {
        return results;
    }

    while (heap.size() > k) {
        heap.pop();
    }

    for (auto j = cur_size - 1; j >= 0; j--) {
        ret_dists[j] = 1 + heap.top().first;
        ret_ids[j] = label_table_->GetLabelById(heap.top().second);
        heap.pop();
    }

    return results;
}

DatasetPtr
DiskSINDI::RangeSearch(const DatasetPtr& query,
                       float radius,
                       const std::string& parameters,
                       const FilterPtr& filter,
                       int64_t limited_size) const {
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        "DiskSINDI does not support RangeSearch in stage 2");
}

void
DiskSINDI::cal_memory_usage() {
    auto memory = sizeof(DiskSINDI);
    if (term_datacell_ != nullptr) {
        memory += term_datacell_->GetMemoryUsage();
    }
    if (this->rerank_flat_ != nullptr) {
        memory += this->rerank_flat_->GetMemoryUsage();
    }
    memory += sizeof(QuantizationParams);

    std::unique_lock lock(this->memory_usage_mutex_);
    this->current_memory_usage_.store(static_cast<int64_t>(memory));
}

void
DiskSINDI::Serialize(StreamWriter& writer) const {
    std::shared_lock rlock(this->global_mutex_);

    // Pre-compute sizes for two-pass serialization (StreamWriter lacks Seek/WriteAt)
    uint64_t metadata_start = writer.GetCursor();

    uint64_t quantization_size = use_quantization_ ? 3 * sizeof(float) : 0;
    uint64_t term_dict_size = (term_id_limit_ + 1) * sizeof(DiskTermEntry);
    uint64_t payload_size = term_datacell_->ComputePayloadSize();

    // Compute segment offsets
    DiskSINDIManifest manifest{};
    manifest.term_dict_offset =
        metadata_start + sizeof(manifest) + sizeof(cur_element_count_) + quantization_size;
    manifest.term_dict_size = term_dict_size;
    manifest.posting_payload_offset = manifest.term_dict_offset + term_dict_size;
    manifest.posting_payload_size = payload_size;

    uint64_t cursor_after_payload = manifest.posting_payload_offset + payload_size;

    if (use_reorder_) {
        manifest.rerank_flat_offset = cursor_after_payload;
        manifest.rerank_flat_size = rerank_flat_->CalcSerializeSize();
        cursor_after_payload += manifest.rerank_flat_size;
    }

    manifest.label_table_offset = cursor_after_payload;
    std::stringstream label_ss;
    {
        IOStreamWriter label_writer(label_ss);
        label_table_->Serialize(label_writer);
    }
    manifest.label_table_size = label_ss.tellp();

    // Pass 2: write everything in order
    writer.Write(reinterpret_cast<const char*>(&manifest), sizeof(manifest));
    StreamWriter::WriteObj(writer, cur_element_count_);
    if (use_quantization_) {
        StreamWriter::WriteObj(writer, quantization_params_->min_val);
        StreamWriter::WriteObj(writer, quantization_params_->max_val);
        StreamWriter::WriteObj(writer, quantization_params_->diff);
    }

    term_datacell_->WriteTermDictAndPayload(writer, manifest.posting_payload_offset);

    if (use_reorder_) {
        rerank_flat_->Serialize(writer);
    }

    writer.Write(label_ss.str().c_str(), manifest.label_table_size);

    if (remap_term_ids_ && term_id_mapper_) {
        term_id_mapper_->Serialize(writer);
    }

    // Footer
    JsonType jsonify_basic_info;
    auto metadata = std::make_shared<Metadata>();
    jsonify_basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());
    if (use_reorder_) {
        jsonify_basic_info[DISKSINDI_RERANK_FLAT_FORMAT_KEY].SetInt(
            DISKSINDI_RERANK_FLAT_FORMAT_DATACELL);
    }
    metadata->Set("basic_info", jsonify_basic_info);
    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}

void
DiskSINDI::SetIO(const std::shared_ptr<Reader> reader) {
    auto reader_param = std::make_shared<ReaderIOParameter>();
    reader_param->reader = reader;
    if (term_datacell_ != nullptr) {
        term_datacell_->SetIO(
            reader, manifest_.posting_payload_offset, manifest_.posting_payload_size);
    }
    if (rerank_flat_ != nullptr && param_->rerank_io_parameter != nullptr &&
        param_->rerank_io_parameter->GetTypeName() == IO_TYPE_VALUE_READER_IO) {
        rerank_flat_->InitIO(reader_param);
    }
}

void
DiskSINDI::Deserialize(const BinarySet& binary_set) {
    if (binary_set.Contains(SERIAL_META_KEY)) {
        auto metadata = std::make_shared<Metadata>(binary_set.Get(SERIAL_META_KEY));
        if (metadata->EmptyIndex()) {
            return;
        }
    } else if (binary_set.Contains(BLANK_INDEX)) {
        return;
    }

    auto binary = binary_set.Get(this->GetName());
    auto reader_holder = std::make_shared<BinaryReader>(binary);
    auto func = [reader_holder](uint64_t offset, uint64_t len, void* dest) {
        reader_holder->Read(offset, len, dest);
    };
    uint64_t cursor = 0;
    auto reader = ReadFuncStreamReader(func, cursor, reader_holder->Size());
    this->Deserialize(reader);
    this->SetIO(reader_holder);
}

void
DiskSINDI::Deserialize(std::istream& in_stream) {
    auto reader_holder = std::make_shared<StreamBackedReader>(in_stream);
    auto func = [reader_holder](uint64_t offset, uint64_t len, void* dest) {
        reader_holder->Read(offset, len, dest);
    };
    uint64_t cursor = 0;
    auto reader = ReadFuncStreamReader(func, cursor, reader_holder->Size());

    auto footer = Footer::Parse(reader);
    if (footer != nullptr) {
        auto metadata = footer->GetMetadata();
        if (metadata->EmptyIndex()) {
            return;
        }
    }
    this->Deserialize(reader);
    this->SetIO(reader_holder);
}

void
DiskSINDI::Deserialize(StreamReader& reader) {
    std::scoped_lock wlock(this->global_mutex_);

    bool has_datacell_rerank_format = false;
    auto footer = Footer::Parse(reader);
    if (footer != nullptr) {
        auto metadata = footer->GetMetadata();
        JsonType jsonify_basic_info = metadata->Get("basic_info");
        {
            auto param = jsonify_basic_info[INDEX_PARAM].GetString();
            DiskSINDIParameterPtr index_param = std::make_shared<DiskSINDIParameter>();
            index_param->FromString(param);
            if (not this->create_param_ptr_->CheckCompatibility(index_param)) {
                auto message =
                    fmt::format("DiskSINDI index parameter not match, current: {}, new: {}",
                                this->create_param_ptr_->ToString(),
                                index_param->ToString());
                logger::error(message);
                throw VsagException(ErrorType::INVALID_ARGUMENT, message);
            }
        }
        if (jsonify_basic_info.Contains(DISKSINDI_RERANK_FLAT_FORMAT_KEY)) {
            has_datacell_rerank_format =
                jsonify_basic_info[DISKSINDI_RERANK_FLAT_FORMAT_KEY].GetInt() ==
                DISKSINDI_RERANK_FLAT_FORMAT_DATACELL;
        }
    } else if (not deserialize_without_footer_) {
        logger::debug("DiskSINDI footer not found, fallback to legacy deserialize path");
    }

    // Read manifest from the beginning
    reader.Seek(0);
    StreamReader::ReadObj(reader, manifest_);
    StreamReader::ReadObj(reader, cur_element_count_);

    if (use_quantization_) {
        StreamReader::ReadObj(reader, quantization_params_->min_val);
        StreamReader::ReadObj(reader, quantization_params_->max_val);
        StreamReader::ReadObj(reader, quantization_params_->diff);
    }

    uint32_t window_count =
        static_cast<uint32_t>(align_up(cur_element_count_, window_size_) / window_size_);
    term_datacell_->Deserialize(reader, manifest_.term_dict_size, window_count);
    if (param_->term_io_parameter != nullptr &&
        param_->term_io_parameter->GetTypeName() != IO_TYPE_VALUE_READER_IO) {
        term_datacell_->InitIO(param_->term_io_parameter);
        if (manifest_.posting_payload_offset != 0) {
            term_datacell_->WritePayloadToIO(
                reader, manifest_.posting_payload_offset, manifest_.posting_payload_size);
        }
    }

    if (use_reorder_) {
        reader.Seek(manifest_.rerank_flat_offset);
        deserialize_rerank_flat(reader, rerank_flat_, allocator_, has_datacell_rerank_format);
    }

    reader.Seek(manifest_.label_table_offset);
    label_table_->Deserialize(reader);

    if (remap_term_ids_ && term_id_mapper_) {
        term_id_mapper_->Deserialize(reader);
    }

    is_deserialized_ = true;
    this->cal_memory_usage();
}

std::pair<int64_t, int64_t>
DiskSINDI::GetMinAndMaxId() const {
    int64_t min_id = INT64_MAX;
    int64_t max_id = INT64_MIN;
    std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
    if (this->cur_element_count_ == 0) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Label map size is zero");
    }
    for (int i = 0; i < this->cur_element_count_; ++i) {
        if (this->label_table_->IsRemoved(i)) {
            continue;
        }
        auto label = this->label_table_->GetLabelById(i);
        max_id = std::max(label, max_id);
        min_id = std::min(label, min_id);
    }
    return {min_id, max_id};
}

uint64_t
DiskSINDI::EstimateMemory(uint64_t num_elements) const {
    uint64_t mem = 0;
    mem += sizeof(DiskSINDI);
    mem += (term_id_limit_ + 1) * sizeof(DiskTermEntry);
    mem += 2 * sizeof(int64_t) * num_elements;
    if (rerank_flat_ != nullptr) {
        const auto rerank_payload_bytes =
            num_elements *
            (sizeof(uint32_t) + avg_doc_term_length_ * (sizeof(uint32_t) + sizeof(float)));
        const auto rerank_offset_bytes = num_elements * (sizeof(uint64_t) + sizeof(uint32_t));
        mem += block_memory_ceil(rerank_offset_bytes);
        if (param_->rerank_io_parameter != nullptr &&
            (param_->rerank_io_parameter->GetTypeName() == IO_TYPE_VALUE_MEMORY_IO ||
             param_->rerank_io_parameter->GetTypeName() == IO_TYPE_VALUE_BLOCK_MEMORY_IO)) {
            mem += block_memory_ceil(rerank_payload_bytes);
        }
    }
    mem += sizeof(QuantizationParams);
    return mem;
}

void
DiskSINDI::GetSparseVectorByInnerId(InnerIdType inner_id,
                                    SparseVector* data,
                                    Allocator* specified_allocator) const {
    std::shared_lock rlock(this->global_mutex_);

    if (use_reorder_) {
        return this->rerank_flat_->GetSparseVectorByInnerId(inner_id, data, specified_allocator);
    }

    term_datacell_->GetSparseVector(inner_id, data, specified_allocator);

    if (remap_term_ids_ && term_id_mapper_) {
        for (uint32_t i = 0; i < data->len_; ++i) {
            data->ids_[i] = term_id_mapper_->ReverseMap(data->ids_[i]);
        }
    }
}

float
DiskSINDI::CalcDistanceById(const DatasetPtr& vector,
                            int64_t id,
                            bool calculate_precise_distance) const {
    std::shared_lock rlock(this->global_mutex_);

    if (use_reorder_ && calculate_precise_distance) {
        auto inner_id = this->label_table_->GetIdByLabel(id);
        auto [sorted_ids, sorted_vals] =
            sort_sparse_vector(vector->GetSparseVectors()[0], allocator_);
        return cal_distance_by_id_unsafe(rerank_flat_, sorted_ids, sorted_vals, inner_id);
    }

    auto inner_id = this->label_table_->GetIdByLabel(id);
    auto sparse_query = vector->GetSparseVectors()[0];
    Vector<uint32_t> tmp_ids(allocator_);
    Vector<float> tmp_vals(allocator_);
    if (remap_term_ids_) {
        sparse_query = remap_sparse_vector_for_query(sparse_query, tmp_ids, tmp_vals);
    }
    DiskSINDISearchParameter search_param;
    search_param.query_prune_ratio = 0;
    search_param.term_prune_ratio = 0;
    auto computer = std::make_shared<SparseTermComputer>(sparse_query, search_param, allocator_);
    auto query_term_ids = collect_query_term_ids(computer, allocator_);
    auto query_term_buffers = term_datacell_->LoadQueryTermBuffers(query_term_ids);
    return term_datacell_->CalcDistanceByInnerId(
        computer, static_cast<uint32_t>(inner_id), query_term_buffers);
}

DatasetPtr
DiskSINDI::CalDistanceById(const DatasetPtr& query,
                           const int64_t* ids,
                           int64_t count,
                           bool calculate_precise_distance) const {
    if (use_reorder_ && calculate_precise_distance) {
        std::shared_lock rlock(this->global_mutex_);
        auto result = Dataset::Make();
        result->Owner(true, allocator_);
        auto* distances = static_cast<float*>(allocator_->Allocate(sizeof(float) * count));
        result->Distances(distances);
        auto [sorted_ids, sorted_vals] =
            sort_sparse_vector(query->GetSparseVectors()[0], allocator_);
        for (int64_t i = 0; i < count; i++) {
            auto [success, inner_id] = this->label_table_->TryGetIdByLabel(ids[i]);
            distances[i] =
                success ? cal_distance_by_id_unsafe(rerank_flat_, sorted_ids, sorted_vals, inner_id)
                        : -1;
        }
        return result;
    }

    auto result = Dataset::Make();
    result->Owner(true, allocator_);
    auto* distances = static_cast<float*>(allocator_->Allocate(sizeof(float) * count));
    std::fill_n(distances, count, -1.0F);
    result->Distances(distances);

    std::unordered_map<int64_t, uint32_t> valid_ids;
    for (auto i = 0; i < count; i++) {
        valid_ids[ids[i]] = i;
    }
    auto filter = [&valid_ids](int64_t id) -> bool { return valid_ids.count(id) != 0; };
    auto filter_ptr = std::make_shared<WhiteListFilter>(filter);

    constexpr auto* search_param_fmt = R"(
    {{
        "disksindi": {{
            "query_prune_ratio": 0,
            "n_candidate": {}
        }}
    }}
    )";
    auto search_res =
        this->KnnSearch(query, count, fmt::format(search_param_fmt, count), filter_ptr);

    for (auto i = 0; i < search_res->GetDim(); i++) {
        float dist = search_res->GetDistances()[i];
        int64_t id = search_res->GetIds()[i];
        distances[valid_ids[id]] = dist;
    }

    return result;
}

void
DiskSINDI::SetImmutable() {
    std::scoped_lock wlock(this->global_mutex_);
    this->immutable_ = true;
}

void
DiskSINDI::InitFeatures() {
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_BUILD_WITH_MULTI_THREAD,
        IndexFeature::SUPPORT_KNN_SEARCH,
        IndexFeature::SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
        IndexFeature::SUPPORT_SERIALIZE_FILE,
        IndexFeature::SUPPORT_DESERIALIZE_FILE,
        IndexFeature::SUPPORT_ESTIMATE_MEMORY,
        IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID,
        IndexFeature::SUPPORT_GET_RAW_VECTOR_BY_IDS,
        IndexFeature::SUPPORT_SEARCH_CONCURRENT,
        IndexFeature::SUPPORT_METRIC_TYPE_INNER_PRODUCT,
    });
}

std::pair<int64_t, int64_t>
DiskSINDI::get_min_max_window_id(const FilterPtr& filter) const {
    int64_t min_window_id = 0;
    auto window_count =
        static_cast<uint32_t>(align_up(cur_element_count_, window_size_) / window_size_);
    int64_t max_window_id = static_cast<int64_t>(window_count) - 1;
    if (max_window_id < 0) {
        max_window_id = 0;
    }

    if (filter) {
        const int64_t* valid_ids = nullptr;
        int64_t valid_count = 0;
        filter->GetValidIds(&valid_ids, valid_count);
        int64_t min_inner_id = INT64_MAX;
        int64_t max_inner_id = INT64_MIN;
        int64_t id;
        for (int i = 0; i < valid_count; i++) {
            if (__builtin_expect(static_cast<long>(label_table_->CheckLabel(valid_ids[i])), 1) !=
                0) {
                id = label_table_->GetIdByLabel(valid_ids[i]);
                min_inner_id = std::min(min_inner_id, id);
                max_inner_id = std::max(max_inner_id, id);
            }
        }
        if (min_inner_id != INT64_MAX) {
            min_window_id = min_inner_id / window_size_;
        }
        if (max_inner_id != INT64_MIN) {
            max_window_id = max_inner_id / window_size_;
        }
    }

    return {min_window_id, max_window_id};
}

SparseVector
DiskSINDI::remap_sparse_vector_for_query(const SparseVector& input,
                                         Vector<uint32_t>& tmp_ids,
                                         Vector<float>& tmp_vals) const {
    tmp_ids.clear();
    tmp_vals.clear();
    tmp_ids.reserve(input.len_);
    tmp_vals.reserve(input.len_);
    for (uint32_t i = 0; i < input.len_; ++i) {
        auto compact = term_id_mapper_->TryMap(input.ids_[i]);
        if (compact.has_value()) {
            tmp_ids.push_back(compact.value());
            tmp_vals.push_back(input.vals_[i]);
        }
    }
    SparseVector remapped;
    remapped.len_ = static_cast<uint32_t>(tmp_ids.size());
    remapped.ids_ = tmp_ids.data();
    remapped.vals_ = tmp_vals.data();
    return remapped;
}

SparseVector
DiskSINDI::remap_sparse_vector_for_build(const SparseVector& input, Vector<uint32_t>& tmp_ids) {
    tmp_ids.resize(input.len_);
    for (uint32_t i = 0; i < input.len_; ++i) {
        tmp_ids[i] = term_id_mapper_->Map(input.ids_[i]);
    }
    SparseVector remapped;
    remapped.len_ = input.len_;
    remapped.ids_ = tmp_ids.data();
    remapped.vals_ = input.vals_;
    return remapped;
}

// Explicit template instantiations

}  // namespace vsag
