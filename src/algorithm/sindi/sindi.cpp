
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

#include "sindi.h"

#include <cstring>
#include <vector>

#include "algorithm/sparse_distance.h"
#include "analyzer/analyzer.h"
#include "datacell/sparse_vector_datacell_parameter.h"
#include "impl/heap/standard_heap.h"
#include "index_feature_list.h"
#include "io/memory_block_io/memory_block_io_parameter.h"
#include "quantization/sparse_quantization/sparse_quantizer.h"
#include "quantization/sparse_quantization/sparse_quantizer_parameter.h"
#include "simd/fp16_simd.h"
#include "storage/serialization.h"
#include "storage/serialization_tags.h"
#include "storage/tlv_section.h"
#include "utils/util_functions.h"
#include "vsag/allocator.h"
#include "vsag/options.h"
#include "vsag_exception.h"

namespace vsag {
namespace {

// Approximate per mapped term: one reverse-map uint32_t plus one uint32_t->uint32_t map node.
constexpr uint64_t TERM_ID_MAPPER_ENTRY_MEMORY_BYTES = 54;
constexpr const char* SINDI_RERANK_FLAT_FORMAT_KEY = "sindi_rerank_flat_format";
constexpr int64_t SINDI_RERANK_FLAT_FORMAT_DATACELL = 2;

uint32_t
sparse_value_code_size(SparseValueQuantizationType type) {
    switch (type) {
        case SparseValueQuantizationType::FP32:
            return sizeof(float);
        case SparseValueQuantizationType::SQ8:
            return sizeof(uint8_t);
        case SparseValueQuantizationType::FP16:
            return sizeof(uint16_t);
        default:
            CHECK_ARGUMENT(false, "unknown sparse value quantization type");
    }
    return sizeof(float);
}

float
calc_distance_by_id_unsafe(const FlattenInterfacePtr& flat,
                           const Vector<uint32_t>& sorted_ids,
                           const Vector<float>& sorted_vals,
                           uint32_t inner_id) {
    bool need_release{false};
    const auto* codes = flat->GetCodesById(inner_id, need_release);
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
    auto distance = 1 - sum;
    if (need_release) {
        flat->Release(codes);
    }
    return distance;
}

DatasetPtr
collect_heap_results(const DistHeapPtr& results, Allocator* allocator) {
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

FlattenInterfacePtr
create_rerank_flat(const IndexCommonParam& common_param) {
    auto rerank_param = std::make_shared<SparseVectorDataCellParameter>();
    rerank_param->io_parameter = std::make_shared<MemoryBlockIOParameter>();
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

bool
detect_datacell_rerank_flat(StreamReader& reader, int64_t cur_element_count) {
    if (reader.Length() < reader.GetCursor() + 4 * sizeof(uint32_t)) {
        return false;
    }

    uint32_t total_count = 0;
    uint32_t max_capacity = 0;
    uint32_t code_size = 0;
    uint32_t maybe_sentinel = 0;
    reader.PushSeek(reader.GetCursor());
    StreamReader::ReadObj(reader, total_count);
    StreamReader::ReadObj(reader, max_capacity);
    StreamReader::ReadObj(reader, code_size);
    StreamReader::ReadObj(reader, maybe_sentinel);
    reader.PopSeek();

    (void)max_capacity;
    (void)code_size;
    return total_count == static_cast<uint32_t>(cur_element_count) &&
           maybe_sentinel == std::numeric_limits<uint32_t>::max();
}

}  // namespace

ParamPtr
SINDI::CheckAndMappingExternalParam(const JsonType& external_param,
                                    const IndexCommonParam& common_param) {
    auto ptr = std::make_shared<SINDIParameter>();
    ptr->FromJson(external_param);
    return ptr;
}

SINDI::SINDI(const SINDIParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param),
      use_reorder_(param->use_reorder),
      sparse_value_quant_type_(param->sparse_value_quant_type),
      term_id_limit_(param->term_id_limit),
      window_size_(param->window_size),
      doc_prune_ratio_(param->doc_prune_ratio),
      doc_retain_ratio_(1.0F - param->doc_prune_ratio),
      window_term_list_(common_param.allocator_.get()),
      deserialize_without_footer_(param->deserialize_without_footer),
      deserialize_without_buffer_(param->deserialize_without_buffer),
      quantization_params_(std::make_shared<QuantizationParams>()),
      avg_doc_term_length_(param->avg_doc_term_length),
      remap_term_ids_(param->remap_term_ids),
      immutable_enabled_(param->immutable) {
    if (remap_term_ids_) {
        term_id_mapper_ =
            std::make_shared<TermIdMapper>(term_id_limit_, common_param.allocator_.get());
    }
    if (use_reorder_) {
        rerank_flat_ = create_rerank_flat(common_param);
    }
}

constexpr int64_t K_ANALYZE_DEFAULT_TOPK = 10;
constexpr uint64_t K_ANALYZE_BASE_SAMPLE_SIZE = 10;

std::string
SINDI::GetStats() const {
    AnalyzerParam analyzer_param(allocator_);
    analyzer_param.topk = K_ANALYZE_DEFAULT_TOPK;
    analyzer_param.base_sample_size =
        std::min<uint64_t>(K_ANALYZE_BASE_SAMPLE_SIZE, cur_element_count_);
    analyzer_param.search_params =
        R"({"sindi": {"query_prune_ratio": 0, "term_prune_ratio": 0, "n_candidate": 500}})";
    auto analyzer = CreateAnalyzer(this, analyzer_param);
    JsonType stats = analyzer->GetStats();
    return stats.Dump(4);
}

std::string
SINDI::AnalyzeIndexBySearch(const SearchRequest& request) {
    AnalyzerParam analyzer_param(allocator_);
    analyzer_param.topk = request.topk_;
    analyzer_param.base_sample_size =
        std::min<uint64_t>(K_ANALYZE_BASE_SAMPLE_SIZE, cur_element_count_);
    analyzer_param.search_params = request.params_str_;
    auto analyzer = CreateAnalyzer(this, analyzer_param);
    JsonType stats =
        request.query_ == nullptr ? analyzer->GetStats() : analyzer->AnalyzeIndexBySearch(request);
    return stats.Dump(4);
}

std::vector<int64_t>
SINDI::Add(const DatasetPtr& base) {
    std::scoped_lock wlock(this->global_mutex_);
    const bool mutable_runtime = not immutable_enabled_ and immutable_data_ == nullptr;
    CHECK_ARGUMENT(mutable_runtime, "immutable SINDI runtime does not support Add");
    std::vector<int64_t> failed_ids;

    auto data_num = base->GetNumElements();
    CHECK_ARGUMENT(data_num > 0, "data_num is zero when add vectors");

    const auto* sparse_vectors = base->GetSparseVectors();
    const auto* ids = base->GetIds();
    const auto* extra_info = base->GetExtraInfos();
    const auto extra_info_size = base->GetExtraInfoSize();

    if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8 && cur_element_count_ == 0) {
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

    // adjust window
    int64_t final_add_window = align_up(cur_element_count_ + data_num, window_size_) / window_size_;
    bool window_changed = false;
    while (window_term_list_.size() < final_add_window) {
        window_term_list_.emplace_back(
            std::make_shared<SparseTermDataCell>(doc_retain_ratio_,
                                                 term_id_limit_,
                                                 allocator_,
                                                 sparse_value_quant_type_,
                                                 quantization_params_));
        window_changed = true;
    }

    // add process
    Vector<uint32_t> tmp_ids(allocator_);
    for (uint32_t i = 0; i < data_num; ++i) {
        auto cur_window = cur_element_count_ / window_size_;
        auto window_start_id = cur_window * window_size_;
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

        auto inner_id = static_cast<uint16_t>(cur_element_count_ - window_start_id);

        try {
            if (remap_term_ids_) {
                auto remapped = remap_sparse_vector_for_build(sparse_vector, tmp_ids);
                window_term_list_[cur_window]->InsertVector(remapped, inner_id);
            } else {
                window_term_list_[cur_window]->InsertVector(sparse_vector, inner_id);
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

        label_table_->Insert(cur_element_count_, ids[i]);  // todo(zxy): check id exists

        if (extra_info_size > 0) {
            extra_infos_->InsertExtraInfo(extra_info + i * extra_info_size, cur_element_count_);
        }

        cur_element_count_++;

        // high precision part
        if (use_reorder_) {
            rerank_flat_->InsertVector(sparse_vectors + i, cur_element_count_ - 1);
        }
    }
    if (window_changed) {
        this->cal_memory_usage();
    }
    return failed_ids;
}

std::vector<int64_t>
SINDI::Build(const DatasetPtr& base) {
    if (immutable_enabled_) {
        return this->build_immutable(base);
    }

    // note that there's a wlock in Add()
    auto failed_ids = this->Add(base);
    {
        std::scoped_lock wlock(this->global_mutex_);
        for (auto& window : window_term_list_) {
            window->Compact();
        }
        this->cal_memory_usage();
    }
    return failed_ids;
}

std::vector<int64_t>
SINDI::build_immutable(const DatasetPtr& base) {
    std::scoped_lock wlock(this->global_mutex_);
    CHECK_ARGUMENT(immutable_data_ == nullptr, "immutable SINDI has already been built");
    std::vector<int64_t> failed_ids;

    auto data_num = base->GetNumElements();
    CHECK_ARGUMENT(data_num > 0, "data_num is zero when add vectors");

    const auto* sparse_vectors = base->GetSparseVectors();
    const auto* ids = base->GetIds();
    const auto* extra_info = base->GetExtraInfos();
    const auto extra_info_size = base->GetExtraInfoSize();

    if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8 && cur_element_count_ == 0) {
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        for (int64_t i = 0; i < data_num; ++i) {
            const auto& vec = sparse_vectors[i];
            for (int j = 0; j < vec.len_; ++j) {
                float val = vec.vals_[j];
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
        }
        quantization_params_->min_val = min_val;
        quantization_params_->max_val = max_val;
        quantization_params_->diff = max_val - min_val;
        if (quantization_params_->diff < 1e-6) {
            quantization_params_->diff = 1.0F;
        }
    }

    immutable_data_ = std::make_unique<ImmutableSINDIData>(allocator_);
    immutable_data_->sparse_value_quant_type = sparse_value_quant_type_;
    immutable_data_->value_code_size = sparse_value_code_size(sparse_value_quant_type_);
    immutable_data_->windows.reserve(align_up(data_num, window_size_) / window_size_);

    SparseTermDataCellPtr current_window = nullptr;
    Vector<uint32_t> tmp_ids(allocator_);
    for (int64_t i = 0; i < data_num; ++i) {
        if (current_window == nullptr) {
            current_window = std::make_shared<SparseTermDataCell>(doc_retain_ratio_,
                                                                  term_id_limit_,
                                                                  allocator_,
                                                                  sparse_value_quant_type_,
                                                                  quantization_params_);
        }

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

        const auto window_start_id =
            static_cast<int64_t>(immutable_data_->windows.size()) * window_size_;
        const auto window_inner_id = cur_element_count_ - window_start_id;
        if (window_inner_id < 0 or window_inner_id > std::numeric_limits<uint16_t>::max()) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "immutable SINDI window-local doc id overflows uint16_t");
        }
        auto inner_id = static_cast<uint16_t>(window_inner_id);

        try {
            if (remap_term_ids_) {
                auto remapped = remap_sparse_vector_for_build(sparse_vector, tmp_ids);
                current_window->InsertVector(remapped, inner_id);
            } else {
                current_window->InsertVector(sparse_vector, inner_id);
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

        cur_element_count_++;

        if (use_reorder_) {
            rerank_flat_->InsertVector(sparse_vectors + i, cur_element_count_ - 1);
        }

        if (cur_element_count_ % window_size_ == 0) {
            immutable_data_->windows.emplace_back(allocator_);
            compact_window_to_immutable(*current_window, immutable_data_->windows.back());
            current_window.reset();
        }
    }

    if (current_window != nullptr && current_window->total_count_ > 0) {
        immutable_data_->windows.emplace_back(allocator_);
        compact_window_to_immutable(*current_window, immutable_data_->windows.back());
    }
    current_window.reset();
    window_term_list_.clear();
    window_term_list_.shrink_to_fit();
    this->cal_memory_usage();
    return failed_ids;
}

bool
SINDI::UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update) {
    // Note:
    // 1. we only check whether the old vector is a subset of the new vector
    // 2. we do not actually update the vector
    uint32_t inner_id;
    {
        std::shared_lock rlock(this->global_mutex_);
        inner_id = this->label_table_->GetIdByLabel(id);
    }
    const auto& new_sv = *new_base->GetSparseVectors();
    auto check_and_cleanup = [this, inner_id, &new_sv](auto&& get_sparse_vector) -> bool {
        SparseVector old_sv;
        get_sparse_vector(inner_id, &old_sv, this->allocator_);
        bool ret = is_subset_of_sparse_vector(old_sv, new_sv);

        this->allocator_->Deallocate(old_sv.vals_);
        this->allocator_->Deallocate(old_sv.ids_);
        return ret;
    };

    if (use_reorder_) {
        if (not check_and_cleanup(
                [this](InnerIdType inner_id, SparseVector* data, Allocator* allocator) {
                    rerank_flat_->GetSparseVectorByInnerId(inner_id, data, allocator);
                })) {
            return false;
        }
    }

    return check_and_cleanup(
        [this](InnerIdType inner_id, SparseVector* data, Allocator* allocator) {
            this->GetSparseVectorByInnerId(inner_id, data, allocator);
        });
}

DatasetPtr
SINDI::KnnSearch(const DatasetPtr& query,
                 int64_t k,
                 const std::string& parameters,
                 const FilterPtr& filter) const {
    return KnnSearch(query, k, parameters, filter, allocator_);
}

DatasetPtr
SINDI::KnnSearch(const DatasetPtr& query,
                 int64_t k,
                 const std::string& parameters,
                 const FilterPtr& filter,
                 vsag::Allocator* allocator) const {
    std::shared_lock rlock(this->global_mutex_);

    // Due to concerns about the performance of this index
    // We have not yet implemented search with filtering capabilities
    const auto* sparse_vectors = query->GetSparseVectors();
    CHECK_ARGUMENT(query->GetNumElements() == 1, "num of query should be 1");
    auto sparse_query = sparse_vectors[0];
    CHECK_ARGUMENT(
        sparse_query.len_ > 0,
        fmt::format("query->GetSparseVectors()->len_ ({}) is invalid", sparse_query.len_));

    // search parameter
    SINDISearchParameter search_param;
    search_param.FromJson(JsonType::Parse(parameters));
    CHECK_ARGUMENT(search_param.n_candidate <= SPARSE_AMPLIFICATION_FACTOR * k,
                   fmt::format("n_candidate ({}) should be less than {} * k ({})",
                               search_param.n_candidate,
                               AMPLIFICATION_FACTOR,
                               k));
    InnerSearchParam inner_param;
    inner_param.ef = std::max(static_cast<int64_t>(search_param.n_candidate), k);
    inner_param.topk = k;

    inner_param.is_inner_id_allowed = this->create_search_filter(filter);

    SparseVector effective_query = sparse_query;
    Vector<uint32_t> tmp_ids(allocator_);
    Vector<float> tmp_vals(allocator_);
    if (remap_term_ids_) {
        effective_query = remap_sparse_vector_for_query(sparse_query, tmp_ids, tmp_vals);
        if (effective_query.len_ == 0) {
            return make_empty_result();
        }
    }

    auto computer = std::make_shared<SparseTermComputer>(effective_query, search_param, allocator_);
    const SparseVector* rerank_query = (remap_term_ids_ && use_reorder_) ? &sparse_query : nullptr;
    if (immutable_data_ != nullptr) {
        return immutable_search_impl<KNN_SEARCH>(
            computer, inner_param, allocator, UseTermListsHeapInsert(search_param), rerank_query);
    }
    return search_impl<KNN_SEARCH>(
        computer, inner_param, allocator, UseTermListsHeapInsert(search_param), rerank_query);
}

std::optional<uint32_t>
SINDI::get_immutable_local_term(const ImmutableSINDIWindow& window, uint32_t term) const {
    if (remap_term_ids_) {
        auto it = std::lower_bound(
            window.sorted_global_terms.begin(), window.sorted_global_terms.end(), term);
        if (it == window.sorted_global_terms.end() || *it != term) {
            return std::nullopt;
        }
        return static_cast<uint32_t>(it - window.sorted_global_terms.begin());
    }
    if (term >= window.offsets.size()) {
        return std::nullopt;
    }
    return term;
}

void
SINDI::map_immutable_query_terms(const ImmutableSINDIWindow& window,
                                 const SparseTermComputerPtr& computer,
                                 ImmutableMappedQueryTerms& mapped_terms) const {
    mapped_terms.clear();
    mapped_terms.reserve(computer->pruned_len_);
    for (uint32_t it = 0; it < computer->pruned_len_; ++it) {
        auto local_term = get_immutable_local_term(window, computer->GetTerm(it));
        if (not local_term.has_value()) {
            continue;
        }
        const auto term = local_term.value();
        if (term + 1 >= window.offsets.size()) {
            continue;
        }
        if (window.offsets[term + 1] == window.offsets[term]) {
            continue;
        }
        mapped_terms.emplace_back(term, it);
    }
}

void
SINDI::scan_immutable_window_by_mapped_terms(float* dists,
                                             const ImmutableSINDIWindow& window,
                                             const SparseTermComputerPtr& computer,
                                             const ImmutableMappedQueryTerms& mapped_terms) const {
    const auto value_code_size = immutable_data_->value_code_size;
    for (uint32_t pos = 0; pos < mapped_terms.size(); ++pos) {
        const auto term_index = mapped_terms[pos].first;
        const auto it = mapped_terms[pos].second;
        if (pos + 1 < mapped_terms.size()) {
            const auto next_term = mapped_terms[pos + 1].first;
            const auto next_begin = window.offsets[next_term];
            __builtin_prefetch(window.id_payloads.data() + next_begin, 0, 3);
            __builtin_prefetch(
                window.value_payloads.data() + static_cast<uint64_t>(next_begin) * value_code_size,
                0,
                3);
        }
        const auto begin_doc = window.offsets[term_index];
        const auto end_doc = window.offsets[term_index + 1];
        const auto doc_count = end_doc - begin_doc;
        if (doc_count == 0) {
            continue;
        }
        const auto term_count =
            static_cast<uint32_t>(static_cast<float>(doc_count) * computer->term_retain_ratio_);
        const auto* ids = window.id_payloads.data() + begin_doc;
        const auto* values =
            window.value_payloads.data() + static_cast<uint64_t>(begin_doc) * value_code_size;
        if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
            computer->ScanForAccumulateSQ8(it, ids, values, term_count, dists);
        } else if (sparse_value_quant_type_ == SparseValueQuantizationType::FP16) {
            computer->ScanForAccumulateFP16Bytes(it, ids, values, term_count, dists);
        } else {
            computer->ScanForAccumulateFloatBytes(it, ids, values, term_count, dists);
        }
    }
    computer->ResetTerm();
}

template <InnerSearchMode mode, InnerSearchType type>
void
SINDI::immutable_insert_candidate_into_heap(uint32_t id,
                                            float& dist,
                                            float& cur_heap_top,
                                            MaxHeap& heap,
                                            uint32_t offset_id,
                                            float radius,
                                            int range_search_limit_size,
                                            const FilterPtr& filter) const {
    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        if (range_search_limit_size == 0) {
            dist = 0;
            return;
        }
    }

    if constexpr (type == InnerSearchType::WITH_FILTER) {
#if __cplusplus >= 202002L
        if (dist > cur_heap_top or not filter->CheckValid(id + offset_id)) [[likely]] {
#else
        if (__builtin_expect(dist > cur_heap_top or not filter->CheckValid(id + offset_id), 1)) {
#endif
            dist = 0;
            return;
        }
    } else {
#if __cplusplus >= 202002L
        if (dist > cur_heap_top) [[likely]] {
#else
        if (__builtin_expect(dist > cur_heap_top, 1)) {
#endif
            dist = 0;
            return;
        }
    }
    heap.emplace(dist, id + offset_id);
    if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
        heap.pop();
        cur_heap_top = heap.top().first;
    }
    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        if (range_search_limit_size > 0 and
            heap.size() > static_cast<uint32_t>(range_search_limit_size)) {
            heap.pop();
        }
        cur_heap_top = range_search_limit_size > 0 and
                               heap.size() == static_cast<uint32_t>(range_search_limit_size)
                           ? heap.top().first
                           : radius - 1;
    }
    dist = 0;
}

template <InnerSearchType type>
bool
SINDI::immutable_fill_heap_initial(uint32_t id,
                                   float& dist,
                                   float& cur_heap_top,
                                   MaxHeap& heap,
                                   uint32_t offset_id,
                                   uint32_t n_candidate,
                                   const FilterPtr& filter) const {
    if (dist < 0) {
        if constexpr (type == InnerSearchType::WITH_FILTER) {
            if (not filter->CheckValid(id + offset_id)) {
                dist = 0;
                return false;
            }
        }
        heap.emplace(dist, id + offset_id);
        cur_heap_top = heap.top().first;
        dist = 0;
        return heap.size() == n_candidate;
    }
    return false;
}

template <InnerSearchMode mode, InnerSearchType type>
void
SINDI::immutable_insert_heap_by_mapped_terms(float* dists,
                                             const ImmutableSINDIWindow& window,
                                             const SparseTermComputerPtr& computer,
                                             const ImmutableMappedQueryTerms& mapped_terms,
                                             MaxHeap& heap,
                                             const InnerSearchParam& param,
                                             uint32_t offset_id) const {
    uint32_t id = 0;
    float cur_heap_top = std::numeric_limits<float>::max();
    auto n_candidate = param.ef;
    auto radius = param.radius;
    auto range_search_limit_size = param.range_search_limit_size;
    auto filter = param.is_inner_id_allowed;

    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        cur_heap_top = radius - 1;
    }

    for (const auto& [term, term_iterator] : mapped_terms) {
        (void)term_iterator;
        const auto begin_doc = window.offsets[term];
        const auto end_doc = window.offsets[term + 1];
        const auto doc_count = end_doc - begin_doc;
        const auto term_count =
            static_cast<uint32_t>(static_cast<float>(doc_count) * computer->term_retain_ratio_);
        const auto* ids = window.id_payloads.data() + begin_doc;
        uint32_t i = 0;
        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            if (heap.size() < n_candidate) {
                for (; i < term_count; ++i) {
                    id = ids[i];
                    if (immutable_fill_heap_initial<type>(
                            id, dists[id], cur_heap_top, heap, offset_id, n_candidate, filter)) {
                        ++i;
                        break;
                    }
                }
            }
        }

        for (; i < term_count; ++i) {
            id = ids[i];
            immutable_insert_candidate_into_heap<mode, type>(id,
                                                             dists[id],
                                                             cur_heap_top,
                                                             heap,
                                                             offset_id,
                                                             radius,
                                                             range_search_limit_size,
                                                             filter);
        }
    }
}

template <InnerSearchMode mode, InnerSearchType type>
void
SINDI::immutable_insert_heap_by_dists(float* dists,
                                      uint32_t dists_size,
                                      MaxHeap& heap,
                                      const InnerSearchParam& param,
                                      uint32_t offset_id) const {
    float cur_heap_top = std::numeric_limits<float>::max();
    auto n_candidate = param.ef;
    auto radius = param.radius;
    auto range_search_limit_size = param.range_search_limit_size;
    auto filter = param.is_inner_id_allowed;

    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        cur_heap_top = radius - 1;
    }

    uint32_t id = 0;
    if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
        if (heap.size() < n_candidate) {
            for (; id < dists_size; ++id) {
                if (immutable_fill_heap_initial<type>(
                        id, dists[id], cur_heap_top, heap, offset_id, n_candidate, filter)) {
                    ++id;
                    break;
                }
            }
        }
    }

    for (; id < dists_size; ++id) {
        immutable_insert_candidate_into_heap<mode, type>(
            id, dists[id], cur_heap_top, heap, offset_id, radius, range_search_limit_size, filter);
    }
}

template <InnerSearchMode mode>
DatasetPtr
SINDI::immutable_search_impl(const SparseTermComputerPtr& computer,
                             const InnerSearchParam& inner_param,
                             Allocator* allocator,
                             bool use_term_lists_heap_insert,
                             const SparseVector* original_query) const {
    Allocator* search_allocator = allocator != nullptr ? allocator : allocator_;
    MaxHeap heap(search_allocator);
    int64_t k = 0;
    if constexpr (mode == KNN_SEARCH) {
        k = inner_param.topk;
    }

    Vector<float> dists(window_size_, 0.0F, search_allocator);
    ImmutableMappedQueryTerms mapped_terms(search_allocator);
    auto filter = inner_param.is_inner_id_allowed;
    const auto [min_window_id, max_window_id] = this->get_min_max_window_id(filter);
    for (auto cur = min_window_id; cur <= max_window_id; cur++) {
        const auto& window = immutable_data_->windows[static_cast<uint32_t>(cur)];
        const auto window_start_id = static_cast<uint32_t>(cur) * window_size_;
        map_immutable_query_terms(window, computer, mapped_terms);
        std::fill(dists.begin(), dists.end(), 0.0F);
        scan_immutable_window_by_mapped_terms(dists.data(), window, computer, mapped_terms);

        if (use_term_lists_heap_insert) {
            if (inner_param.is_inner_id_allowed) {
                immutable_insert_heap_by_mapped_terms<mode, WITH_FILTER>(dists.data(),
                                                                         window,
                                                                         computer,
                                                                         mapped_terms,
                                                                         heap,
                                                                         inner_param,
                                                                         window_start_id);
            } else {
                immutable_insert_heap_by_mapped_terms<mode, PURE>(dists.data(),
                                                                  window,
                                                                  computer,
                                                                  mapped_terms,
                                                                  heap,
                                                                  inner_param,
                                                                  window_start_id);
            }
        } else {
            const auto window_doc_count = static_cast<uint32_t>(std::min<int64_t>(
                window_size_, cur_element_count_ - static_cast<int64_t>(window_start_id)));
            if (inner_param.is_inner_id_allowed) {
                immutable_insert_heap_by_dists<mode, WITH_FILTER>(
                    dists.data(), window_doc_count, heap, inner_param, window_start_id);
            } else {
                immutable_insert_heap_by_dists<mode, PURE>(
                    dists.data(), window_doc_count, heap, inner_param, window_start_id);
            }
        }
    }

    if (use_reorder_) {
        float cur_heap_top = std::numeric_limits<float>::max();
        auto candidate_size = heap.size();
        auto high_precise_heap = std::make_shared<StandardHeap<true, false>>(search_allocator, -1);
        auto [sorted_ids, sorted_vals] =
            sort_sparse_vector(original_query ? *original_query : computer->raw_query_, allocator_);
        for (auto i = 0; i < candidate_size; i++) {
            auto inner_id = heap.top().second;
            auto high_precise_distance =
                calc_distance_by_id_unsafe(rerank_flat_, sorted_ids, sorted_vals, inner_id);
            auto label = label_table_->GetLabelById(inner_id);
            if constexpr (mode == KNN_SEARCH) {
                if (high_precise_distance < cur_heap_top or high_precise_heap->Size() < k) {
                    high_precise_heap->Push(high_precise_distance, label);
                }
                if (high_precise_heap->Size() > k) {
                    high_precise_heap->Pop();
                }
                cur_heap_top = high_precise_heap->Top().first;
            }
            if constexpr (mode == RANGE_SEARCH) {
                if (high_precise_distance <= inner_param.radius) {
                    high_precise_heap->Push(high_precise_distance, label);
                }
                if (inner_param.range_search_limit_size != -1 and
                    high_precise_heap->Size() > inner_param.range_search_limit_size) {
                    high_precise_heap->Pop();
                }
            }
            heap.pop();
        }
        return collect_heap_results(high_precise_heap, search_allocator);
    }

    if constexpr (mode == RANGE_SEARCH) {
        k = static_cast<int64_t>(heap.size());
        if (inner_param.range_search_limit_size != -1) {
            k = inner_param.range_search_limit_size;
        }
    }
    int64_t cur_size = std::min(static_cast<int64_t>(heap.size()), k);
    auto [results, ret_dists, ret_ids] = create_fast_dataset(cur_size, search_allocator);
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

template <InnerSearchMode mode>
DatasetPtr
SINDI::search_impl(const SparseTermComputerPtr& computer,
                   const InnerSearchParam& inner_param,
                   Allocator* allocator,
                   bool use_term_lists_heap_insert,
                   const SparseVector* original_query) const {
    // computer and heap
    MaxHeap heap(allocator);
    int64_t k = 0;

    if constexpr (mode == KNN_SEARCH) {
        k = inner_param.topk;
    }

    // window iteration
    Vector<float> dists(window_size_, 0.0, allocator);
    auto filter = inner_param.is_inner_id_allowed;
    const auto [min_window_id, max_window_id] = this->get_min_max_window_id(filter);
    for (auto cur = min_window_id; cur <= max_window_id; cur++) {
        auto window_start_id = cur * window_size_;
        auto term_list = this->window_term_list_[cur];

        // compute
        term_list->Query(dists.data(), computer);

        // insert heap
        if (use_term_lists_heap_insert) {
            if (inner_param.is_inner_id_allowed) {
                term_list->InsertHeapByTermLists<mode, WITH_FILTER>(
                    dists.data(), computer, heap, inner_param, window_start_id);
            } else {
                term_list->InsertHeapByTermLists<mode, PURE>(
                    dists.data(), computer, heap, inner_param, window_start_id);
            }
        } else {
            if (inner_param.is_inner_id_allowed) {
                term_list->InsertHeapByDists<mode, WITH_FILTER>(
                    dists.data(), dists.size(), heap, inner_param, window_start_id);
            } else {
                term_list->InsertHeapByDists<mode, PURE>(
                    dists.data(), dists.size(), heap, inner_param, window_start_id);
            }
        }
    }

    // rerank
    if (use_reorder_) {
        // high precision
        float cur_heap_top = std::numeric_limits<float>::max();
        auto candidate_size = heap.size();
        auto high_precise_heap = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
        auto [sorted_ids, sorted_vals] =
            sort_sparse_vector(original_query ? *original_query : computer->raw_query_, allocator_);
        for (auto i = 0; i < candidate_size; i++) {
            auto inner_id = heap.top().second;
            auto high_precise_distance =
                calc_distance_by_id_unsafe(rerank_flat_, sorted_ids, sorted_vals, inner_id);
            auto label = label_table_->GetLabelById(inner_id);
            if constexpr (mode == KNN_SEARCH) {
                if (high_precise_distance < cur_heap_top or high_precise_heap->Size() < k) {
                    high_precise_heap->Push(high_precise_distance, label);
                }
                if (high_precise_heap->Size() > k) {
                    high_precise_heap->Pop();
                }
                cur_heap_top = high_precise_heap->Top().first;
            }
            if constexpr (mode == RANGE_SEARCH) {
                if (high_precise_distance <= inner_param.radius) {
                    high_precise_heap->Push(high_precise_distance, label);
                }
                if (inner_param.range_search_limit_size != -1 and
                    high_precise_heap->Size() > inner_param.range_search_limit_size) {
                    high_precise_heap->Pop();
                }
            }
            heap.pop();
        }

        return collect_heap_results(high_precise_heap, allocator);
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
        ret_dists[j] = 1 + heap.top().first;  // dist = -ip -> 1 + dist = 1 - ip
        ret_ids[j] = label_table_->GetLabelById(heap.top().second);
        heap.pop();
    }

    return results;
}

DatasetPtr
SINDI::RangeSearch(const DatasetPtr& query,
                   float radius,
                   const std::string& parameters,
                   const FilterPtr& filter,
                   int64_t limited_size) const {
    std::shared_lock rlock(this->global_mutex_);

    // Due to concerns about the performance of this index
    // We have not yet implemented search with filtering capabilities
    const auto* sparse_vectors = query->GetSparseVectors();
    CHECK_ARGUMENT(query->GetNumElements() == 1, "num of query should be 1");
    auto sparse_query = sparse_vectors[0];
    CHECK_ARGUMENT(
        sparse_query.len_ > 0,
        fmt::format("query->GetSparseVectors()->len_ ({}) is invalid", sparse_query.len_));

    // search parameter
    SINDISearchParameter search_param;
    search_param.FromJson(JsonType::Parse(parameters));
    InnerSearchParam inner_param;

    inner_param.range_search_limit_size = static_cast<int>(limited_size);
    inner_param.radius = radius;

    inner_param.is_inner_id_allowed = this->create_search_filter(filter);

    SparseVector effective_query = sparse_query;
    Vector<uint32_t> tmp_ids(allocator_);
    Vector<float> tmp_vals(allocator_);
    if (remap_term_ids_) {
        effective_query = remap_sparse_vector_for_query(sparse_query, tmp_ids, tmp_vals);
        if (effective_query.len_ == 0) {
            return make_empty_result();
        }
    }

    auto computer = std::make_shared<SparseTermComputer>(effective_query, search_param, allocator_);
    const SparseVector* rerank_query = (remap_term_ids_ && use_reorder_) ? &sparse_query : nullptr;
    if (immutable_data_ != nullptr) {
        return immutable_search_impl<RANGE_SEARCH>(
            computer, inner_param, allocator_, UseTermListsHeapInsert(search_param), rerank_query);
    }
    return search_impl<RANGE_SEARCH>(
        computer, inner_param, allocator_, UseTermListsHeapInsert(search_param), rerank_query);
}

bool
SINDI::UseTermListsHeapInsert(const SINDISearchParameter& search_param) const {
    // Low build-time doc pruning and low search-time query pruning keep the old
    // distance-array heap insertion path for accuracy. Once either side exceeds the
    // threshold, term-list heap insertion avoids scanning the whole window for heap updates.
    return doc_prune_ratio_ > SINDI::K_TERM_LISTS_HEAP_INSERT_PRUNE_THRESHOLD ||
           search_param.query_prune_ratio > SINDI::K_TERM_LISTS_HEAP_INSERT_PRUNE_THRESHOLD;
}

void
SINDI::cal_memory_usage() {
    auto memory = sizeof(SINDI);
    if (immutable_data_ != nullptr) {
        memory += sizeof(ImmutableSINDIData);
        memory += immutable_data_->windows.size() * sizeof(ImmutableSINDIWindow);
        for (const auto& window : immutable_data_->windows) {
            memory += window.sorted_global_terms.size() * sizeof(uint32_t);
            memory += window.offsets.size() * sizeof(uint32_t);
            memory += window.id_payloads.size() * sizeof(uint16_t);
            memory += window.value_payloads.size() * sizeof(uint8_t);
        }
    } else {
        memory += window_term_list_.size() * sizeof(SparseTermDataCellPtr);
        for (auto& window : window_term_list_) {
            memory += window->GetMemoryUsage();
        }
    }
    memory += label_table_->GetMemoryUsage();
    if (this->rerank_flat_ != nullptr) {
        memory += this->rerank_flat_->GetMemoryUsage();
    }
    memory += sizeof(QuantizationParams);
    if (remap_term_ids_ && term_id_mapper_) {
        memory +=
            static_cast<uint64_t>(term_id_mapper_->Size()) * TERM_ID_MAPPER_ENTRY_MEMORY_BYTES;
    }

    std::unique_lock lock(this->memory_usage_mutex_);
    this->current_memory_usage_.store(memory);
}

void
SINDI::Serialize(StreamWriter& writer) const {
    std::shared_lock rlock(this->global_mutex_);

    if (cur_element_count_ == 0) {
        auto cur_element_count = cur_element_count_.load();
        StreamWriter::WriteObj(writer, cur_element_count);
        if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
            StreamWriter::WriteObj(writer, quantization_params_->min_val);
            StreamWriter::WriteObj(writer, quantization_params_->max_val);
            StreamWriter::WriteObj(writer, quantization_params_->diff);
        }
        uint32_t window_term_list_size = 0;
        StreamWriter::WriteObj(writer, window_term_list_size);
        label_table_->Serialize(writer);

        auto metadata = std::make_shared<Metadata>();
        metadata->SetEmptyIndex(true);
        auto footer = std::make_shared<Footer>(metadata);
        footer->Write(writer);
        return;
    }

    auto cur_element_count = cur_element_count_.load();
    StreamWriter::WriteObj(writer, cur_element_count);

    if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
        StreamWriter::WriteObj(writer, quantization_params_->min_val);
        StreamWriter::WriteObj(writer, quantization_params_->max_val);
        StreamWriter::WriteObj(writer, quantization_params_->diff);
    }

    uint32_t window_term_list_size =
        immutable_data_ != nullptr ? immutable_data_->windows.size() : window_term_list_.size();
    StreamWriter::WriteObj(writer, window_term_list_size);
    if (immutable_data_ != nullptr) {
        for (const auto& window : immutable_data_->windows) {
            serialize_immutable_window(writer, window);
        }
    } else {
        for (const auto& window : window_term_list_) {
            window->Serialize(writer);
        }
    }

    label_table_->Serialize(writer);

    if (use_reorder_) {
        rerank_flat_->Serialize(writer);
    }

    if (remap_term_ids_ && term_id_mapper_) {
        term_id_mapper_->Serialize(writer);
    }

    JsonType jsonify_basic_info;
    jsonify_basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());
    if (use_reorder_) {
        jsonify_basic_info[SINDI_RERANK_FLAT_FORMAT_KEY].SetInt(SINDI_RERANK_FLAT_FORMAT_DATACELL);
    }
    write_index_footer(writer, jsonify_basic_info);
}

MetadataPtr
SINDI::collect_streaming_header() const {
    auto metadata = std::make_shared<Metadata>();
    metadata->Set("format", "vsag_stream_v1");
    metadata->Set("index_name", this->GetName());

    JsonType basic_info;
    basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());
    basic_info["dim"].SetInt(dim_);
    basic_info["metric"].SetInt(static_cast<int64_t>(metric_));
    basic_info["data_type"].SetInt(static_cast<int64_t>(data_type_));
    basic_info["extra_info_size"].SetInt(static_cast<int64_t>(extra_info_size_));
    basic_info["cur_element_count"].SetInt(this->cur_element_count_.load());
    basic_info["use_reorder"].SetBool(this->use_reorder_);
    basic_info["remap_term_ids"].SetBool(this->remap_term_ids_);
    if (use_reorder_) {
        basic_info[SINDI_RERANK_FLAT_FORMAT_KEY].SetInt(SINDI_RERANK_FLAT_FORMAT_DATACELL);
    }
    metadata->Set(BASIC_INFO, basic_info);

    JsonType manifest;
    auto windows_tag = static_cast<uint32_t>(StreamSerializationTag::SINDI_WINDOWS);
    auto label_tag = static_cast<uint32_t>(StreamSerializationTag::LABEL_TABLE);
    AppendStreamingManifestBlock(manifest,
                                 windows_tag,
                                 StreamSerializationBlockCurrentVersion(windows_tag),
                                 StreamSerializationTagCritical(windows_tag));
    AppendStreamingManifestBlock(manifest,
                                 label_tag,
                                 StreamSerializationBlockCurrentVersion(label_tag),
                                 StreamSerializationTagCritical(label_tag));
    if (this->use_reorder_) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::SINDI_RERANK_INDEX);
        AppendStreamingManifestBlock(manifest,
                                     tag,
                                     StreamSerializationBlockCurrentVersion(tag),
                                     StreamSerializationTagCritical(tag));
    }
    if (this->remap_term_ids_ && this->term_id_mapper_) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::SINDI_TERM_ID_MAPPER);
        AppendStreamingManifestBlock(manifest,
                                     tag,
                                     StreamSerializationBlockCurrentVersion(tag),
                                     StreamSerializationTagCritical(tag));
    }
    metadata->Set("block_manifest", manifest);
    metadata->SetEmptyIndex(this->GetNumElements() == 0);
    return metadata;
}

void
SINDI::serialize_windows(StreamWriter& writer) const {
    auto cur_element_count = cur_element_count_.load();
    StreamWriter::WriteObj(writer, cur_element_count);

    if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
        StreamWriter::WriteObj(writer, quantization_params_->min_val);
        StreamWriter::WriteObj(writer, quantization_params_->max_val);
        StreamWriter::WriteObj(writer, quantization_params_->diff);
    }

    uint32_t window_term_list_size = window_term_list_.size();
    StreamWriter::WriteObj(writer, window_term_list_size);
    for (const auto& window : window_term_list_) {
        window->Serialize(writer);
    }
}

void
SINDI::serialize_streaming_body(StreamWriter& writer) const {
    std::shared_lock rlock(this->global_mutex_);
    const bool mutable_runtime = not immutable_enabled_ and immutable_data_ == nullptr;
    CHECK_ARGUMENT(mutable_runtime, "immutable SINDI runtime does not support SerializeStreaming");

    auto windows_tag = static_cast<uint32_t>(StreamSerializationTag::SINDI_WINDOWS);
    auto label_tag = static_cast<uint32_t>(StreamSerializationTag::LABEL_TABLE);
    WriteStreamingBlock(
        writer, windows_tag, StreamSerializationTagCritical(windows_tag), [this](StreamWriter& w) {
            this->serialize_windows(w);
        });
    WriteStreamingBlock(
        writer, label_tag, StreamSerializationTagCritical(label_tag), [this](StreamWriter& w) {
            this->label_table_->Serialize(w);
        });
    if (this->use_reorder_) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::SINDI_RERANK_INDEX);
        WriteStreamingBlock(
            writer, tag, StreamSerializationTagCritical(tag), [this](StreamWriter& w) {
                this->rerank_flat_->Serialize(w);
            });
    }
    if (this->remap_term_ids_ && this->term_id_mapper_) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::SINDI_TERM_ID_MAPPER);
        WriteStreamingBlock(
            writer, tag, StreamSerializationTagCritical(tag), [this](StreamWriter& w) {
                this->term_id_mapper_->Serialize(w);
            });
    }
}

void
SINDI::deserialize_windows(StreamReader& reader_ref) {
    uint64_t cur_element_count = 0;
    StreamReader::ReadObj(reader_ref, cur_element_count);
    CHECK_ARGUMENT(cur_element_count <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                   "SINDI element count overflows int64_t");
    cur_element_count_.store(static_cast<int64_t>(cur_element_count));

    if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
        StreamReader::ReadObj(reader_ref, quantization_params_->min_val);
        StreamReader::ReadObj(reader_ref, quantization_params_->max_val);
        StreamReader::ReadObj(reader_ref, quantization_params_->diff);
    }

    uint32_t window_term_list_size = 0;
    StreamReader::ReadObj(reader_ref, window_term_list_size);
    if (immutable_enabled_) {
        immutable_data_ = std::make_unique<ImmutableSINDIData>(allocator_);
        immutable_data_->sparse_value_quant_type = sparse_value_quant_type_;
        immutable_data_->value_code_size = sparse_value_code_size(sparse_value_quant_type_);
        immutable_data_->windows.reserve(window_term_list_size);
        for (uint32_t i = 0; i < window_term_list_size; ++i) {
            immutable_data_->windows.emplace_back(allocator_);
            deserialize_immutable_window(reader_ref, immutable_data_->windows.back());
        }
        window_term_list_.clear();
        window_term_list_.shrink_to_fit();
    } else {
        window_term_list_.resize(window_term_list_size);
        for (auto& window : window_term_list_) {
            window = std::make_shared<SparseTermDataCell>(doc_retain_ratio_,
                                                          term_id_limit_,
                                                          allocator_,
                                                          sparse_value_quant_type_,
                                                          quantization_params_);
            window->Deserialize(reader_ref);
        }
    }
}

void
SINDI::deserialize_streaming_body(StreamReader& reader, const MetadataPtr& metadata) {
    this->read_streaming_body(reader, metadata);
}

void
SINDI::load_streaming_body(StreamReader& reader,
                           const MetadataPtr& metadata,
                           const LoadParameters& parameters) {
    (void)parameters;
    this->read_streaming_body(reader, metadata);
}

void
SINDI::read_streaming_body(StreamReader& reader, const MetadataPtr& metadata) {
    std::scoped_lock wlock(this->global_mutex_);

    auto basic_info = metadata->Get(BASIC_INFO);
    if (basic_info.Contains(INDEX_PARAM)) {
        auto index_param = std::make_shared<SINDIParameter>();
        index_param->FromString(basic_info[INDEX_PARAM].GetString());
        if (not this->create_param_ptr_->CheckCompatibility(index_param)) {
            auto message = fmt::format("SINDI index parameter not match, current: {}, new: {}",
                                       this->create_param_ptr_->ToString(),
                                       index_param->ToString());
            logger::error(message);
            throw VsagException(ErrorType::INVALID_ARGUMENT, message);
        }
    }

    bool loaded_windows = false;
    bool loaded_label_table = false;
    bool loaded_rerank = false;
    bool loaded_term_mapper = false;

    while (true) {
        auto block_header = StreamBlockHeader::Read(reader);
        if (block_header.IsSectionEnd()) {
            break;
        }
        BoundedForwardReader block_reader(&reader, block_header.value_len);
        if (!StreamSerializationBlockVersionSupported(block_header.tag,
                                                      block_header.block_version)) {
            if (block_header.IsCritical()) {
                throw VsagException(
                    ErrorType::UNSUPPORTED_INDEX_OPERATION,
                    fmt::format("unsupported SINDI streaming block version: tag={}, "
                                "name={}, version={}, flags={}, value_len={}",
                                block_header.tag,
                                StreamSerializationTagName(block_header.tag),
                                block_header.block_version,
                                block_header.flags,
                                block_header.value_len));
            }
            block_reader.SkipRemaining();
            continue;
        }

        switch (static_cast<StreamSerializationTag>(block_header.tag)) {
            case StreamSerializationTag::SINDI_WINDOWS:
                ReadSeekableBlockPayload(block_reader, block_header, [this](StreamReader& block) {
                    this->deserialize_windows(block);
                });
                loaded_windows = true;
                break;
            case StreamSerializationTag::LABEL_TABLE:
                ReadSeekableBlockPayload(block_reader, block_header, [this](StreamReader& block) {
                    this->label_table_->Deserialize(block);
                });
                this->delete_count_.store(
                    static_cast<int64_t>(this->label_table_->GetAllDeletedIds().size()),
                    std::memory_order_relaxed);
                loaded_label_table = true;
                break;
            case StreamSerializationTag::SINDI_RERANK_INDEX:
                if (this->use_reorder_) {
                    ReadSeekableBlockPayload(
                        block_reader, block_header, [this](StreamReader& block) {
                            deserialize_rerank_flat(
                                block, this->rerank_flat_, this->allocator_, true);
                        });
                    loaded_rerank = true;
                }
                break;
            case StreamSerializationTag::SINDI_TERM_ID_MAPPER:
                if (this->remap_term_ids_ && this->term_id_mapper_) {
                    ReadSeekableBlockPayload(
                        block_reader, block_header, [this](StreamReader& block) {
                            this->term_id_mapper_->Deserialize(block);
                        });
                    loaded_term_mapper = true;
                }
                break;
            default:
                if (block_header.IsCritical()) {
                    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                        fmt::format("unknown SINDI streaming serialization block: "
                                                    "tag={}, name={}, version={}, flags={}, "
                                                    "value_len={}",
                                                    block_header.tag,
                                                    StreamSerializationTagName(block_header.tag),
                                                    block_header.block_version,
                                                    block_header.flags,
                                                    block_header.value_len));
                }
                break;
        }
        block_reader.SkipRemaining();
    }

    if (!loaded_windows || !loaded_label_table) {
        throw VsagException(ErrorType::READ_ERROR,
                            "SINDI streaming serialization required block is missing");
    }
    if (this->use_reorder_ && !loaded_rerank) {
        throw VsagException(ErrorType::READ_ERROR,
                            "SINDI streaming serialization rerank block is missing");
    }
    if (this->remap_term_ids_ && this->term_id_mapper_ && !loaded_term_mapper) {
        throw VsagException(ErrorType::READ_ERROR,
                            "SINDI streaming serialization term mapper block is missing");
    }
    this->cal_memory_usage();
}

void
SINDI::Deserialize(StreamReader& reader) {
    std::scoped_lock wlock(this->global_mutex_);

    bool has_datacell_rerank_format = false;
    bool has_footer = false;
    if (not deserialize_without_footer_) {
        JsonType jsonify_basic_info;
        try {
            if (read_index_footer(reader, jsonify_basic_info)) {
                has_footer = true;
                // Check if the index parameter is compatible
                {
                    auto param = jsonify_basic_info[INDEX_PARAM].GetString();
                    SINDIParameterPtr index_param = std::make_shared<SINDIParameter>();
                    index_param->FromString(param);
                    CHECK_ARGUMENT(
                        index_param->immutable == immutable_enabled_,
                        fmt::format("SINDI immutable format mismatch: runtime={}, storage={}",
                                    immutable_enabled_,
                                    index_param->immutable));
                    if (not this->create_param_ptr_->CheckCompatibility(index_param)) {
                        auto message =
                            fmt::format("SINDI index parameter not match, current: {}, new: {}",
                                        this->create_param_ptr_->ToString(),
                                        index_param->ToString());
                        logger::error(message);
                        throw VsagException(ErrorType::INVALID_ARGUMENT, message);
                    }
                    doc_prune_ratio_ = index_param->doc_prune_ratio;
                    doc_retain_ratio_ = 1.0F - doc_prune_ratio_;
                }
                if (jsonify_basic_info.Contains(SINDI_RERANK_FLAT_FORMAT_KEY)) {
                    has_datacell_rerank_format =
                        jsonify_basic_info[SINDI_RERANK_FLAT_FORMAT_KEY].GetInt() ==
                        SINDI_RERANK_FLAT_FORMAT_DATACELL;
                }
            } else {
                logger::debug("SINDI footer not found, fallback to legacy deserialize path");
            }
        } catch (const VsagException& e) {
            if (e.error_.type == ErrorType::INDEX_EMPTY) {
                return;
            }
            throw;
        }
    }
    const bool valid_footer = not immutable_enabled_ or has_footer;
    CHECK_ARGUMENT(valid_footer,
                   "cannot deserialize SINDI storage without immutable format metadata into "
                   "immutable runtime");
    auto* reader_ptr = &reader;

    BufferStreamReader buffer_reader(
        &reader, std::numeric_limits<uint64_t>::max(), this->allocator_);
    if (not deserialize_without_buffer_ && not immutable_enabled_) {
        reader_ptr = &buffer_reader;
    }
    auto& reader_ref = *reader_ptr;

    uint64_t cur_element_count = 0;
    StreamReader::ReadObj(reader_ref, cur_element_count);
    CHECK_ARGUMENT(cur_element_count <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                   "SINDI element count overflows int64_t");
    cur_element_count_.store(static_cast<int64_t>(cur_element_count));

    if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
        StreamReader::ReadObj(reader_ref, quantization_params_->min_val);
        StreamReader::ReadObj(reader_ref, quantization_params_->max_val);
        StreamReader::ReadObj(reader_ref, quantization_params_->diff);
    }

    uint32_t window_term_list_size = 0;
    StreamReader::ReadObj(reader_ref, window_term_list_size);
    if (immutable_enabled_) {
        immutable_data_ = std::make_unique<ImmutableSINDIData>(allocator_);
        immutable_data_->sparse_value_quant_type = sparse_value_quant_type_;
        immutable_data_->value_code_size = sparse_value_code_size(sparse_value_quant_type_);
        immutable_data_->windows.reserve(window_term_list_size);
        for (uint32_t i = 0; i < window_term_list_size; ++i) {
            immutable_data_->windows.emplace_back(allocator_);
            deserialize_immutable_window(reader_ref, immutable_data_->windows.back());
        }
        window_term_list_.clear();
        window_term_list_.shrink_to_fit();
    } else {
        window_term_list_.resize(window_term_list_size);
        for (auto& window : window_term_list_) {
            window = std::make_shared<SparseTermDataCell>(doc_retain_ratio_,
                                                          term_id_limit_,
                                                          allocator_,
                                                          sparse_value_quant_type_,
                                                          quantization_params_);
            window->Deserialize(reader_ref);
        }
    }

    label_table_->Deserialize(reader_ref);
    delete_count_.store(static_cast<int64_t>(label_table_->GetAllDeletedIds().size()),
                        std::memory_order_relaxed);

    if (cur_element_count_ == 0) {
        this->cal_memory_usage();
        return;
    }

    if (use_reorder_) {
        has_datacell_rerank_format = has_datacell_rerank_format ||
                                     detect_datacell_rerank_flat(reader_ref, cur_element_count_);
        deserialize_rerank_flat(reader_ref, rerank_flat_, allocator_, has_datacell_rerank_format);
    }

    if (remap_term_ids_ && term_id_mapper_) {
        term_id_mapper_->Deserialize(reader_ref);
    }

    this->cal_memory_usage();
}

void
SINDI::compact_window_to_immutable(const SparseTermDataCell& term_list,
                                   ImmutableSINDIWindow& window) const {
    const auto value_code_size = sparse_value_code_size(sparse_value_quant_type_);
    uint64_t total_posting_count = 0;
    window.offsets.clear();
    window.id_payloads.clear();
    window.value_payloads.clear();
    window.sorted_global_terms.clear();

    if (not remap_term_ids_) {
        window.offsets.reserve(static_cast<uint64_t>(term_list.term_capacity_) + 1);
    }
    window.offsets.push_back(0);

    for (uint32_t term = 0; term < term_list.term_capacity_; ++term) {
        const auto posting_count = term_list.term_sizes_[term];
        if (posting_count == 0) {
            if (not remap_term_ids_) {
                window.offsets.push_back(static_cast<uint32_t>(total_posting_count));
            }
            continue;
        }

        if (total_posting_count >
            std::numeric_limits<uint32_t>::max() - static_cast<uint64_t>(posting_count)) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "immutable SINDI posting offset overflows uint32_t");
        }
        if (remap_term_ids_) {
            window.sorted_global_terms.push_back(term);
        }

        const auto old_id_size = window.id_payloads.size();
        window.id_payloads.resize(old_id_size + posting_count);
        const auto old_id_position =
            static_cast<decltype(window.id_payloads)::difference_type>(old_id_size);
        std::copy(term_list.term_ids_[term]->begin(),
                  term_list.term_ids_[term]->end(),
                  window.id_payloads.begin() + old_id_position);

        const auto payload_bytes = static_cast<uint64_t>(posting_count) * value_code_size;
        const auto old_value_size = window.value_payloads.size();
        window.value_payloads.resize(old_value_size + payload_bytes);
        std::memcpy(window.value_payloads.data() + old_value_size,
                    term_list.term_datas_[term]->data(),
                    payload_bytes);

        total_posting_count += posting_count;
        window.offsets.push_back(static_cast<uint32_t>(total_posting_count));
    }
}

void
SINDI::serialize_immutable_window(StreamWriter& writer, const ImmutableSINDIWindow& window) {
    StreamWriter::WriteVector(writer, window.sorted_global_terms);
    StreamWriter::WriteVector(writer, window.offsets);
    StreamWriter::WriteVector(writer, window.id_payloads);
    StreamWriter::WriteVector(writer, window.value_payloads);
}

void
SINDI::deserialize_immutable_window(StreamReader& reader_ref, ImmutableSINDIWindow& window) const {
    const auto value_code_size = sparse_value_code_size(sparse_value_quant_type_);
    const auto read_vector = [&reader_ref](auto& values, uint64_t max_size, const char* message) {
        uint64_t size = 0;
        StreamReader::ReadObj(reader_ref, size);
        const auto max_value_size = static_cast<uint64_t>(values.max_size());
        const bool valid_size = size <= max_size and size <= max_value_size;
        CHECK_ARGUMENT(valid_size, message);
        values.resize(size);
        reader_ref.Read(reinterpret_cast<char*>(values.data()),
                        size * sizeof(typename std::decay_t<decltype(values)>::value_type));
    };

    read_vector(window.sorted_global_terms,
                remap_term_ids_ ? term_id_limit_ : 0,
                remap_term_ids_ ? "immutable SINDI remapped term count exceeds term_id_limit"
                                : "immutable SINDI terms require remap_term_ids");
    uint64_t offset_count = 0;
    StreamReader::ReadObj(reader_ref, offset_count);
    const auto max_offset_count = remap_term_ids_ ? static_cast<uint64_t>(term_id_limit_) + 1
                                                  : static_cast<uint64_t>(term_id_limit_) * 2 + 1;
    const auto max_offset_capacity = static_cast<uint64_t>(window.offsets.max_size());
    const bool valid_offset_count =
        offset_count <= max_offset_count and offset_count <= max_offset_capacity;
    CHECK_ARGUMENT(valid_offset_count, "immutable SINDI offset count exceeds capacity bound");
    if (remap_term_ids_) {
        CHECK_ARGUMENT(offset_count == static_cast<uint64_t>(window.sorted_global_terms.size()) + 1,
                       "immutable SINDI term and offset counts do not match");
    }
    window.offsets.resize(offset_count);
    reader_ref.Read(reinterpret_cast<char*>(window.offsets.data()),
                    offset_count * sizeof(uint32_t));

    CHECK_ARGUMENT(not window.offsets.empty(), "immutable SINDI offsets must not be empty");
    CHECK_ARGUMENT(window.offsets.front() == 0, "immutable SINDI offsets must start at zero");
    CHECK_ARGUMENT(std::is_sorted(window.offsets.begin(), window.offsets.end()),
                   "immutable SINDI offsets must be sorted");
    if (remap_term_ids_) {
        CHECK_ARGUMENT(
            std::adjacent_find(window.sorted_global_terms.begin(),
                               window.sorted_global_terms.end(),
                               std::greater_equal<>()) == window.sorted_global_terms.end(),
            "immutable SINDI remapped terms must be strictly increasing");
    } else {
        CHECK_ARGUMENT(window.sorted_global_terms.empty(),
                       "immutable SINDI dense window must not contain remapped terms");
        CHECK_ARGUMENT(window.offsets.size() <= static_cast<uint64_t>(term_id_limit_) * 2 + 1,
                       "immutable SINDI term capacity exceeds capacity bound");
    }
    for (uint64_t i = 1; i < window.offsets.size(); ++i) {
        CHECK_ARGUMENT(window.offsets[i] - window.offsets[i - 1] <= window_size_,
                       "immutable SINDI posting count exceeds window size");
    }

    const auto posting_count = static_cast<uint64_t>(window.offsets.back());
    read_vector(
        window.id_payloads, posting_count, "immutable SINDI id payload count exceeds offsets");
    CHECK_ARGUMENT(window.id_payloads.size() == posting_count,
                   "immutable SINDI id payload size does not match offsets");
    CHECK_ARGUMENT(posting_count <= std::numeric_limits<uint64_t>::max() /
                                        static_cast<uint64_t>(value_code_size),
                   "immutable SINDI value payload size overflows uint64_t");
    read_vector(window.value_payloads,
                posting_count * value_code_size,
                "immutable SINDI value payload count exceeds offsets");
    CHECK_ARGUMENT(window.value_payloads.size() == posting_count * value_code_size,
                   "immutable SINDI value payload size does not match offsets");
    for (const auto id : window.id_payloads) {
        CHECK_ARGUMENT(id < window_size_,
                       "immutable SINDI window-local doc id exceeds window size");
    }
}

std::pair<int64_t, int64_t>
SINDI::GetMinAndMaxId() const {
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
SINDI::EstimateMemory(uint64_t num_elements) const {
    uint64_t mem = 0;
    // size of label table
    mem += 2 * sizeof(int64_t) * num_elements;

    // size of term id + term data
    mem += avg_doc_term_length_ * num_elements *
           (sparse_value_code_size(sparse_value_quant_type_) + sizeof(uint16_t));

    if (use_reorder_) {
        mem += num_elements *
               (sizeof(uint32_t) + avg_doc_term_length_ * (sizeof(uint32_t) + sizeof(float)));

        const auto block_size = Options::Instance().block_size_limit();
        const auto offset_bytes = num_elements * (sizeof(uint64_t) + sizeof(uint32_t));
        mem += ((offset_bytes + block_size - 1) / block_size) * block_size;
    }

    // size of term list
    mem += sizeof(std::vector<float>) * 2 * term_id_limit_;

    // size of term id mapper (unordered_map ~50B per entry + vector 4B per entry)
    if (remap_term_ids_) {
        mem += static_cast<uint64_t>(term_id_limit_) * 54;
    }

    return mem;
}

void
SINDI::GetSparseVectorByInnerId(InnerIdType inner_id,
                                SparseVector* data,
                                Allocator* specified_allocator) const {
    std::shared_lock rlock(this->global_mutex_);
    CHECK_ARGUMENT(immutable_data_ == nullptr,
                   "immutable SINDI runtime does not support GetSparseVectorByInnerId");

    if (use_reorder_) {
        return this->rerank_flat_->GetSparseVectorByInnerId(inner_id, data, specified_allocator);
    }

    auto cur_window = inner_id / window_size_;
    auto window_start_id = cur_window * window_size_;
    auto term_list = this->window_term_list_[cur_window];

    term_list->GetSparseVector(inner_id - window_start_id, data, specified_allocator);

    // Reverse map compact IDs back to original term IDs
    if (remap_term_ids_ && term_id_mapper_) {
        for (uint32_t i = 0; i < data->len_; ++i) {
            data->ids_[i] = term_id_mapper_->ReverseMap(data->ids_[i]);
        }
    }
}

float
SINDI::CalcDistanceById(const DatasetPtr& vector,
                        int64_t id,
                        bool calculate_precise_distance) const {
    std::shared_lock rlock(this->global_mutex_);
    CHECK_ARGUMENT(immutable_data_ == nullptr,
                   "immutable SINDI runtime does not support CalcDistanceById");

    if (use_reorder_ && calculate_precise_distance) {
        auto [success, inner_id] = this->label_table_->TryGetIdByLabel(id);
        if (not success) {
            return -1.0F;
        }
        auto [sorted_ids, sorted_vals] =
            sort_sparse_vector(vector->GetSparseVectors()[0], allocator_);
        return calc_distance_by_id_unsafe(rerank_flat_, sorted_ids, sorted_vals, inner_id);
    }

    auto inner_id = this->label_table_->GetIdByLabel(id);
    auto cur_window = inner_id / window_size_;
    auto window_start_id = cur_window * window_size_;
    auto term_list = this->window_term_list_[cur_window];

    auto sparse_query = vector->GetSparseVectors()[0];
    Vector<uint32_t> tmp_ids(allocator_);
    Vector<float> tmp_vals(allocator_);
    if (remap_term_ids_) {
        sparse_query = remap_sparse_vector_for_query(sparse_query, tmp_ids, tmp_vals);
    }
    SINDISearchParameter search_param;
    search_param.query_prune_ratio = 0;
    search_param.term_prune_ratio = 0;
    auto computer = std::make_shared<SparseTermComputer>(sparse_query, search_param, allocator_);
    return term_list->CalcDistanceByInnerId(computer,
                                            static_cast<uint16_t>(inner_id - window_start_id));
}

DatasetPtr
SINDI::CalDistanceById(const DatasetPtr& query,
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
            distances[i] = success ? calc_distance_by_id_unsafe(
                                         rerank_flat_, sorted_ids, sorted_vals, inner_id)
                                   : -1;
        }
        return result;
    }

    // prepare result
    auto result = Dataset::Make();
    result->Owner(true, allocator_);
    auto* distances = static_cast<float*>(allocator_->Allocate(sizeof(float) * count));
    std::fill_n(distances, count, -1.0F);
    result->Distances(distances);

    // assume count is small, otherwise we should use bitmap to construct filter function
    std::unordered_map<int64_t, uint32_t> valid_ids;
    for (auto i = 0; i < count; i++) {
        valid_ids[ids[i]] = i;
    }
    auto filter = [&valid_ids](int64_t id) -> bool { return valid_ids.count(id) != 0; };
    auto filter_ptr = std::make_shared<WhiteListFilter>(filter);

    // search
    CHECK_ARGUMENT(immutable_data_ == nullptr,
                   "immutable SINDI runtime does not support CalDistanceById");
    constexpr auto* search_param_fmt = R"(
    {{
        "sindi": {{
            "query_prune_ratio": 0,
            "n_candidate": {}
        }}
    }}
    )";
    auto search_res =
        this->KnnSearch(query, count, fmt::format(search_param_fmt, count), filter_ptr);

    // flush results
    for (auto i = 0; i < search_res->GetDim(); i++) {
        float dist = search_res->GetDistances()[i];
        int64_t id = search_res->GetIds()[i];
        distances[valid_ids[id]] = dist;
    }

    return result;
}

void
SINDI::SetImmutable() {
    std::scoped_lock wlock(this->global_mutex_);
    this->immutable_.store(true, std::memory_order_release);
}

void
SINDI::InitFeatures() {
    // build & add
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_BUILD_WITH_MULTI_THREAD,
        IndexFeature::SUPPORT_ADD_AFTER_BUILD,
    });

    // search
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_KNN_SEARCH,
        IndexFeature::SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
        IndexFeature::SUPPORT_RANGE_SEARCH,
        IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER,
    });

    // serialize
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_DESERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_DESERIALIZE_FILE,
        IndexFeature::SUPPORT_DESERIALIZE_READER_SET,
        IndexFeature::SUPPORT_SERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_SERIALIZE_FILE,
    });

    // info
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID);
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_ESTIMATE_MEMORY);
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_GET_RAW_VECTOR_BY_IDS);

    // concurrency
    this->index_feature_list_->SetFeatures({IndexFeature::SUPPORT_SEARCH_CONCURRENT,
                                            IndexFeature::SUPPORT_ADD_CONCURRENT,
                                            IndexFeature::SUPPORT_UPDATE_ID_CONCURRENT,
                                            IndexFeature::SUPPORT_UPDATE_VECTOR_CONCURRENT});

    // metric
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_DELETE_BY_ID);
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_INNER_PRODUCT);
}

std::pair<int64_t, int64_t>
SINDI::get_min_max_window_id(const FilterPtr& filter) const {
    int64_t min_window_id = 0;
    auto num_windows =
        immutable_data_ != nullptr ? immutable_data_->windows.size() : window_term_list_.size();
    auto max_window_id = static_cast<int64_t>(num_windows) - 1;

    // get min and max window id
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
SINDI::remap_sparse_vector_for_query(const SparseVector& input,
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
SINDI::remap_sparse_vector_for_build(const SparseVector& input, Vector<uint32_t>& tmp_ids) {
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
uint32_t
SINDI::Remove(const std::vector<int64_t>& ids, RemoveMode mode) {
    if (mode != RemoveMode::MARK_REMOVE) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "SINDI only supports MARK_REMOVE");
    }
    std::scoped_lock lock(this->global_mutex_, this->label_lookup_mutex_);
    uint32_t delete_count = this->label_table_->MarkRemove(ids);
    delete_count_.fetch_add(delete_count, std::memory_order_relaxed);
    return delete_count;
}

}  // namespace vsag
