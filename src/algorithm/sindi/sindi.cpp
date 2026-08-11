
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

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "analyzer/analyzer.h"
#include "datacell/sparse_dmq_datacell.h"
#include "datacell/sparse_vector_datacell_parameter.h"
#include "impl/heap/standard_heap.h"
#include "impl/reasoning/search_reasoning.h"
#include "index_feature_list.h"
#include "io/memory_block_io/memory_block_io_parameter.h"
#include "quantization/sparse_quantization/sparse_quantizer_parameter.h"
#include "simd/fp16_simd.h"
#include "storage/serialization.h"
#include "storage/serialization_tags.h"
#include "storage/tlv_section.h"
#include "utils/search_threshold.h"
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
constexpr int64_t SINDI_RERANK_FLAT_FORMAT_DMQ = 3;
constexpr const char* SINDI_POSTING_LIST_FORMAT_VERSION_KEY = "sindi_posting_list_format_version";
constexpr int64_t SINDI_SORTED_POSTING_LIST_FORMAT_VERSION = 1;

bool
has_sorted_posting_lists(const JsonType& basic_info) {
    return basic_info.Contains(SINDI_POSTING_LIST_FORMAT_VERSION_KEY) and
           basic_info[SINDI_POSTING_LIST_FORMAT_VERSION_KEY].GetInt() >=
               SINDI_SORTED_POSTING_LIST_FORMAT_VERSION;
}

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

DistanceEvaluationBackend
sparse_backend(SparseValueQuantizationType quant_type) {
    switch (quant_type) {
        case SparseValueQuantizationType::FP16:
            return DistanceEvaluationBackend::SPARSE_FP16;
        case SparseValueQuantizationType::SQ8:
            return DistanceEvaluationBackend::SPARSE_SQ8;
        case SparseValueQuantizationType::FP32:
        default:
            return DistanceEvaluationBackend::SPARSE_FP32;
    }
}

uint32_t
get_bits_for_term_id_limit(uint32_t term_id_limit) {
    if (term_id_limit <= 1) {
        return 1;
    }

    uint32_t bits = 0;
    do {
        ++bits;
        term_id_limit >>= 1;
    } while (term_id_limit > 0);
    return bits;
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
create_rerank_flat(const IndexCommonParam& common_param,
                   const std::string& rerank_type,
                   uint32_t term_id_limit,
                   uint32_t dmq_shared_codebook_threshold) {
    if (rerank_type == SPARSE_RERANK_TYPE_DMQ8) {
        return std::make_shared<SparseDmqDataCell>(
            term_id_limit, common_param, dmq_shared_codebook_threshold);
    }
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
      rerank_type_(param->rerank_type),
      dmq_shared_codebook_threshold_(param->dmq_shared_codebook_threshold),
      term_id_limit_(param->term_id_limit),
      window_size_(param->window_size),
      doc_prune_ratio_(param->doc_prune_ratio),
      deserialize_without_footer_(param->deserialize_without_footer),
      deserialize_without_buffer_(param->deserialize_without_buffer),
      quantization_params_(std::make_shared<QuantizationParams>()),
      avg_doc_term_length_(param->avg_doc_term_length),
      remap_term_ids_(param->remap_term_ids),
      immutable_enabled_(param->immutable) {
    if (immutable_enabled_) {
        immutable_term_datacell_ =
            std::make_shared<ImmutableSindiTermDataCell>(term_id_limit_,
                                                         window_size_,
                                                         remap_term_ids_,
                                                         sparse_value_quant_type_,
                                                         quantization_params_,
                                                         allocator_);
        term_datacell_ = immutable_term_datacell_;
    } else {
        mutable_term_datacell_ =
            std::make_shared<MutableSindiTermDataCell>(term_id_limit_,
                                                       window_size_,
                                                       allocator_,
                                                       sparse_value_quant_type_,
                                                       quantization_params_);
        term_datacell_ = mutable_term_datacell_;
    }
    if (remap_term_ids_) {
        term_id_mapper_ =
            std::make_shared<TermIdMapper>(term_id_limit_, common_param.allocator_.get());
    }
    if (use_reorder_) {
        const uint32_t rerank_term_id_limit =
            remap_term_ids_ ? std::numeric_limits<uint32_t>::max() : term_id_limit_;
        rerank_flat_ = create_rerank_flat(
            common_param, rerank_type_, rerank_term_id_limit, param->dmq_shared_codebook_threshold);
    }
}

FlattenInterface*
SINDI::get_rerank_datacell() const {
    return rerank_flat_.get();
}

constexpr int64_t K_ANALYZE_DEFAULT_TOPK = 10;
constexpr uint64_t K_ANALYZE_BASE_SAMPLE_SIZE = 10;

std::unordered_map<std::string, uint64_t>
SINDI::GetMemoryUsageDetail() const {
    std::shared_lock rlock(this->global_mutex_);
    if (rerank_type_ != SPARSE_RERANK_TYPE_DMQ8 || rerank_flat_ == nullptr) {
        return {};
    }
    return {{"rerank_backend", rerank_flat_->GetMemoryUsage()}};
}

std::string
SINDI::GetStats() const {
    AnalyzerParam analyzer_param(allocator_);
    analyzer_param.topk = K_ANALYZE_DEFAULT_TOPK;
    analyzer_param.base_sample_size =
        std::min<uint64_t>(K_ANALYZE_BASE_SAMPLE_SIZE, cur_element_count_);
    analyzer_param.search_params = R"({"sindi": {"query_prune_ratio": 0, "n_candidate": 500}})";
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

SparseVector
SINDI::sort_and_prune_sparse_vector_for_build(const SparseVector& input,
                                              Vector<std::pair<uint32_t, float>>& sorted_terms,
                                              Vector<uint32_t>& pruned_ids,
                                              Vector<float>& pruned_vals) const {
    if (not remap_term_ids_) {
        for (uint32_t index = 0; index < input.len_; ++index) {
            CHECK_ARGUMENT(
                input.ids_[index] <= term_id_limit_,
                fmt::format("term id of sparse vector {} is greater than term id limit {}",
                            input.ids_[index],
                            term_id_limit_));
        }
    }

    if (doc_prune_ratio_ == 0.0F) {
        return input;
    }

    sorted_terms.clear();
    sorted_terms.reserve(input.len_);
    for (uint32_t index = 0; index < input.len_; ++index) {
        sorted_terms.emplace_back(input.ids_[index], input.vals_[index]);
    }
    std::sort(sorted_terms.begin(), sorted_terms.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.second != rhs.second) {
            return lhs.second > rhs.second;
        }
        return lhs.first < rhs.first;
    });

    auto retained_count = static_cast<uint32_t>(sorted_terms.size());
    if (sorted_terms.size() > 1) {
        float total_mass = 0.0F;
        for (const auto& [_, value] : sorted_terms) {
            total_mass += value;
        }
        const float retained_mass = total_mass * (1.0F - doc_prune_ratio_);
        float current_mass = 0.0F;
        retained_count = 0;
        while (current_mass < retained_mass && retained_count < sorted_terms.size()) {
            current_mass += sorted_terms[retained_count].second;
            ++retained_count;
        }
    }

    pruned_ids.resize(retained_count);
    pruned_vals.resize(retained_count);
    for (uint32_t index = 0; index < retained_count; ++index) {
        pruned_ids[index] = sorted_terms[index].first;
        pruned_vals[index] = sorted_terms[index].second;
    }
    return {retained_count, pruned_ids.data(), pruned_vals.data()};
}

void
SINDI::init_quantization_params_from_vectors(const DatasetPtr& base) {
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    bool has_value = false;
    const auto* sparse_vectors = base->GetSparseVectors();
    for (int64_t document = 0; document < base->GetNumElements(); ++document) {
        const auto& sparse_vector = sparse_vectors[document];
        for (uint32_t term = 0; term < sparse_vector.len_; ++term) {
            min_val = std::min(min_val, sparse_vector.vals_[term]);
            max_val = std::max(max_val, sparse_vector.vals_[term]);
            has_value = true;
        }
    }
    if (not has_value) {
        min_val = 0.0F;
        max_val = 0.0F;
    }
    quantization_params_->min_val = min_val;
    quantization_params_->max_val = max_val;
    quantization_params_->diff = max_val - min_val;
    if (quantization_params_->diff < 1e-6F) {
        quantization_params_->diff = 1.0F;
    }
}

std::vector<int64_t>
SINDI::Add(const DatasetPtr& base) {
    std::scoped_lock wlock(this->global_mutex_);
    CHECK_ARGUMENT(not immutable_enabled_, "immutable SINDI runtime does not support Add");
    if (rerank_type_ == SPARSE_RERANK_TYPE_DMQ8 and
        cur_element_count_.load(std::memory_order_relaxed) != 0) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "SINDI DMQ rerank does not support incremental Add");
    }
    const auto previous_window_count = mutable_term_datacell_->GetWindowCount();
    auto failed_ids = this->add(base, true);
    if (mutable_term_datacell_->GetWindowCount() != previous_window_count) {
        this->cal_memory_usage();
    }
    return failed_ids;
}

std::vector<int64_t>
SINDI::add(const DatasetPtr& base, bool sort_affected_windows) {
    CHECK_ARGUMENT(mutable_term_datacell_ != nullptr, "mutable SINDI data cell is not initialized");
    std::vector<int64_t> failed_ids;

    auto data_num = base->GetNumElements();
    CHECK_ARGUMENT(data_num > 0, "data_num is zero when add vectors");

    const auto* sparse_vectors = base->GetSparseVectors();
    const auto* ids = base->GetIds();
    const auto* extra_info = base->GetExtraInfos();
    const auto extra_info_size = base->GetExtraInfoSize();

    if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8 && cur_element_count_ == 0) {
        this->init_quantization_params_from_vectors(base);
    }

    // add process
    Vector<std::pair<uint32_t, float>> sorted_terms(allocator_);
    Vector<uint32_t> pruned_ids(allocator_);
    Vector<float> pruned_vals(allocator_);
    Vector<uint32_t> remapped_ids(allocator_);
    const auto first_affected_window = cur_element_count_ / window_size_;
    int64_t last_affected_window = -1;
    std::vector<SparseVector> rerank_vectors;
    if (use_reorder_) {
        rerank_vectors.reserve(data_num);
    }
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
            const auto pruned = this->sort_and_prune_sparse_vector_for_build(
                sparse_vector, sorted_terms, pruned_ids, pruned_vals);
            if (remap_term_ids_) {
                auto remapped = remap_sparse_vector_for_build(pruned, remapped_ids);
                mutable_term_datacell_->InsertVector(remapped, cur_element_count_);
            } else {
                mutable_term_datacell_->InsertVector(pruned, cur_element_count_);
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
            logger::warn("memory allocation failed: {}", e.what());
            throw;
        }

        label_table_->Insert(cur_element_count_, ids[i]);  // todo(zxy): check id exists

        if (extra_info_size > 0) {
            extra_infos_->InsertExtraInfo(extra_info + i * extra_info_size, cur_element_count_);
        }

        if (use_reorder_) {
            rerank_vectors.push_back(sparse_vectors[i]);
        }
        last_affected_window = cur_element_count_ / window_size_;
        cur_element_count_++;
    }
    if (not rerank_vectors.empty()) {
        this->get_rerank_datacell()->BatchInsertVector(
            rerank_vectors.data(), static_cast<InnerIdType>(rerank_vectors.size()));
    }
    if (sort_affected_windows) {
        for (int64_t window = first_affected_window; window <= last_affected_window; ++window) {
            mutable_term_datacell_->SortByValue(static_cast<uint32_t>(window));
        }
    }
    return failed_ids;
}

std::vector<int64_t>
SINDI::Build(const DatasetPtr& base) {
    if (immutable_enabled_) {
        return this->build_immutable(base);
    }
    std::scoped_lock wlock(this->global_mutex_);
    CHECK_ARGUMENT(base->GetNumElements() > 0, "data_num is zero when add vectors");
    auto failed_ids = this->add(base, false);
    mutable_term_datacell_->Finalize();
    this->cal_memory_usage();
    return failed_ids;
}

std::vector<int64_t>
SINDI::build_immutable(const DatasetPtr& base) {
    std::scoped_lock wlock(this->global_mutex_);
    CHECK_ARGUMENT(immutable_enabled_, "mutable SINDI cannot use immutable build");
    CHECK_ARGUMENT(not immutable_build_started_, "immutable SINDI has already been built");

    const auto data_num = base->GetNumElements();
    CHECK_ARGUMENT(data_num > 0, "data_num is zero when build immutable SINDI");
    const auto* sparse_vectors = base->GetSparseVectors();
    const auto* ids = base->GetIds();
    const auto* extra_info = base->GetExtraInfos();
    const auto extra_info_size = base->GetExtraInfoSize();

    if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
        this->init_quantization_params_from_vectors(base);
    }

    CHECK_ARGUMENT(immutable_term_datacell_ != nullptr,
                   "immutable SINDI data cell is not initialized");
    immutable_build_started_ = true;
    immutable_term_datacell_->Reserve(align_up(data_num, window_size_) / window_size_);

    std::vector<int64_t> failed_ids;
    Vector<std::pair<uint32_t, float>> sorted_terms(allocator_);
    Vector<uint32_t> pruned_ids(allocator_);
    Vector<float> pruned_vals(allocator_);
    Vector<uint32_t> remapped_ids(allocator_);
    std::vector<SparseVector> rerank_vectors;
    auto mutable_term_datacell = std::make_shared<MutableSindiTermDataCell>(
        term_id_limit_, window_size_, allocator_, sparse_value_quant_type_, quantization_params_);
    auto flush_staging_window = [&]() {
        mutable_term_datacell->SortByValue(0);
        immutable_term_datacell_->AppendWindow(mutable_term_datacell->GetWindow(0));
        mutable_term_datacell = std::make_shared<MutableSindiTermDataCell>(term_id_limit_,
                                                                           window_size_,
                                                                           allocator_,
                                                                           sparse_value_quant_type_,
                                                                           quantization_params_);
    };
    if (use_reorder_) {
        rerank_vectors.reserve(data_num);
    }
    for (int64_t i = 0; i < data_num; ++i) {
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
            const auto local_id = static_cast<uint32_t>(cur_element_count_ % window_size_);
            const auto pruned = this->sort_and_prune_sparse_vector_for_build(
                sparse_vector, sorted_terms, pruned_ids, pruned_vals);
            if (remap_term_ids_) {
                const auto remapped = remap_sparse_vector_for_build(pruned, remapped_ids);
                mutable_term_datacell->InsertVector(remapped, local_id);
            } else {
                mutable_term_datacell->InsertVector(pruned, local_id);
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
            logger::warn("memory allocation failed: {}", e.what());
            throw;
        }

        label_table_->Insert(cur_element_count_, ids[i]);
        if (extra_info_size > 0) {
            extra_infos_->InsertExtraInfo(extra_info + i * extra_info_size, cur_element_count_);
        }
        if (use_reorder_) {
            rerank_vectors.push_back(sparse_vectors[i]);
        }
        ++cur_element_count_;

        if (cur_element_count_ % window_size_ == 0) {
            flush_staging_window();
        }
    }
    if (mutable_term_datacell->total_count_ > 0) {
        flush_staging_window();
    }
    if (not rerank_vectors.empty()) {
        this->get_rerank_datacell()->BatchInsertVector(
            rerank_vectors.data(), static_cast<InnerIdType>(rerank_vectors.size()));
    }
    this->cal_memory_usage();
    return failed_ids;
}

bool
SINDI::UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update) {
    // Note:
    // 1. we only check whether the old vector is a subset of the new vector
    // 2. we do not actually update the vector
    if (rerank_type_ == SPARSE_RERANK_TYPE_DMQ8) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "SINDI DMQ rerank does not support UpdateVector");
    }
    CHECK_ARGUMENT(new_base != nullptr, "new base dataset is nullptr");
    CHECK_ARGUMENT(new_base->GetNumElements() == 1, "num of new base should be 1");
    CHECK_ARGUMENT(new_base->GetSparseVectors() != nullptr, "new base sparse vectors is nullptr");
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
                    this->get_rerank_datacell()->GetSparseVectorByInnerId(
                        inner_id, data, allocator);
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
    const auto threshold = ParseSearchThreshold(parameters);
    CHECK_ARGUMENT(search_param.n_candidate <= SPARSE_AMPLIFICATION_FACTOR * k,
                   fmt::format("n_candidate ({}) should be less than {} * k ({})",
                               search_param.n_candidate,
                               AMPLIFICATION_FACTOR,
                               k));
    InnerSearchParam inner_param;
    inner_param.ef = std::max(static_cast<int64_t>(search_param.n_candidate), k);
    inner_param.topk = threshold.has_value() ? static_cast<int64_t>(inner_param.ef) : k;
    inner_param.distance_threshold = threshold;
    inner_param.enable_reorder = use_reorder_;

    inner_param.is_inner_id_allowed = this->create_search_filter(filter);

    SearchStatistics statistics;
    SparseVector effective_query = sparse_query;
    Vector<uint32_t> tmp_ids(allocator);
    Vector<float> tmp_vals(allocator);
    if (remap_term_ids_) {
        effective_query = remap_sparse_vector_for_query(sparse_query, tmp_ids, tmp_vals);
        if (effective_query.len_ == 0) {
            auto result = make_empty_result();
            result->Statistics(statistics.Dump());
            return result;
        }
    }

    auto computer = std::make_shared<SparseTermComputer>(
        effective_query, search_param, allocator_, term_datacell_->GetWindowCount());
    const SparseVector* rerank_query = (remap_term_ids_ && use_reorder_) ? &sparse_query : nullptr;
    auto result = search_impl<KNN_SEARCH>(computer,
                                          inner_param,
                                          allocator,
                                          UseTermListsHeapInsert(search_param, threshold),
                                          rerank_query,
                                          nullptr,
                                          &statistics);
    result->Statistics(statistics.Dump());
    return FilterDatasetByThreshold(result, threshold, allocator, k);
}

template <InnerSearchMode mode>
DatasetPtr
SINDI::search_impl(const SparseTermComputerPtr& computer,
                   const InnerSearchParam& inner_param,
                   Allocator* allocator,
                   bool use_term_lists_heap_insert,
                   const SparseVector* original_query,
                   ReasoningContext* reasoning_ctx,
                   SearchStatistics* statistics) const {
    auto* search_allocator = allocator != nullptr ? allocator : allocator_;
    // computer and heap
    MaxHeap heap(search_allocator);
    int64_t k = 0;

    if constexpr (mode == KNN_SEARCH) {
        k = inner_param.topk;
    }

    // window iteration
    Vector<float> dists(window_size_, 0.0F, search_allocator);
    auto filter = inner_param.is_inner_id_allowed;
    const auto [min_window_id, max_window_id] = this->get_min_max_window_id(filter);
    auto selected_buckets = reasoning_ctx != nullptr
                                ? std::make_unique<Vector<BucketIdType>>(search_allocator)
                                : nullptr;
    SindiQueryContext query_context(search_allocator);
    for (auto cur = min_window_id; cur <= max_window_id; cur++) {
        const auto window_start_id = static_cast<uint32_t>(cur) * window_size_;
        // compute
        term_datacell_->QueryWindow(dists.data(),
                                    static_cast<uint32_t>(cur),
                                    computer,
                                    use_term_lists_heap_insert,
                                    query_context);
        if (statistics != nullptr) {
            statistics->AddDistance(SearchStatistics::DistancePhase::APPROXIMATE,
                                    sparse_backend(sparse_value_quant_type_),
                                    query_context.evaluation_tracker.Count());
        }

        if (reasoning_ctx != nullptr) {
            selected_buckets->push_back(static_cast<BucketIdType>(cur));
            auto doc_count = static_cast<uint32_t>(std::min<int64_t>(
                window_size_, cur_element_count_ - static_cast<int64_t>(window_start_id)));
            for (uint32_t i = 0; i < doc_count; ++i) {
                if (dists[i] != 0.0F) {
                    auto inner_id = window_start_id + i;
                    reasoning_ctx->RecordVisit(inner_id, 1.0F + dists[i], 0);
                    if (filter and not filter->CheckValid(inner_id)) {
                        reasoning_ctx->RecordFilterReject(inner_id);
                    }
                }
            }
        }

        // insert heap
        if (use_term_lists_heap_insert) {
            term_datacell_->InsertHeapByWindow(dists.data(),
                                               static_cast<uint32_t>(cur),
                                               computer,
                                               heap,
                                               inner_param,
                                               window_start_id,
                                               mode,
                                               inner_param.is_inner_id_allowed != nullptr,
                                               query_context);
        } else {
            const auto remaining_count =
                static_cast<uint64_t>(cur_element_count_.load()) - window_start_id;
            const auto window_document_count =
                static_cast<uint32_t>(std::min<uint64_t>(window_size_, remaining_count));
            term_datacell_->InsertHeapByDists(dists.data(),
                                              window_document_count,
                                              heap,
                                              inner_param,
                                              window_start_id,
                                              mode,
                                              inner_param.is_inner_id_allowed != nullptr);
        }
    }

    if (selected_buckets != nullptr and not selected_buckets->empty()) {
        reasoning_ctx->RecordBucketSelection(*selected_buckets);
    }

    // rerank
    if (use_reorder_) {
        // high precision
        float cur_heap_top = std::numeric_limits<float>::max();
        auto candidate_size = heap.size();
        auto high_precise_heap = std::make_shared<StandardHeap<true, false>>(search_allocator, -1);
        const auto& rerank_query = original_query ? *original_query : computer->raw_query_;
        auto* rerank_datacell = this->get_rerank_datacell();
        auto rerank_computer = rerank_datacell->FactoryComputer(&rerank_query);
        for (auto i = 0; i < candidate_size; i++) {
            auto inner_id = heap.top().second;
            float high_precise_distance = 0.0F;
            QueryContext query_context{.stats = statistics,
                                       .distance_phase = DistanceEvaluationPhase::RERANK};
            rerank_datacell->Query(
                &high_precise_distance, rerank_computer, &inner_id, 1, &query_context);
            auto label = label_table_->GetLabelById(inner_id);
            if (reasoning_ctx != nullptr) {
                reasoning_ctx->RecordReorder(
                    inner_id, 1.0F + heap.top().first, high_precise_distance);
            }
            if constexpr (mode == KNN_SEARCH) {
                const bool eligible =
                    not inner_param.distance_threshold.has_value() or
                    (std::isfinite(high_precise_distance) and
                     high_precise_distance <= inner_param.distance_threshold.value());
                if (eligible and
                    (high_precise_distance < cur_heap_top or high_precise_heap->Size() < k)) {
                    high_precise_heap->Push(high_precise_distance, label);
                    if (high_precise_heap->Size() > k) {
                        if (reasoning_ctx != nullptr) {
                            auto evicted_label = high_precise_heap->Top().second;
                            auto evicted_inner_id = this->label_table_->GetIdByLabel(evicted_label);
                            reasoning_ctx->RecordReorderEviction(evicted_inner_id, 0);
                        }
                        high_precise_heap->Pop();
                    }
                    cur_heap_top = high_precise_heap->Top().first;
                }
            }
            if constexpr (mode == RANGE_SEARCH) {
                if (high_precise_distance <= inner_param.radius) {
                    high_precise_heap->Push(high_precise_distance, label);
                }
                if (inner_param.range_search_limit_size != -1 and
                    high_precise_heap->Size() > inner_param.range_search_limit_size) {
                    if (reasoning_ctx != nullptr) {
                        auto evicted_label = high_precise_heap->Top().second;
                        auto evicted_inner_id = this->label_table_->GetIdByLabel(evicted_label);
                        reasoning_ctx->RecordReorderEviction(evicted_inner_id, 0);
                    }
                    high_precise_heap->Pop();
                }
            }
            heap.pop();
        }

        return collect_heap_results(high_precise_heap, search_allocator);
    }

    // low precision
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

    SearchStatistics statistics;
    SparseVector effective_query = sparse_query;
    Vector<uint32_t> tmp_ids(allocator_);
    Vector<float> tmp_vals(allocator_);
    if (remap_term_ids_) {
        effective_query = remap_sparse_vector_for_query(sparse_query, tmp_ids, tmp_vals);
        if (effective_query.len_ == 0) {
            auto result = make_empty_result();
            result->Statistics(statistics.Dump());
            return result;
        }
    }

    auto computer = std::make_shared<SparseTermComputer>(
        effective_query, search_param, allocator_, term_datacell_->GetWindowCount());
    const SparseVector* rerank_query = (remap_term_ids_ && use_reorder_) ? &sparse_query : nullptr;
    auto result = search_impl<RANGE_SEARCH>(computer,
                                            inner_param,
                                            allocator_,
                                            UseTermListsHeapInsert(search_param),
                                            rerank_query,
                                            nullptr,
                                            &statistics);
    result->Statistics(statistics.Dump());
    return result;
}

DatasetPtr
SINDI::SearchWithRequest(const SearchRequest& request) const {
    std::shared_lock rlock(this->global_mutex_);

    CHECK_ARGUMENT(request.query_ != nullptr, "query should not be null");
    const auto* sparse_vectors = request.query_->GetSparseVectors();
    CHECK_ARGUMENT(request.query_->GetNumElements() == 1, "num of query should be 1");
    auto sparse_query = sparse_vectors[0];
    CHECK_ARGUMENT(
        sparse_query.len_ > 0,
        fmt::format("query->GetSparseVectors()->len_ ({}) is invalid", sparse_query.len_));

    SINDISearchParameter search_param;
    search_param.FromJson(JsonType::Parse(request.params_str_));

    Allocator* allocator = select_query_allocator(request.search_allocator_, this->allocator_);

    bool is_range = (request.mode_ == SearchMode::RANGE_SEARCH);
    SearchStatistics statistics;

    InnerSearchParam inner_param;
    const bool filter_enabled = request.enable_filter_ and request.filter_ != nullptr;
    inner_param.is_inner_id_allowed =
        filter_enabled ? this->create_search_filter(request.filter_) : nullptr;

    std::shared_ptr<ReasoningContext> reasoning_ctx;
    if (not request.expected_labels_.empty()) {
        reasoning_ctx = std::make_shared<ReasoningContext>(this->allocator_);
        reasoning_ctx->SetSearchParams(request.topk_, "SINDI", use_reorder_, filter_enabled);

        UnorderedMap<int64_t, InnerIdType> label_to_inner_id(this->allocator_);
        {
            std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
            for (const auto& label : request.expected_labels_) {
                auto [success, inner_id] = this->label_table_->TryGetIdByLabel(label, true);
                if (success) {
                    label_to_inner_id[label] = inner_id;
                }
            }
        }

        Vector<int64_t> expected_labels_vec(this->allocator_);
        expected_labels_vec.reserve(request.expected_labels_.size());
        for (const auto& label : request.expected_labels_) {
            expected_labels_vec.push_back(label);
        }
        reasoning_ctx->InitializeExpectedTargets(expected_labels_vec, label_to_inner_id);
    }

    SparseVector effective_query = sparse_query;
    Vector<uint32_t> tmp_ids(allocator);
    Vector<float> tmp_vals(allocator);
    if (remap_term_ids_) {
        effective_query = remap_sparse_vector_for_query(sparse_query, tmp_ids, tmp_vals);
        if (effective_query.len_ == 0) {
            auto result = make_empty_result();
            result->Statistics(statistics.Dump());
            this->AttachReasoningReport(result, reasoning_ctx.get());
            return result;
        }
    }

    auto computer = std::make_shared<SparseTermComputer>(
        effective_query, search_param, allocator, term_datacell_->GetWindowCount());
    const SparseVector* rerank_query = (remap_term_ids_ && use_reorder_) ? &sparse_query : nullptr;

    DatasetPtr result;
    if (is_range) {
        inner_param.range_search_limit_size = static_cast<int>(request.limited_size_);
        inner_param.radius = request.radius_;
        result = search_impl<RANGE_SEARCH>(computer,
                                           inner_param,
                                           allocator,
                                           UseTermListsHeapInsert(search_param),
                                           rerank_query,
                                           reasoning_ctx.get(),
                                           &statistics);
    } else {
        CHECK_ARGUMENT(search_param.n_candidate <= SPARSE_AMPLIFICATION_FACTOR * request.topk_,
                       fmt::format("n_candidate ({}) should be less than {} * k ({})",
                                   search_param.n_candidate,
                                   SPARSE_AMPLIFICATION_FACTOR,
                                   request.topk_));
        inner_param.ef = std::max(static_cast<int64_t>(search_param.n_candidate), request.topk_);
        inner_param.topk = request.topk_;
        result = search_impl<KNN_SEARCH>(computer,
                                         inner_param,
                                         allocator,
                                         UseTermListsHeapInsert(search_param),
                                         rerank_query,
                                         reasoning_ctx.get(),
                                         &statistics);
    }

    result->Statistics(statistics.Dump());
    this->AttachReasoningReport(result, reasoning_ctx.get());
    return result;
}

void
SINDI::AttachReasoningReport(const DatasetPtr& dataset_results,
                             ReasoningContext* reasoning_ctx) const {
    if (reasoning_ctx == nullptr or dataset_results == nullptr) {
        return;
    }
    auto count = dataset_results->GetNumElements();
    if (count > 0 and dataset_results->GetIds() != nullptr) {
        Vector<InnerIdType> result_inner_ids(static_cast<uint64_t>(count), this->allocator_);
        {
            std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
            for (int64_t i = 0; i < count; ++i) {
                result_inner_ids[i] =
                    this->label_table_->GetIdByLabel(dataset_results->GetIds()[i]);
            }
        }
        reasoning_ctx->MarkResult(result_inner_ids);
    }
    reasoning_ctx->DiagnoseExpectedTargets();
    dataset_results->Reasoning(reasoning_ctx->GenerateReport());
}

bool
SINDI::UseTermListsHeapInsert(const SINDISearchParameter& search_param,
                              const std::optional<float>& distance_threshold) const {
    // Low build-time doc pruning and low search-time query pruning keep the old
    // distance-array heap insertion path for accuracy. Once either side exceeds the
    // threshold, term-list heap insertion avoids scanning the whole window for heap updates.
    const bool term_lists_enabled =
        doc_prune_ratio_ > SINDI::K_TERM_LISTS_HEAP_INSERT_PRUNE_THRESHOLD ||
        search_param.query_prune_ratio > SINDI::K_TERM_LISTS_HEAP_INSERT_PRUNE_THRESHOLD;
    if (not term_lists_enabled or not distance_threshold.has_value()) {
        return term_lists_enabled;
    }
    // Posting lists omit zero-overlap documents, whose non-reorder distance is exactly 1.
    return not use_reorder_ and distance_threshold.value() < 1.0F;
}

void
SINDI::cal_memory_usage() {
    auto memory = sizeof(SINDI);
    if (term_datacell_ != nullptr) {
        memory += term_datacell_->GetMemoryUsage();
    }
    memory += label_table_->GetMemoryUsage();
    if (const auto* rerank_datacell = this->get_rerank_datacell(); rerank_datacell != nullptr) {
        memory += rerank_datacell->GetMemoryUsage();
    }
    memory += sizeof(QuantizationParams);
    if (remap_term_ids_ && term_id_mapper_) {
        memory +=
            static_cast<uint64_t>(term_id_mapper_->Size()) * TERM_ID_MAPPER_ENTRY_MEMORY_BYTES;
    }

    std::unique_lock lock(this->memory_usage_mutex_);
    this->current_memory_usage_.store(static_cast<int64_t>(memory));
}

void
SINDI::Serialize(StreamWriter& writer) const {
    std::shared_lock rlock(this->global_mutex_);

    if (cur_element_count_ == 0) {
        const auto cur_element_count = cur_element_count_.load();
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

    const auto cur_element_count = cur_element_count_.load();
    StreamWriter::WriteObj(writer, cur_element_count);

    if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
        StreamWriter::WriteObj(writer, quantization_params_->min_val);
        StreamWriter::WriteObj(writer, quantization_params_->max_val);
        StreamWriter::WriteObj(writer, quantization_params_->diff);
    }

    uint32_t window_term_list_size = term_datacell_->GetWindowCount();
    StreamWriter::WriteObj(writer, window_term_list_size);
    if (immutable_term_datacell_ != nullptr) {
        immutable_term_datacell_->SerializeWindows(writer);
    } else {
        mutable_term_datacell_->SerializeWindows(writer);
    }

    label_table_->Serialize(writer);

    if (use_reorder_) {
        this->get_rerank_datacell()->Serialize(writer);
    }

    if (remap_term_ids_ && term_id_mapper_) {
        term_id_mapper_->Serialize(writer);
    }

    JsonType jsonify_basic_info;
    jsonify_basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());
    jsonify_basic_info[SINDI_POSTING_LIST_FORMAT_VERSION_KEY].SetInt(
        SINDI_SORTED_POSTING_LIST_FORMAT_VERSION);
    if (use_reorder_ && rerank_type_ == SPARSE_RERANK_TYPE_DMQ8) {
        jsonify_basic_info[SINDI_RERANK_FLAT_FORMAT_KEY].SetInt(SINDI_RERANK_FLAT_FORMAT_DMQ);
    } else if (use_reorder_) {
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
    basic_info[SINDI_POSTING_LIST_FORMAT_VERSION_KEY].SetInt(
        SINDI_SORTED_POSTING_LIST_FORMAT_VERSION);
    if (use_reorder_) {
        const auto rerank_format = rerank_type_ == SPARSE_RERANK_TYPE_DMQ8
                                       ? SINDI_RERANK_FLAT_FORMAT_DMQ
                                       : SINDI_RERANK_FLAT_FORMAT_DATACELL;
        basic_info[SINDI_RERANK_FLAT_FORMAT_KEY].SetInt(rerank_format);
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

    const uint32_t window_term_list_size =
        term_datacell_ == nullptr ? 0 : term_datacell_->GetWindowCount();
    StreamWriter::WriteObj(writer, window_term_list_size);
    if (immutable_term_datacell_ != nullptr) {
        immutable_term_datacell_->SerializeWindows(writer);
    } else if (mutable_term_datacell_ != nullptr) {
        mutable_term_datacell_->SerializeWindows(writer);
    }
}

void
SINDI::serialize_streaming_body(StreamWriter& writer) const {
    std::shared_lock rlock(this->global_mutex_);
    CHECK_ARGUMENT(not immutable_enabled_,
                   "immutable SINDI runtime does not support SerializeStreaming");

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
                this->get_rerank_datacell()->Serialize(w);
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
SINDI::deserialize_windows(StreamReader& reader_ref, bool postings_sorted) {
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
        auto immutable = std::make_shared<ImmutableSindiTermDataCell>(term_id_limit_,
                                                                      window_size_,
                                                                      remap_term_ids_,
                                                                      sparse_value_quant_type_,
                                                                      quantization_params_,
                                                                      allocator_);
        immutable->DeserializeWindows(reader_ref, window_term_list_size, postings_sorted);
        immutable_term_datacell_ = std::move(immutable);
        mutable_term_datacell_.reset();
        term_datacell_ = immutable_term_datacell_;
        immutable_build_started_ = true;
    } else {
        auto mutable_datacell = std::make_shared<MutableSindiTermDataCell>(term_id_limit_,
                                                                           window_size_,
                                                                           allocator_,
                                                                           sparse_value_quant_type_,
                                                                           quantization_params_);
        mutable_datacell->DeserializeWindows(reader_ref, window_term_list_size, postings_sorted);
        mutable_term_datacell_ = std::move(mutable_datacell);
        immutable_term_datacell_.reset();
        term_datacell_ = mutable_term_datacell_;
    }
    this->trim_deserialized_trailing_windows();
}

void
SINDI::trim_deserialized_trailing_windows() {
    const auto element_count = static_cast<uint64_t>(cur_element_count_.load());
    const auto populated_window_count =
        element_count / window_size_ + static_cast<uint64_t>(element_count % window_size_ != 0);
    const auto serialized_window_count =
        term_datacell_ == nullptr ? 0 : term_datacell_->GetWindowCount();
    CHECK_ARGUMENT(serialized_window_count >= populated_window_count,
                   "serialized SINDI has fewer windows than its element count requires");
    if (immutable_term_datacell_ != nullptr) {
        immutable_term_datacell_->ResizeWindowCount(static_cast<uint32_t>(populated_window_count));
    } else if (mutable_term_datacell_ != nullptr) {
        mutable_term_datacell_->ResizeWindowCount(static_cast<uint32_t>(populated_window_count));
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
    const auto postings_sorted = has_sorted_posting_lists(basic_info);
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

    bool has_dmq_rerank_format = false;
    if (this->use_reorder_) {
        auto rerank_format = SINDI_RERANK_FLAT_FORMAT_DATACELL;
        if (basic_info.Contains(SINDI_RERANK_FLAT_FORMAT_KEY)) {
            rerank_format = basic_info[SINDI_RERANK_FLAT_FORMAT_KEY].GetInt();
        }
        has_dmq_rerank_format = rerank_format == SINDI_RERANK_FLAT_FORMAT_DMQ;
        const bool expects_dmq = this->rerank_type_ == SPARSE_RERANK_TYPE_DMQ8;
        CHECK_ARGUMENT(has_dmq_rerank_format == expects_dmq,
                       "SINDI streaming rerank format does not match rerank_type");
    }

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
                ReadSeekableBlockPayload(
                    block_reader, block_header, [this, postings_sorted](StreamReader& block) {
                        this->deserialize_windows(block, postings_sorted);
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
                        block_reader,
                        block_header,
                        [this, has_dmq_rerank_format](StreamReader& block) {
                            if (has_dmq_rerank_format) {
                                this->rerank_flat_->Deserialize(block);
                            } else {
                                deserialize_rerank_flat(
                                    block, this->rerank_flat_, this->allocator_, true);
                            }
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
    bool has_dmq_rerank_format = false;
    bool postings_sorted = false;
    if (not deserialize_without_footer_) {
        JsonType jsonify_basic_info;
        try {
            if (read_index_footer(reader, jsonify_basic_info)) {
                has_footer = true;
                postings_sorted = has_sorted_posting_lists(jsonify_basic_info);
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
                }
                if (jsonify_basic_info.Contains(SINDI_RERANK_FLAT_FORMAT_KEY)) {
                    const auto rerank_format =
                        jsonify_basic_info[SINDI_RERANK_FLAT_FORMAT_KEY].GetInt();
                    has_datacell_rerank_format = rerank_format == SINDI_RERANK_FLAT_FORMAT_DATACELL;
                    has_dmq_rerank_format = rerank_format == SINDI_RERANK_FLAT_FORMAT_DMQ;
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
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        has_footer || not immutable_enabled_,
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
        auto immutable = std::make_shared<ImmutableSindiTermDataCell>(term_id_limit_,
                                                                      window_size_,
                                                                      remap_term_ids_,
                                                                      sparse_value_quant_type_,
                                                                      quantization_params_,
                                                                      allocator_);
        immutable->DeserializeWindows(reader_ref, window_term_list_size, postings_sorted);
        immutable_term_datacell_ = std::move(immutable);
        mutable_term_datacell_.reset();
        term_datacell_ = immutable_term_datacell_;
        immutable_build_started_ = true;
    } else {
        auto mutable_datacell = std::make_shared<MutableSindiTermDataCell>(term_id_limit_,
                                                                           window_size_,
                                                                           allocator_,
                                                                           sparse_value_quant_type_,
                                                                           quantization_params_);
        mutable_datacell->DeserializeWindows(reader_ref, window_term_list_size, postings_sorted);
        mutable_term_datacell_ = std::move(mutable_datacell);
        immutable_term_datacell_.reset();
        term_datacell_ = mutable_term_datacell_;
    }
    this->trim_deserialized_trailing_windows();

    label_table_->Deserialize(reader_ref);
    delete_count_.store(static_cast<int64_t>(label_table_->GetAllDeletedIds().size()),
                        std::memory_order_relaxed);

    if (cur_element_count_ == 0) {
        this->cal_memory_usage();
        return;
    }

    if (use_reorder_ && rerank_type_ == SPARSE_RERANK_TYPE_DMQ8) {
        CHECK_ARGUMENT(has_dmq_rerank_format || deserialize_without_footer_,
                       "serialized SINDI rerank payload is not DMQ format");
        rerank_flat_->Deserialize(reader_ref);
    } else if (use_reorder_) {
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
SINDI::serialize_immutable_window(StreamWriter& writer, const ImmutableSINDIWindow& window) const {
    ImmutableSindiTermDataCell adapter(term_id_limit_,
                                       window_size_,
                                       remap_term_ids_,
                                       sparse_value_quant_type_,
                                       quantization_params_,
                                       allocator_);
    ImmutableSindiTermDataCell::SerializeWindow(writer, window);
}

void
SINDI::deserialize_immutable_window(StreamReader& reader_ref,
                                    ImmutableSINDIWindow& window,
                                    bool postings_sorted) const {
    ImmutableSindiTermDataCell adapter(term_id_limit_,
                                       window_size_,
                                       remap_term_ids_,
                                       sparse_value_quant_type_,
                                       quantization_params_,
                                       allocator_);
    adapter.DeserializeWindow(reader_ref, window, postings_sorted);
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
        uint64_t total_sparse_values = static_cast<uint64_t>(avg_doc_term_length_) * num_elements;
        if (rerank_type_ == SPARSE_RERANK_TYPE_DMQ8) {
            const uint64_t estimated_term_count =
                std::min<uint64_t>(term_id_limit_, total_sparse_values);
            const uint32_t rerank_id_bits = get_bits_for_term_id_limit(
                estimated_term_count == 0 ? 0 : static_cast<uint32_t>(estimated_term_count - 1));
            mem += (total_sparse_values * rerank_id_bits + 7) / 8;
            mem += total_sparse_values;
            mem += num_elements * (sizeof(uint64_t) + sizeof(SparseDmqQuantizer::EncodedHeader));
            uint64_t estimated_codebook_count = estimated_term_count;
            if (dmq_shared_codebook_threshold_ != 0 && total_sparse_values != 0) {
                const uint64_t minimum_dedicated_term_frequency =
                    static_cast<uint64_t>(dmq_shared_codebook_threshold_) + 1;
                estimated_codebook_count =
                    std::min(estimated_term_count,
                             total_sparse_values / minimum_dedicated_term_frequency + 1);
            }
            mem += estimated_codebook_count * sizeof(SparseDmqQuantizer::Codebook);
            mem += estimated_term_count * 2 * sizeof(uint32_t);
        } else {
            mem += num_elements *
                   (sizeof(uint32_t) + avg_doc_term_length_ * (sizeof(uint32_t) + sizeof(float)));

            const auto block_size = Options::Instance().block_size_limit();
            const auto offset_bytes = num_elements * (sizeof(uint64_t) + sizeof(uint32_t));
            mem += ((offset_bytes + block_size - 1) / block_size) * block_size;
        }
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
    CHECK_ARGUMENT(immutable_term_datacell_ == nullptr,
                   "immutable SINDI runtime does not support GetSparseVectorByInnerId");

    if (use_reorder_) {
        return this->get_rerank_datacell()->GetSparseVectorByInnerId(
            inner_id, data, specified_allocator);
    }

    term_datacell_->GetSparseVector(inner_id, data, specified_allocator);

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
    CHECK_ARGUMENT(immutable_term_datacell_ == nullptr,
                   "immutable SINDI runtime does not support CalcDistanceById");

    if (vector == nullptr || vector->GetNumElements() == 0 ||
        vector->GetSparseVectors() == nullptr) {
        return -1.0F;
    }

    if (use_reorder_ && calculate_precise_distance) {
        const auto [success, inner_id] = this->label_table_->TryGetIdByLabel(id);
        if (not success) {
            return -1.0F;
        }
        auto* rerank_datacell = this->get_rerank_datacell();
        auto computer = rerank_datacell->FactoryComputer(vector->GetSparseVectors());
        float distance = 0.0F;
        rerank_datacell->Query(&distance, computer, &inner_id, 1);
        return distance;
    }

    const auto inner_id = this->label_table_->GetIdByLabel(id);
    auto sparse_query = vector->GetSparseVectors()[0];
    Vector<uint32_t> tmp_ids(allocator_);
    Vector<float> tmp_vals(allocator_);
    if (remap_term_ids_) {
        sparse_query = remap_sparse_vector_for_query(sparse_query, tmp_ids, tmp_vals);
    }
    SINDISearchParameter search_param;
    search_param.query_prune_ratio = 0;
    auto computer = std::make_shared<SparseTermComputer>(sparse_query, search_param, allocator_);
    const QueryTermBuffers query_term_buffers(allocator_);
    return term_datacell_->CalcDistanceByInnerId(computer, inner_id, query_term_buffers);
}

DatasetPtr
SINDI::CalcDistancesById(const DatasetPtr& query,
                         const int64_t* ids,
                         int64_t count,
                         bool calculate_precise_distance) const {
    return this->CalDistanceById(query, ids, count, calculate_precise_distance);
}

DatasetPtr
SINDI::CalDistanceById(const DatasetPtr& query,
                       const int64_t* ids,
                       int64_t count,
                       bool calculate_precise_distance,
                       int64_t topk) const {
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        topk == -1 || topk > 0,
        "CalDistanceById topk must be -1 or positive");
    CHECK_ARGUMENT(query != nullptr, "CalDistanceById query must not be null");
    CHECK_ARGUMENT(count >= 0, "CalDistanceById count must be non-negative");
    if (count > 0) {
        CHECK_ARGUMENT(ids != nullptr, "CalDistanceById ids must not be null");
    }
    const int64_t num_queries = query->GetNumElements();
    CHECK_ARGUMENT(num_queries > 0, "CalDistanceById query count must be positive");
    CHECK_ARGUMENT(query->GetSparseVectors() != nullptr,
                   "CalDistanceById query sparse vectors must not be null");

    const auto count_size = static_cast<uint64_t>(count);
    const auto num_queries_size = static_cast<uint64_t>(num_queries);
    const auto max_distance_count = std::numeric_limits<uint64_t>::max() / sizeof(float);
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        count_size == 0 || num_queries_size <= max_distance_count / count_size,
        "CalDistanceById distance buffer size overflows");

    auto result = Dataset::Make();
    result->NumElements(num_queries)->Dim(count)->Owner(true, allocator_);
    if (count == 0) {
        return result;
    }

    auto* distances =
        static_cast<float*>(allocator_->Allocate(sizeof(float) * num_queries_size * count_size));
    CHECK_ARGUMENT(distances != nullptr, "Failed to allocate SINDI multi-query distance buffer");
    result->Distances(distances);

    std::shared_lock rlock(this->global_mutex_);
    CHECK_ARGUMENT(immutable_term_datacell_ == nullptr,
                   "immutable SINDI runtime does not support CalDistanceById");

    Vector<int64_t> inner_ids(count, -1, allocator_);
    std::unordered_map<int64_t, std::vector<int64_t>> window_positions;
    Vector<float> window_dists(window_size_, 0.0F, allocator_);
    std::vector<bool> validity(num_queries_size * count_size, false);
    for (int64_t query_index = 0; query_index < num_queries; ++query_index) {
        const auto& sparse_query = query->GetSparseVectors()[query_index];
        const auto* row_ids = ids + query_index * count;
        auto* row_distances = distances + query_index * count;
        const auto validity_offset = static_cast<uint64_t>(query_index) * count_size;

        if (use_reorder_ && calculate_precise_distance) {
            auto* rerank_datacell = this->get_rerank_datacell();
            auto computer = rerank_datacell->FactoryComputer(&sparse_query);
            for (int64_t index = 0; index < count; ++index) {
                const auto [success, inner_id] =
                    this->label_table_->TryGetIdByLabel(row_ids[index]);
                if (not success) {
                    row_distances[index] = -1.0F;
                    continue;
                }
                validity[validity_offset + static_cast<uint64_t>(index)] = true;
                rerank_datacell->Query(row_distances + index, computer, &inner_id, 1);
            }
            continue;
        }

        auto mapped_query = sparse_query;
        Vector<uint32_t> tmp_ids(allocator_);
        Vector<float> tmp_vals(allocator_);
        if (remap_term_ids_) {
            mapped_query = remap_sparse_vector_for_query(mapped_query, tmp_ids, tmp_vals);
        }
        SINDISearchParameter search_param;
        search_param.query_prune_ratio = 0;
        auto computer =
            std::make_shared<SparseTermComputer>(mapped_query, search_param, allocator_);
        const QueryTermBuffers query_term_buffers(allocator_);
        window_positions.clear();
        for (int64_t index = 0; index < count; ++index) {
            const auto [success, inner_id] = this->label_table_->TryGetIdByLabel(row_ids[index]);
            if (not success) {
                inner_ids[index] = -1;
                row_distances[index] = -1.0F;
                continue;
            }
            inner_ids[index] = inner_id;
            validity[validity_offset + static_cast<uint64_t>(index)] = true;
            window_positions[inner_id / window_size_].push_back(index);
        }

        if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
            for (int64_t index = 0; index < count; ++index) {
                if (inner_ids[index] < 0) {
                    continue;
                }
                row_distances[index] = term_datacell_->CalcDistanceByInnerId(
                    computer, static_cast<uint32_t>(inner_ids[index]), query_term_buffers);
            }
        } else {
            SindiQueryContext query_context(allocator_);
            for (const auto& [window_id, positions] : window_positions) {
                std::fill(window_dists.begin(), window_dists.end(), 0.0F);
                term_datacell_->QueryWindow(window_dists.data(),
                                            static_cast<uint32_t>(window_id),
                                            computer,
                                            false,
                                            query_context);
                const auto window_start_id = window_id * window_size_;
                for (const auto position : positions) {
                    row_distances[position] =
                        1.0F + window_dists[inner_ids[position] - window_start_id];
                }
            }
        }
    }

    if (topk == -1) {
        return result;
    }
    return ApplyTopkWithValidity(distances, ids, count, num_queries, topk, validity, allocator_);
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
    if (not immutable_enabled_) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_BATCH_CALC_DISTANCE_BY_ID);
    }
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
    auto num_windows = term_datacell_ == nullptr ? 0 : term_datacell_->GetWindowCount();
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
    if (doc_prune_ratio_ != 0.0F) {
        tmp_ids.clear();
        tmp_ids.reserve(input.len_);
        for (uint32_t index = 0; index < input.len_; ++index) {
            if (not term_id_mapper_->TryMap(input.ids_[index]).has_value()) {
                tmp_ids.push_back(input.ids_[index]);
            }
        }
        std::sort(tmp_ids.begin(), tmp_ids.end());
        tmp_ids.erase(std::unique(tmp_ids.begin(), tmp_ids.end()), tmp_ids.end());
        CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
            term_id_mapper_->Size() <= term_id_limit_ &&
                tmp_ids.size() <= static_cast<uint64_t>(term_id_limit_ - term_id_mapper_->Size()),
            fmt::format("term id mapper is full: mapper size ({}) + new terms ({}) exceeds "
                        "term_id_limit ({})",
                        term_id_mapper_->Size(),
                        tmp_ids.size(),
                        term_id_limit_));
    }

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
