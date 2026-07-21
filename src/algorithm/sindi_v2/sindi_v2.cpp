
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

#include "sindi_v2.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <istream>
#include <limits>
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
#include "io/reader_io/reader_io_parameter.h"
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

constexpr const char* SINDI_V2_TERM_LAYOUT_VERSION_KEY = "sindi_v2_term_layout_version";
constexpr int64_t SINDI_V2_TERM_LAYOUT_VERSION = 1;
constexpr const char* SINDI_V2_TERM_LAYOUT_KIND_KEY = "sindi_v2_term_layout_kind";
constexpr const char* SINDI_V2_TERM_LAYOUT_KIND = "term";
constexpr const char* SINDI_V2_RERANK_LAYOUT_NONE = "none";
constexpr const char* SINDI_V2_RERANK_LAYOUT_TOP_TERMS_SIGNATURE = "top_terms_signature";

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

    [[nodiscard]] uint64_t
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

    [[nodiscard]] uint64_t
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
                            uint64_t codes_size,
                            const Vector<uint32_t>& sorted_ids,
                            const Vector<float>& sorted_vals) {
    if (codes_size < sizeof(uint32_t)) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "invalid SINDIV2 rerank code: missing length prefix");
    }
    auto len = *reinterpret_cast<const uint32_t*>(codes);
    const auto expected_size = sizeof(uint32_t) + static_cast<uint64_t>(len) * sizeof(BufferEntry);
    if (expected_size > codes_size) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format(
                "invalid SINDIV2 rerank code: length {} exceeds blob size {}", len, codes_size));
    }
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
    bool need_release = false;
    const auto* codes = flat->GetCodesById(inner_id, need_release);
    const auto len = *reinterpret_cast<const uint32_t*>(codes);
    const auto codes_size = sizeof(uint32_t) + static_cast<uint64_t>(len) * sizeof(BufferEntry);
    const auto distance = compute_distance_from_codes(codes, codes_size, sorted_ids, sorted_vals);
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

std::vector<uint32_t>
make_top_terms_signature(const SparseVector& vector, uint32_t top_terms) {
    std::vector<std::pair<float, uint32_t>> weighted_terms;
    weighted_terms.reserve(vector.len_);
    for (uint32_t i = 0; i < vector.len_; ++i) {
        weighted_terms.emplace_back(vector.vals_[i], vector.ids_[i]);
    }
    auto compare_by_weight = [](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first > rhs.first;
        }
        return lhs.second < rhs.second;
    };
    if (weighted_terms.size() > top_terms) {
        std::partial_sort(weighted_terms.begin(),
                          weighted_terms.begin() + top_terms,
                          weighted_terms.end(),
                          compare_by_weight);
        weighted_terms.resize(top_terms);
    } else {
        std::sort(weighted_terms.begin(), weighted_terms.end(), compare_by_weight);
    }

    std::vector<uint32_t> signature;
    signature.reserve(weighted_terms.size());
    for (const auto& [_, term_id] : weighted_terms) {
        signature.push_back(term_id);
    }
    std::sort(signature.begin(), signature.end());
    return signature;
}

std::vector<uint64_t>
make_top_terms_signature_key(const SparseVector& vector, uint32_t top_terms) {
    auto signature = make_top_terms_signature(vector, top_terms);
    return {signature.begin(), signature.end()};
}

struct RerankLayoutRecord {
    const SparseVector* vector{nullptr};
    InnerIdType inner_id{0};
    std::vector<uint64_t> signature;
};

void
write_rerank_flat_with_layout(const FlattenInterfacePtr& rerank_flat,
                              std::vector<RerankLayoutRecord>& records,
                              const std::string& rerank_layout,
                              uint32_t rerank_layout_top_terms) {
    if (rerank_layout != SINDI_V2_RERANK_LAYOUT_NONE) {
        for (auto& record : records) {
            if (rerank_layout == SINDI_V2_RERANK_LAYOUT_TOP_TERMS_SIGNATURE) {
                record.signature =
                    make_top_terms_signature_key(*record.vector, rerank_layout_top_terms);
            }
        }
        std::sort(records.begin(), records.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.signature != rhs.signature) {
                return lhs.signature < rhs.signature;
            }
            return lhs.inner_id < rhs.inner_id;
        });
    }

    for (const auto& record : records) {
        rerank_flat->InsertVector(record.vector, record.inner_id);
    }
}

}  // namespace

ParamPtr
SINDIV2::CheckAndMappingExternalParam(const JsonType& external_param,
                                      const IndexCommonParam& common_param) {
    auto ptr = std::make_shared<SINDIV2Parameter>();
    ptr->FromJson(external_param);
    return ptr;
}

SINDIV2::SINDIV2(const SINDIV2ParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param),
      use_reorder_(param->use_reorder),
      sparse_value_quant_type_(param->sparse_value_quant_type),
      term_id_limit_(param->term_id_limit),
      window_size_(param->window_size),
      doc_retain_ratio_(1.0F - param->doc_prune_ratio),
      quantization_params_(std::make_shared<QuantizationParams>()),
      avg_doc_term_length_(param->avg_doc_term_length),
      remap_term_ids_(param->remap_term_ids),
      immutable_enabled_(param->immutable),
      rerank_layout_(param->rerank_layout),
      rerank_layout_top_terms_(param->rerank_layout_top_terms),
      param_(param),
      common_param_(common_param) {
    CHECK_ARGUMENT(window_size_ > 0, "window_size must be in (0, 65536]");
    CHECK_ARGUMENT(window_size_ <= 65536, "window_size must be in (0, 65536]");
    CHECK_ARGUMENT(term_id_limit_ > 0, "term_id_limit must be > 0");
    if (remap_term_ids_) {
        term_id_mapper_ =
            std::make_shared<TermIdMapper>(term_id_limit_, common_param.allocator_.get());
    }
    if (use_reorder_) {
        rerank_flat_ = create_rerank_flat(common_param, param->rerank_io_parameter);
    }
    if (immutable_enabled_) {
        term_datacell_ = std::make_shared<ImmutableSindiTermDataCell>(term_id_limit_,
                                                                      window_size_,
                                                                      remap_term_ids_,
                                                                      sparse_value_quant_type_,
                                                                      quantization_params_,
                                                                      allocator_);
    } else {
        term_datacell_ = std::make_shared<MutableSindiTermDataCell>(term_id_limit_,
                                                                    window_size_,
                                                                    allocator_,
                                                                    sparse_value_quant_type_,
                                                                    quantization_params_);
    }
}

MutableSindiTermDataCellPtr
SINDIV2::get_mutable_term_datacell() const {
    auto mutable_datacell = std::dynamic_pointer_cast<MutableSindiTermDataCell>(term_datacell_);
    CHECK_ARGUMENT(mutable_datacell != nullptr,
                   "SINDIV2 mutable operation requires MutableSindiTermDataCell");
    return mutable_datacell;
}

std::string
SINDIV2::GetStats() const {
    return "";
}

SparseVector
SINDIV2::sort_and_prune_sparse_vector_for_build(const SparseVector& input,
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

    uint32_t retained_count = static_cast<uint32_t>(sorted_terms.size());
    if (sorted_terms.size() > 1 && doc_retain_ratio_ != 1.0F) {
        float total_mass = 0.0F;
        for (const auto& [_, value] : sorted_terms) {
            total_mass += value;
        }
        const float retained_mass = total_mass * doc_retain_ratio_;
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
SINDIV2::init_quantization_params_from_pruned_vectors(const DatasetPtr& base) {
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    bool has_retained_value = false;
    Vector<std::pair<uint32_t, float>> sorted_terms(allocator_);
    Vector<uint32_t> pruned_ids(allocator_);
    Vector<float> pruned_vals(allocator_);
    const auto* sparse_vectors = base->GetSparseVectors();
    const auto* ids = base->GetIds();
    for (int64_t document = 0; document < base->GetNumElements(); ++document) {
        const auto& sparse_vector = sparse_vectors[document];
        if (sparse_vector.len_ == 0 || label_table_->CheckLabel(ids[document])) {
            continue;
        }
        try {
            const auto pruned = this->sort_and_prune_sparse_vector_for_build(
                sparse_vector, sorted_terms, pruned_ids, pruned_vals);
            for (uint32_t term = 0; term < pruned.len_; ++term) {
                min_val = std::min(min_val, pruned.vals_[term]);
                max_val = std::max(max_val, pruned.vals_[term]);
                has_retained_value = true;
            }
        } catch (const VsagException&) {
            continue;
        }
    }
    if (not has_retained_value) {
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
SINDIV2::Add(const DatasetPtr& base) {
    std::scoped_lock wlock(this->global_mutex_);

    CHECK_ARGUMENT(not immutable_enabled_, "immutable SINDIV2 does not support Add");

    if (is_deserialized_) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "SINDIV2 does not support Add after Deserialize");
    }
    const auto mutable_term_datacell = this->get_mutable_term_datacell();

    std::vector<int64_t> failed_ids;

    auto data_num = base->GetNumElements();
    CHECK_ARGUMENT(data_num > 0, "data_num is zero when add vectors");

    const auto* sparse_vectors = base->GetSparseVectors();
    const auto* ids = base->GetIds();
    const auto* extra_info = base->GetExtraInfos();
    const auto extra_info_size = base->GetExtraInfoSize();

    if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8 && cur_element_count_ == 0) {
        this->init_quantization_params_from_pruned_vectors(base);
    }

    Vector<std::pair<uint32_t, float>> sorted_terms(allocator_);
    Vector<uint32_t> pruned_ids(allocator_);
    Vector<float> pruned_vals(allocator_);
    Vector<uint32_t> remapped_ids(allocator_);
    std::vector<RerankLayoutRecord> rerank_layout_records;
    if (use_reorder_ && rerank_layout_ != SINDI_V2_RERANK_LAYOUT_NONE) {
        rerank_layout_records.reserve(data_num);
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
                mutable_term_datacell->InsertVector(remapped,
                                                    static_cast<uint32_t>(cur_element_count_));
            } else {
                mutable_term_datacell->InsertVector(pruned,
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
            logger::warn("memory allocation failed: {}", e.what());
            throw;
        }

        label_table_->Insert(cur_element_count_, ids[i]);

        if (extra_info_size > 0) {
            extra_infos_->InsertExtraInfo(extra_info + i * extra_info_size, cur_element_count_);
        }

        if (use_reorder_ && rerank_layout_ == SINDI_V2_RERANK_LAYOUT_NONE) {
            rerank_flat_->InsertVector(sparse_vectors + i, cur_element_count_);
        } else if (use_reorder_) {
            rerank_layout_records.push_back(
                {sparse_vectors + i, static_cast<InnerIdType>(cur_element_count_), {}});
        }

        cur_element_count_++;
    }

    if (use_reorder_ && rerank_layout_ != SINDI_V2_RERANK_LAYOUT_NONE) {
        write_rerank_flat_with_layout(
            rerank_flat_, rerank_layout_records, rerank_layout_, rerank_layout_top_terms_);
    }

    this->cal_memory_usage();
    return failed_ids;
}

std::vector<int64_t>
SINDIV2::Build(const DatasetPtr& base) {
    CHECK_ARGUMENT(cur_element_count_ == 0, "SINDIV2 has already been built");
    if (immutable_enabled_) {
        return this->build_immutable(base);
    }
    auto failed_ids = this->Add(base);
    this->get_mutable_term_datacell()->Finalize();
    this->cal_memory_usage();
    return failed_ids;
}

std::vector<int64_t>
SINDIV2::build_immutable(const DatasetPtr& base) {
    std::scoped_lock wlock(this->global_mutex_);
    CHECK_ARGUMENT(not is_deserialized_, "SINDIV2 does not support Build after Deserialize");
    const auto data_num = base->GetNumElements();
    CHECK_ARGUMENT(data_num > 0, "data_num is zero when build immutable SINDIV2");
    const auto* sparse_vectors = base->GetSparseVectors();
    const auto* ids = base->GetIds();
    const auto* extra_info = base->GetExtraInfos();
    const auto extra_info_size = base->GetExtraInfoSize();

    if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
        this->init_quantization_params_from_pruned_vectors(base);
    }

    auto immutable = std::make_shared<ImmutableSindiTermDataCell>(term_id_limit_,
                                                                  window_size_,
                                                                  remap_term_ids_,
                                                                  sparse_value_quant_type_,
                                                                  quantization_params_,
                                                                  allocator_);
    immutable->Reserve(static_cast<uint32_t>(align_up(data_num, window_size_) / window_size_));
    MutableSindiTermDataCellPtr staging;
    const auto create_staging = [this]() {
        return std::make_shared<MutableSindiTermDataCell>(term_id_limit_,
                                                          window_size_,
                                                          allocator_,
                                                          sparse_value_quant_type_,
                                                          quantization_params_);
    };
    std::vector<int64_t> failed_ids;
    Vector<std::pair<uint32_t, float>> sorted_terms(allocator_);
    Vector<uint32_t> pruned_ids(allocator_);
    Vector<float> pruned_vals(allocator_);
    Vector<uint32_t> remapped_ids(allocator_);
    std::vector<RerankLayoutRecord> rerank_layout_records;
    if (use_reorder_ && rerank_layout_ != SINDI_V2_RERANK_LAYOUT_NONE) {
        rerank_layout_records.reserve(data_num);
    }
    for (int64_t i = 0; i < data_num; ++i) {
        const auto& sparse_vector = sparse_vectors[i];
        if (label_table_->CheckLabel(ids[i]) || sparse_vector.len_ == 0) {
            failed_ids.push_back(ids[i]);
            continue;
        }
        if (staging == nullptr) {
            staging = create_staging();
        }
        try {
            const auto local_id = static_cast<uint32_t>(cur_element_count_ % window_size_);
            const auto pruned = this->sort_and_prune_sparse_vector_for_build(
                sparse_vector, sorted_terms, pruned_ids, pruned_vals);
            if (remap_term_ids_) {
                const auto remapped = remap_sparse_vector_for_build(pruned, remapped_ids);
                staging->InsertVector(remapped, local_id);
            } else {
                staging->InsertVector(pruned, local_id);
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
        if (use_reorder_ && rerank_layout_ == SINDI_V2_RERANK_LAYOUT_NONE) {
            rerank_flat_->InsertVector(sparse_vectors + i, cur_element_count_);
        } else if (use_reorder_) {
            rerank_layout_records.push_back(
                {sparse_vectors + i, static_cast<InnerIdType>(cur_element_count_), {}});
        }
        cur_element_count_++;
        if (cur_element_count_ % window_size_ == 0) {
            staging->Compact();
            immutable->AppendWindow(staging->GetWindow(0));
            staging.reset();
        }
    }
    if (staging != nullptr && staging->total_count_ > 0) {
        staging->Compact();
        immutable->AppendWindow(staging->GetWindow(0));
        staging.reset();
    }
    if (use_reorder_ && rerank_layout_ != SINDI_V2_RERANK_LAYOUT_NONE) {
        write_rerank_flat_with_layout(
            rerank_flat_, rerank_layout_records, rerank_layout_, rerank_layout_top_terms_);
    }
    term_datacell_ = std::move(immutable);
    this->cal_memory_usage();
    return failed_ids;
}

bool
SINDIV2::UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update) {
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        "SINDIV2 does not support UpdateVector");
}

DatasetPtr
SINDIV2::KnnSearch(const DatasetPtr& query,
                   int64_t k,
                   const std::string& parameters,
                   const FilterPtr& filter) const {
    return KnnSearch(query, k, parameters, filter, allocator_);
}

DatasetPtr
SINDIV2::KnnSearch(const DatasetPtr& query,
                   int64_t k,
                   const std::string& parameters,
                   const FilterPtr& filter,
                   vsag::Allocator* allocator) const {
    std::shared_lock rlock(this->global_mutex_);
    auto* search_allocator = allocator != nullptr ? allocator : allocator_;

    const auto* sparse_vectors = query->GetSparseVectors();
    CHECK_ARGUMENT(query->GetNumElements() == 1, "num of query should be 1");
    CHECK_ARGUMENT(k > 0, "k must be greater than 0");
    auto sparse_query = sparse_vectors[0];
    CHECK_ARGUMENT(
        sparse_query.len_ > 0,
        fmt::format("query->GetSparseVectors()->len_ ({}) is invalid", sparse_query.len_));
    if (cur_element_count_ == 0) {
        auto [results, ret_dists, ret_ids] = create_fast_dataset(0, search_allocator);
        return results;
    }

    SINDIV2SearchParameter search_param;
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
    Vector<uint32_t> tmp_ids(search_allocator);
    Vector<float> tmp_vals(search_allocator);
    if (remap_term_ids_) {
        effective_query = remap_sparse_vector_for_query(sparse_query, tmp_ids, tmp_vals);
        if (effective_query.len_ == 0) {
            auto [results, ret_dists, ret_ids] = create_fast_dataset(0, search_allocator);
            return results;
        }
    }

    auto computer =
        std::make_shared<SparseTermComputer>(effective_query, search_param, search_allocator);
    const SparseVector* rerank_query = (remap_term_ids_ && use_reorder_) ? &sparse_query : nullptr;

    SindiQueryContext query_context(search_allocator);
    if (param_->term_io_parameter->GetTypeName() != IO_TYPE_VALUE_MEMORY_IO) {
        auto query_term_ids = collect_query_term_ids(computer, search_allocator);
        query_context.query_term_buffers = term_datacell_->LoadQueryTermBuffers(query_term_ids);
    }

    return search_impl<KNN_SEARCH>(computer,
                                   inner_param,
                                   search_allocator,
                                   search_param.use_term_lists_heap_insert,
                                   query_context,
                                   rerank_query);
}

template <InnerSearchMode mode>
DatasetPtr
SINDIV2::search_impl(const SparseTermComputerPtr& computer,
                     const InnerSearchParam& inner_param,
                     Allocator* allocator,
                     bool use_term_lists_heap_insert,
                     SindiQueryContext& query_context,
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
        term_datacell_->QueryWindow(dists.data(),
                                    static_cast<uint32_t>(cur),
                                    computer,
                                    use_term_lists_heap_insert,
                                    query_context);

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
            uint32_t valid_window_size = 0;
            if (window_start_id < static_cast<uint64_t>(cur_element_count_)) {
                auto remaining_count = static_cast<uint64_t>(cur_element_count_) - window_start_id;
                valid_window_size =
                    static_cast<uint32_t>(std::min<uint64_t>(window_size_, remaining_count));
            }
            term_datacell_->InsertHeapByDists(dists.data(),
                                              valid_window_size,
                                              heap,
                                              inner_param,
                                              window_start_id,
                                              mode,
                                              inner_param.is_inner_id_allowed != nullptr);
        }
    }

    // rerank
    if (use_reorder_) {
        float cur_heap_top = std::numeric_limits<float>::max();
        auto candidate_size = heap.size();
        auto high_precise_heap = std::make_shared<StandardHeap<true, false>>(allocator, -1);
        auto [sorted_ids, sorted_vals] =
            sort_sparse_vector(original_query ? *original_query : computer->raw_query_, allocator);

        const auto insert_result = [&](InnerIdType inner_id, float high_precise_distance) {
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
        };

        const auto rerank_io_type = param_->rerank_io_parameter->GetTypeName();
        const bool memory_rerank = rerank_io_type == IO_TYPE_VALUE_MEMORY_IO ||
                                   rerank_io_type == IO_TYPE_VALUE_BLOCK_MEMORY_IO;
        if (rerank_io_type == IO_TYPE_VALUE_MMAP_IO) {
            struct MMapCandidate {
                InnerIdType inner_id{0};
                const uint8_t* codes{nullptr};
            };
            Vector<MMapCandidate> candidates(allocator);
            candidates.reserve(candidate_size);
            for (uint64_t i = 0; i < candidate_size; ++i) {
                bool need_release = false;
                const auto inner_id = heap.top().second;
                const auto* codes = rerank_flat_->GetCodesById(inner_id, need_release);
                CHECK_ARGUMENT(codes != nullptr && not need_release,
                               "failed to access mmap SINDI V2 rerank payload");
                candidates.push_back({inner_id, codes});
                heap.pop();
            }
            std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
                return reinterpret_cast<uintptr_t>(lhs.codes) <
                       reinterpret_cast<uintptr_t>(rhs.codes);
            });
            for (const auto& candidate : candidates) {
                const auto len = *reinterpret_cast<const uint32_t*>(candidate.codes);
                const auto codes_size =
                    sizeof(uint32_t) + static_cast<uint64_t>(len) * sizeof(BufferEntry);
                const auto high_precise_distance = compute_distance_from_codes(
                    candidate.codes, codes_size, sorted_ids, sorted_vals);
                insert_result(candidate.inner_id, high_precise_distance);
            }
        } else if (memory_rerank) {
            for (uint64_t i = 0; i < candidate_size; ++i) {
                const auto inner_id = heap.top().second;
                const auto high_precise_distance =
                    cal_distance_by_id_unsafe(rerank_flat_, sorted_ids, sorted_vals, inner_id);
                insert_result(inner_id, high_precise_distance);
                heap.pop();
            }
        } else {
            // File-backed asynchronous IO benefits from sorting and merging candidate reads.
            Vector<InnerIdType> cand_ids(allocator);
            cand_ids.resize(candidate_size);
            for (int64_t i = static_cast<int64_t>(candidate_size) - 1; i >= 0; --i) {
                cand_ids[i] = heap.top().second;
                heap.pop();
            }
            std::sort(cand_ids.begin(), cand_ids.end());
            auto batch = rerank_flat_->GetCodesByIdsBatch(
                cand_ids.data(), static_cast<InnerIdType>(candidate_size), allocator);
            for (uint64_t i = 0; i < candidate_size; ++i) {
                const auto* codes = batch.buffer.data() + batch.in_buffer_offsets[i];
                const auto high_precise_distance =
                    compute_distance_from_codes(codes, batch.sizes[i], sorted_ids, sorted_vals);
                insert_result(cand_ids[i], high_precise_distance);
            }
        }

        return collect_results(high_precise_heap, allocator);
    }

    // low precision
    if constexpr (mode == RANGE_SEARCH) {
        k = static_cast<int64_t>(heap.size());
        if (inner_param.range_search_limit_size != -1) {
            k = inner_param.range_search_limit_size;
        }
    }

    int64_t cur_size = std::min(static_cast<int64_t>(heap.size()), k);

    auto [results, ret_dists, ret_ids] = create_fast_dataset(cur_size, allocator);
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
SINDIV2::RangeSearch(const DatasetPtr& query,
                     float radius,
                     const std::string& parameters,
                     const FilterPtr& filter,
                     int64_t limited_size) const {
    std::shared_lock rlock(this->global_mutex_);
    CHECK_ARGUMENT(query->GetNumElements() == 1, "num of query should be 1");
    auto sparse_query = query->GetSparseVectors()[0];
    CHECK_ARGUMENT(sparse_query.len_ > 0,
                   fmt::format("query sparse vector length {} is invalid", sparse_query.len_));
    if (cur_element_count_ == 0) {
        auto [results, ret_dists, ret_ids] = create_fast_dataset(0, allocator_);
        return results;
    }

    SINDIV2SearchParameter search_param;
    search_param.FromJson(JsonType::Parse(parameters));
    InnerSearchParam inner_param;
    inner_param.radius = radius;
    inner_param.range_search_limit_size = static_cast<int>(limited_size);
    if (filter != nullptr) {
        inner_param.is_inner_id_allowed =
            std::make_shared<InnerIdWrapperFilter>(filter, *this->label_table_);
    }

    Vector<uint32_t> tmp_ids(allocator_);
    Vector<float> tmp_vals(allocator_);
    if (remap_term_ids_) {
        sparse_query = remap_sparse_vector_for_query(sparse_query, tmp_ids, tmp_vals);
        if (sparse_query.len_ == 0) {
            auto [results, ret_dists, ret_ids] = create_fast_dataset(0, allocator_);
            return results;
        }
    }
    auto computer = std::make_shared<SparseTermComputer>(sparse_query, search_param, allocator_);
    const auto query_term_ids = collect_query_term_ids(computer, allocator_);
    SindiQueryContext query_context(allocator_);
    query_context.query_term_buffers = term_datacell_->LoadQueryTermBuffers(query_term_ids);
    const SparseVector* rerank_query =
        remap_term_ids_ && use_reorder_ ? &query->GetSparseVectors()[0] : nullptr;
    return search_impl<RANGE_SEARCH>(computer,
                                     inner_param,
                                     allocator_,
                                     search_param.use_term_lists_heap_insert,
                                     query_context,
                                     rerank_query);
}

void
SINDIV2::cal_memory_usage() {
    auto memory = sizeof(SINDIV2);
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

uint32_t
SINDIV2::get_term_dict_count() const {
    if (remap_term_ids_) {
        return term_id_mapper_ == nullptr ? 0 : term_id_mapper_->Size();
    }
    return term_datacell_ == nullptr ? 0 : term_datacell_->GetTermDictCount();
}

void
SINDIV2::serialize_term_layout(StreamWriter& writer) const {
    const auto term_dict_count = this->get_term_dict_count();
    CHECK_ARGUMENT(term_datacell_ != nullptr, "SINDIV2 has no term data cell to serialize");
    term_datacell_->SerializeTermLayout(writer, term_dict_count);
}

void
SINDIV2::Serialize(StreamWriter& writer) const {
    std::shared_lock rlock(this->global_mutex_);

    StreamWriter::WriteObj(writer, cur_element_count_);
    if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
        StreamWriter::WriteObj(writer, quantization_params_->min_val);
        StreamWriter::WriteObj(writer, quantization_params_->max_val);
        StreamWriter::WriteObj(writer, quantization_params_->diff);
    }

    this->serialize_term_layout(writer);

    if (use_reorder_) {
        rerank_flat_->Serialize(writer);
    }

    label_table_->Serialize(writer);

    if (remap_term_ids_ && term_id_mapper_) {
        term_id_mapper_->Serialize(writer);
    }

    // Footer
    JsonType jsonify_basic_info;
    auto metadata = std::make_shared<Metadata>();
    jsonify_basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());
    jsonify_basic_info[SINDI_V2_TERM_LAYOUT_VERSION_KEY].SetInt(SINDI_V2_TERM_LAYOUT_VERSION);
    jsonify_basic_info[SINDI_V2_TERM_LAYOUT_KIND_KEY].SetString(SINDI_V2_TERM_LAYOUT_KIND);
    metadata->Set("basic_info", jsonify_basic_info);
    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}

void
SINDIV2::SetIO(const std::shared_ptr<Reader> reader) {
    auto reader_param = std::make_shared<ReaderIOParameter>();
    reader_param->reader = reader;
    const auto disk_datacell =
        std::dynamic_pointer_cast<DiskSindiTermDataCellInterface>(term_datacell_);
    if (disk_datacell != nullptr) {
        disk_datacell->SetIO(reader);
    }
    if (rerank_flat_ != nullptr && param_->rerank_io_parameter != nullptr &&
        param_->rerank_io_parameter->GetTypeName() == IO_TYPE_VALUE_READER_IO) {
        rerank_flat_->InitIO(reader_param);
    }
}

void
SINDIV2::Deserialize(const BinarySet& binary_set) {
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
SINDIV2::Deserialize(std::istream& in_stream) {
    auto reader_holder = std::make_shared<StreamBackedReader>(in_stream);
    auto func = [reader_holder](uint64_t offset, uint64_t len, void* dest) {
        reader_holder->Read(offset, len, dest);
    };
    uint64_t cursor = static_cast<uint64_t>(in_stream.tellg());
    auto reader = ReadFuncStreamReader(func, cursor, reader_holder->Size());

    this->Deserialize(reader);
    this->SetIO(reader_holder);
}

void
SINDIV2::Deserialize(StreamReader& reader) {
    std::scoped_lock wlock(this->global_mutex_);

    const auto serialized_base_offset = reader.GetCursor();
    auto footer = Footer::Parse(reader);
    CHECK_ARGUMENT(footer != nullptr, "SINDI V2 footer is required");
    auto metadata = footer->GetMetadata();
    if (metadata->EmptyIndex()) {
        return;
    }
    JsonType jsonify_basic_info = metadata->Get("basic_info");
    CHECK_ARGUMENT(jsonify_basic_info.Contains(SINDI_V2_TERM_LAYOUT_VERSION_KEY),
                   "SINDI V2 term layout version is missing");
    CHECK_ARGUMENT(jsonify_basic_info[SINDI_V2_TERM_LAYOUT_VERSION_KEY].GetInt() ==
                       SINDI_V2_TERM_LAYOUT_VERSION,
                   "unsupported SINDI V2 term layout version");
    CHECK_ARGUMENT(jsonify_basic_info.Contains(SINDI_V2_TERM_LAYOUT_KIND_KEY) &&
                       jsonify_basic_info[SINDI_V2_TERM_LAYOUT_KIND_KEY].GetString() ==
                           SINDI_V2_TERM_LAYOUT_KIND,
                   "invalid SINDIV2 term layout kind");
    auto param = jsonify_basic_info[INDEX_PARAM].GetString();
    SINDIV2ParameterPtr index_param = std::make_shared<SINDIV2Parameter>();
    index_param->FromString(param);
    if (not this->create_param_ptr_->CheckCompatibility(index_param)) {
        auto message = fmt::format("SINDI V2 index parameter not match, current: {}, new: {}",
                                   this->create_param_ptr_->ToString(),
                                   index_param->ToString());
        logger::error(message);
        throw VsagException(ErrorType::INVALID_ARGUMENT, message);
    }

    reader.Seek(serialized_base_offset);
    StreamReader::ReadObj(reader, cur_element_count_);

    if (sparse_value_quant_type_ == SparseValueQuantizationType::SQ8) {
        StreamReader::ReadObj(reader, quantization_params_->min_val);
        StreamReader::ReadObj(reader, quantization_params_->max_val);
        StreamReader::ReadObj(reader, quantization_params_->diff);
    }

    auto window_count =
        static_cast<uint32_t>(align_up(cur_element_count_, window_size_) / window_size_);
    const auto use_memory_term_layout =
        param_->term_io_parameter != nullptr &&
        param_->term_io_parameter->GetTypeName() == IO_TYPE_VALUE_MEMORY_IO;
    if (use_memory_term_layout) {
        if (immutable_enabled_) {
            auto immutable_datacell =
                std::make_shared<ImmutableSindiTermDataCell>(term_id_limit_,
                                                             window_size_,
                                                             remap_term_ids_,
                                                             sparse_value_quant_type_,
                                                             quantization_params_,
                                                             allocator_);
            immutable_datacell->DeserializeTermLayout(reader, window_count, cur_element_count_);
            term_datacell_ = std::move(immutable_datacell);
        } else {
            auto mutable_datacell =
                std::make_shared<MutableSindiTermDataCell>(term_id_limit_,
                                                           window_size_,
                                                           allocator_,
                                                           sparse_value_quant_type_,
                                                           quantization_params_);
            mutable_datacell->DeserializeTermLayout(reader, window_count, cur_element_count_);
            term_datacell_ = std::move(mutable_datacell);
        }
    } else {
        auto disk_datacell = DiskSindiTermDataCellInterface::MakeInstance(term_id_limit_,
                                                                          allocator_,
                                                                          sparse_value_quant_type_,
                                                                          quantization_params_,
                                                                          window_size_,
                                                                          param_->term_io_parameter,
                                                                          common_param_);
        disk_datacell->DeserializeTermLayout(reader, window_count, cur_element_count_);
        term_datacell_ = std::move(disk_datacell);
    }

    if (use_reorder_) {
        rerank_flat_->Deserialize(reader);
    }

    label_table_->Deserialize(reader);

    if (remap_term_ids_ && term_id_mapper_) {
        term_id_mapper_->Deserialize(reader);
        CHECK_ARGUMENT(term_id_mapper_->Size() == term_datacell_->GetTermDictCount(),
                       "SINDIV2 remapped term dict count does not match mapper size");
    }

    is_deserialized_ = true;
    this->cal_memory_usage();
}

std::pair<int64_t, int64_t>
SINDIV2::GetMinAndMaxId() const {
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
SINDIV2::EstimateMemory(uint64_t num_elements) const {
    uint64_t mem = 0;
    mem += sizeof(SINDIV2);
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
SINDIV2::GetSparseVectorByInnerId(InnerIdType inner_id,
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
SINDIV2::CalcDistanceById(const DatasetPtr& vector,
                          int64_t id,
                          bool calculate_precise_distance) const {
    std::shared_lock rlock(this->global_mutex_);

    auto [success, inner_id] = this->label_table_->TryGetIdByLabel(id);
    if (not success) {
        return -1.0F;
    }

    if (use_reorder_ && calculate_precise_distance) {
        auto [sorted_ids, sorted_vals] =
            sort_sparse_vector(vector->GetSparseVectors()[0], allocator_);
        return cal_distance_by_id_unsafe(rerank_flat_, sorted_ids, sorted_vals, inner_id);
    }

    auto sparse_query = vector->GetSparseVectors()[0];
    Vector<uint32_t> tmp_ids(allocator_);
    Vector<float> tmp_vals(allocator_);
    if (remap_term_ids_) {
        sparse_query = remap_sparse_vector_for_query(sparse_query, tmp_ids, tmp_vals);
    }
    SINDIV2SearchParameter search_param;
    search_param.query_prune_ratio = 0;
    search_param.term_prune_ratio = 0;
    auto computer = std::make_shared<SparseTermComputer>(sparse_query, search_param, allocator_);
    auto query_term_ids = collect_query_term_ids(computer, allocator_);
    auto query_term_buffers = term_datacell_->LoadQueryTermBuffers(query_term_ids);
    return term_datacell_->CalcDistanceByInnerId(
        computer, static_cast<uint32_t>(inner_id), query_term_buffers);
}

DatasetPtr
SINDIV2::CalDistanceById(const DatasetPtr& query,
                         const int64_t* ids,
                         int64_t count,
                         bool calculate_precise_distance) const {
    std::shared_lock rlock(this->global_mutex_);

    if (use_reorder_ && calculate_precise_distance) {
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

    auto sparse_query = query->GetSparseVectors()[0];
    Vector<uint32_t> tmp_ids(allocator_);
    Vector<float> tmp_vals(allocator_);
    if (remap_term_ids_) {
        sparse_query = remap_sparse_vector_for_query(sparse_query, tmp_ids, tmp_vals);
    }
    SINDIV2SearchParameter search_param;
    search_param.query_prune_ratio = 0;
    search_param.term_prune_ratio = 0;
    auto computer = std::make_shared<SparseTermComputer>(sparse_query, search_param, allocator_);
    auto query_term_ids = collect_query_term_ids(computer, allocator_);
    auto query_term_buffers = term_datacell_->LoadQueryTermBuffers(query_term_ids);

    for (int64_t i = 0; i < count; i++) {
        auto [success, inner_id] = this->label_table_->TryGetIdByLabel(ids[i]);
        if (success) {
            distances[i] = term_datacell_->CalcDistanceByInnerId(
                computer, static_cast<uint32_t>(inner_id), query_term_buffers);
        }
    }

    return result;
}

void
SINDIV2::SetImmutable() {
    std::scoped_lock wlock(this->global_mutex_);
    this->immutable_ = true;
}

void
SINDIV2::InitFeatures() {
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_BUILD_WITH_MULTI_THREAD,
        IndexFeature::SUPPORT_KNN_SEARCH,
        IndexFeature::SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
        IndexFeature::SUPPORT_RANGE_SEARCH,
        IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER,
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
SINDIV2::get_min_max_window_id(const FilterPtr& filter) const {
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
SINDIV2::remap_sparse_vector_for_query(const SparseVector& input,
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
SINDIV2::remap_sparse_vector_for_build(const SparseVector& input, Vector<uint32_t>& tmp_ids) {
    tmp_ids.clear();
    tmp_ids.reserve(input.len_);
    for (uint32_t index = 0; index < input.len_; ++index) {
        if (not term_id_mapper_->TryMap(input.ids_[index]).has_value()) {
            tmp_ids.push_back(input.ids_[index]);
        }
    }
    std::sort(tmp_ids.begin(), tmp_ids.end());
    tmp_ids.erase(std::unique(tmp_ids.begin(), tmp_ids.end()), tmp_ids.end());
    CHECK_ARGUMENT(
        term_id_mapper_->Size() <= term_id_limit_ &&
            tmp_ids.size() <= static_cast<uint64_t>(term_id_limit_ - term_id_mapper_->Size()),
        fmt::format("term id mapper is full: mapper size ({}) + new terms ({}) exceeds "
                    "term_id_limit ({})",
                    term_id_mapper_->Size(),
                    tmp_ids.size(),
                    term_id_limit_));

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
