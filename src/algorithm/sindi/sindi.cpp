
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

#include "impl/heap/standard_heap.h"
#include "index_feature_list.h"
#include "storage/serialization.h"
#include "utils/util_functions.h"
#include "vsag/allocator.h"

namespace vsag {
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
      use_quantization_(param->use_quantization),
      term_id_limit_(param->term_id_limit),
      window_size_(param->window_size),
      doc_retain_ratio_(1.0F - param->doc_prune_ratio),
      window_term_list_(common_param.allocator_.get()),
      deserialize_without_footer_(param->deserialize_without_footer),
      deserialize_without_buffer_(param->deserialize_without_buffer),
      quantization_params_(std::make_shared<QuantizationParams>()),
      avg_doc_term_length_(param->avg_doc_term_length) {
    if (use_reorder_) {
        SparseIndexParameterPtr rerank_param = std::make_shared<SparseIndexParameters>();
        rerank_param->need_sort = true;
        rerank_flat_index_ = std::make_shared<SparseIndex>(rerank_param, common_param);
    }
}

std::vector<int64_t>
SINDI::Add(const DatasetPtr& base, AddMode mode) {
    std::scoped_lock wlock(this->global_mutex_);
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

    // adjust window
    int64_t final_add_window = align_up(cur_element_count_ + data_num, window_size_) / window_size_;
    bool window_changed = false;
    while (window_term_list_.size() < final_add_window) {
        window_term_list_.emplace_back(std::make_shared<SparseTermDataCell>(doc_retain_ratio_,
                                                                            term_id_limit_,
                                                                            allocator_,
                                                                            use_quantization_,
                                                                            quantization_params_));
        window_changed = true;
    }

    // add process
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
            window_term_list_[cur_window]->InsertVector(sparse_vector, inner_id);
        } catch (const std::runtime_error& e) {
            failed_ids.push_back(ids[i]);
            logger::warn("runtime error: {}", e.what());
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
            auto single_base = Dataset::Make();
            single_base->NumElements(1)
                ->SparseVectors(sparse_vectors + i)
                ->Ids(ids + i)
                ->Owner(false);
            rerank_flat_index_->Add(single_base);
        }
    }
    if (window_changed) {
        this->cal_memory_usage();
    }
    return failed_ids;
}

std::vector<int64_t>
SINDI::Build(const DatasetPtr& base) {
    // note that there's a wlock in Add()
    return this->Add(base);
}

bool
SINDI::UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update) {
    // Note:
    // 1. we only check whether the old vector is a subset of the new vector
    // 2. we do not actually update the vector
    auto check_and_cleanup = [this, id, &new_base](InnerIndexInterface* index) -> bool {
        SparseVector old_sv;
        uint32_t inner_id;
        {
            std::scoped_lock rlock(this->global_mutex_);
            inner_id = this->label_table_->GetIdByLabel(id);
        }
        index->GetSparseVectorByInnerId(inner_id, &old_sv, this->allocator_);

        const auto& new_sv = *new_base->GetSparseVectors();
        bool ret = is_subset_of_sparse_vector(old_sv, new_sv);

        this->allocator_->Deallocate(old_sv.vals_);
        this->allocator_->Deallocate(old_sv.ids_);
        return ret;
    };

    if (use_reorder_) {
        if (not check_and_cleanup(rerank_flat_index_.get())) {
            return false;
        }
    }

    return check_and_cleanup(this);
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

    FilterPtr ft = nullptr;
    if (filter != nullptr) {
        ft = std::make_shared<InnerIdWrapperFilter>(filter, *this->label_table_);
    }
    inner_param.is_inner_id_allowed = ft;

    auto computer = std::make_shared<SparseTermComputer>(sparse_query, search_param, allocator_);
    return search_impl<KNN_SEARCH>(
        computer, inner_param, allocator, search_param.use_term_lists_heap_insert);
}

template <InnerSearchMode mode>
DatasetPtr
SINDI::search_impl(const SparseTermComputerPtr& computer,
                   const InnerSearchParam& inner_param,
                   Allocator* allocator,
                   bool use_term_lists_heap_insert) const {
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
            rerank_flat_index_->sort_sparse_vector(computer->raw_query_);
        for (auto i = 0; i < candidate_size; i++) {
            auto inner_id = heap.top().second;
            auto high_precise_distance = rerank_flat_index_->CalDistanceByIdUnsafe(
                sorted_ids,
                sorted_vals,
                inner_id);  // TODO(ZXY): use flat to replace rerank_flat_index_
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

        return rerank_flat_index_->collect_results(high_precise_heap);
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

    FilterPtr ft = nullptr;
    if (filter != nullptr) {
        ft = std::make_shared<InnerIdWrapperFilter>(filter, *this->label_table_);
    }
    inner_param.is_inner_id_allowed = ft;

    auto computer = std::make_shared<SparseTermComputer>(sparse_query, search_param, allocator_);
    return search_impl<RANGE_SEARCH>(
        computer, inner_param, allocator_, search_param.use_term_lists_heap_insert);
}

void
SINDI::cal_memory_usage() {
    auto memory = sizeof(SINDI);
    memory += window_term_list_.size() * sizeof(SparseTermDataCellPtr);
    for (auto& window : window_term_list_) {
        memory += window->GetMemoryUsage();
    }
    if (this->rerank_flat_index_ != nullptr) {
        memory += this->rerank_flat_index_->GetMemoryUsage();
    }
    memory += sizeof(QuantizationParams);

    std::unique_lock lock(this->memory_usage_mutex_);
    this->current_memory_usage_.store(static_cast<int64_t>(memory));
}

void
SINDI::Serialize(StreamWriter& writer) const {
    std::shared_lock rlock(this->global_mutex_);

    StreamWriter::WriteObj(writer, cur_element_count_);

    if (use_quantization_) {
        StreamWriter::WriteObj(writer, quantization_params_->min_val);
        StreamWriter::WriteObj(writer, quantization_params_->max_val);
        StreamWriter::WriteObj(writer, quantization_params_->diff);
    }

    uint32_t window_term_list_size = window_term_list_.size();
    StreamWriter::WriteObj(writer, window_term_list_size);
    for (const auto& window : window_term_list_) {
        window->Serialize(writer);
    }

    label_table_->Serialize(writer);

    if (use_reorder_) {
        rerank_flat_index_->Serialize(writer);
    }

    JsonType jsonify_basic_info;
    auto metadata = std::make_shared<Metadata>();
    jsonify_basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());
    metadata->Set("basic_info", jsonify_basic_info);
    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}

void
SINDI::Deserialize(StreamReader& reader) {
    std::scoped_lock wlock(this->global_mutex_);

    if (not deserialize_without_footer_) {
        auto footer = Footer::Parse(reader);
        auto metadata = footer->GetMetadata();
        JsonType jsonify_basic_info = metadata->Get("basic_info");
        // Check if the index parameter is compatible
        {
            auto param = jsonify_basic_info[INDEX_PARAM].GetString();
            SINDIParameterPtr index_param = std::make_shared<SINDIParameter>();
            index_param->FromString(param);
            if (not this->create_param_ptr_->CheckCompatibility(index_param)) {
                auto message = fmt::format("SINDI index parameter not match, current: {}, new: {}",
                                           this->create_param_ptr_->ToString(),
                                           index_param->ToString());
                logger::error(message);
                throw VsagException(ErrorType::INVALID_ARGUMENT, message);
            }
        }
    }
    auto* reader_ptr = &reader;

    BufferStreamReader buffer_reader(
        &reader, std::numeric_limits<uint64_t>::max(), this->allocator_);
    if (not deserialize_without_buffer_) {
        reader_ptr = &buffer_reader;
    }
    auto& reader_ref = *reader_ptr;

    StreamReader::ReadObj(reader_ref, cur_element_count_);

    if (use_quantization_) {
        StreamReader::ReadObj(reader_ref, quantization_params_->min_val);
        StreamReader::ReadObj(reader_ref, quantization_params_->max_val);
        StreamReader::ReadObj(reader_ref, quantization_params_->diff);
    }

    uint32_t window_term_list_size = 0;
    StreamReader::ReadObj(reader_ref, window_term_list_size);
    window_term_list_.resize(window_term_list_size);
    for (auto& window : window_term_list_) {
        window = std::make_shared<SparseTermDataCell>(
            doc_retain_ratio_, term_id_limit_, allocator_, use_quantization_, quantization_params_);
        window->Deserialize(reader_ref);
    }

    label_table_->Deserialize(reader_ref);

    if (use_reorder_) {
        rerank_flat_index_->Deserialize(reader_ref);
    }
    this->cal_memory_usage();
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
    if (use_quantization_) {
        mem += avg_doc_term_length_ * num_elements * (sizeof(uint8_t) + sizeof(uint16_t));
    } else {
        mem += avg_doc_term_length_ * num_elements * (sizeof(float) + sizeof(uint16_t));
    }

    // size of rerank index is same as sindi
    if (use_reorder_) {
        mem *= 2;
    }

    // size of term list
    mem += sizeof(std::vector<float>) * 2 * term_id_limit_;

    return mem;
}

void
SINDI::GetSparseVectorByInnerId(InnerIdType inner_id,
                                SparseVector* data,
                                Allocator* specified_allocator) const {
    std::shared_lock rlock(this->global_mutex_);

    if (use_reorder_) {
        return this->rerank_flat_index_->GetSparseVectorByInnerId(
            inner_id, data, specified_allocator);
    }

    auto cur_window = inner_id / window_size_;
    auto window_start_id = cur_window * window_size_;
    auto term_list = this->window_term_list_[cur_window];

    term_list->GetSparseVector(inner_id - window_start_id, data, specified_allocator);
}

float
SINDI::CalcDistanceById(const DatasetPtr& vector,
                        int64_t id,
                        bool calculate_precise_distance) const {
    std::shared_lock rlock(this->global_mutex_);

    if (use_reorder_ && calculate_precise_distance) {
        return this->rerank_flat_index_->CalcDistanceById(vector, id);
    }

    auto inner_id = this->label_table_->GetIdByLabel(id);
    auto cur_window = inner_id / window_size_;
    auto window_start_id = cur_window * window_size_;
    auto term_list = this->window_term_list_[cur_window];

    const auto sparse_query = vector->GetSparseVectors()[0];
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
        return this->rerank_flat_index_->CalDistanceById(query, ids, count);
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
    this->immutable_ = true;
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
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_INNER_PRODUCT);
}

std::pair<int64_t, int64_t>
SINDI::get_min_max_window_id(const FilterPtr& filter) const {
    int64_t min_window_id = 0;
    auto max_window_id = static_cast<int64_t>(window_term_list_.size() - 1);

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

}  // namespace vsag
