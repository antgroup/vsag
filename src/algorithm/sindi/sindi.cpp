
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
#include "utils/util_functions.h"

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
      window_size_(param->window_size),
      doc_prune_ratio_(param->doc_prune_ratio),
      window_term_list_(common_param.allocator_.get()) {
    window_term_list_.resize(1, nullptr);
    this->window_term_list_[0] =
        std::make_shared<SparseTermDataCell>(doc_prune_ratio_, common_param.allocator_.get());

    if (use_reorder_) {
        SparseIndexParameterPtr rerank_param = std::make_shared<SparseIndexParameters>();
        rerank_param->need_sort = true;
        rerank_flat_index_ = std::make_shared<SparseIndex>(rerank_param, common_param);
    }
}

std::vector<int64_t>
SINDI::Add(const DatasetPtr& base) {
    auto data_num = base->GetNumElements();
    CHECK_ARGUMENT(data_num > 0, "data_num is zero when add vectors");

    const auto* sparse_vectors = base->GetSparseVectors();
    const auto* ids = base->GetIds();

    // adjust window
    uint32_t start_add_window = cur_element_count_ / window_size_;
    uint32_t final_add_window = (cur_element_count_ + data_num) / window_size_;
    if ((cur_element_count_ + data_num) % window_size_ == 0) {
        window_term_list_.resize(final_add_window);
    } else {
        window_term_list_.resize(final_add_window + 1);
    }
    for (uint32_t i = start_add_window + 1; i < window_term_list_.size(); i++) {
        window_term_list_[i] = std::make_shared<SparseTermDataCell>(doc_prune_ratio_, allocator_);
    }

    // add process
    for (uint32_t i = 0; i < data_num; ++i) {
        auto cur_window = cur_element_count_ / window_size_;
        auto window_start_id = cur_window * window_size_;
        const auto& sparse_vector = sparse_vectors[i];

        label_table_->Insert(cur_element_count_, ids[i]);
        uint32_t inner_id = cur_element_count_ - window_start_id;
        window_term_list_[cur_window]->InsertVector(sparse_vector, inner_id);

        cur_element_count_++;
    }
    // high precision part
    if (use_reorder_) {
        rerank_flat_index_->Add(base);
    }

    return {};
}

std::vector<int64_t>
SINDI::Build(const DatasetPtr& base) {
    return this->Add(base);
}

DatasetPtr
SINDI::KnnSearch(const DatasetPtr& query,
                 int64_t k,
                 const std::string& parameters,
                 const FilterPtr& filter) const {
    // Due to concerns about the performance of this index
    // We have not yet implemented search with filtering capabilities
    const auto* sparse_vectors = query->GetSparseVectors();
    CHECK_ARGUMENT(query->GetNumElements() == 1, "num of query should be 1");
    auto sparse_query = sparse_vectors[0];

    // search parameter
    SINDISearchParameter search_param;
    search_param.FromJson(JsonType::parse(parameters));
    InnerSearchParam inner_param;
    inner_param.ef = search_param.n_candidate;
    if (search_param.n_candidate == DEFAULT_N_CANDIDATE or search_param.n_candidate <= k) {
        inner_param.ef = k;
    }
    inner_param.topk = k;

    FilterPtr ft = nullptr;
    if (filter != nullptr) {
        ft = std::make_shared<InnerIdWrapperFilter>(filter, *this->label_table_);
    }
    inner_param.is_inner_id_allowed = ft;

    auto computer = this->window_term_list_[0]->FactoryComputer(sparse_query, search_param);
    return search_impl<KNN_SEARCH>(computer, inner_param);
}

template <InnerSearchMode mode>
DatasetPtr
SINDI::search_impl(const SparseTermComputerPtr& computer,
                   const InnerSearchParam& inner_param) const {
    // computer and heap
    MaxHeap heap(this->allocator_);
    uint32_t k = 0;

    if constexpr (mode == KNN_SEARCH) {
        k = inner_param.topk;
    }

    // window iteration
    std::vector<float> dists(window_size_, 0.0);

    for (auto cur = 0; cur < window_term_list_.size(); cur++) {
        auto window_start_id = cur * window_size_;
        auto term_list = this->window_term_list_[cur];

        // compute
        term_list->Query(dists.data(), computer);

        // insert heap
        term_list->InsertHeap<mode>(dists.data(), computer, heap, inner_param, window_start_id);
    }

    if constexpr (mode == KNN_SEARCH) {
        // fill up to k
        while (heap.size() < k) {
            heap.push(
                {std::numeric_limits<float>::max(), 0});  // TODO(ZXY): replace with random points
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
                if (high_precise_distance < inner_param.radius) {
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
        k = heap.size();
        if (inner_param.range_search_limit_size != -1) {
            k = inner_param.range_search_limit_size;
        }
    }
    auto [results, ret_dists, ret_ids] = create_fast_dataset(static_cast<int64_t>(k), allocator_);

    while (heap.size() > k) {
        heap.pop();
    }

    int cur_size = static_cast<int>(heap.size());

    for (int j = cur_size - 1; j >= 0; j--) {
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
    // Due to concerns about the performance of this index
    // We have not yet implemented search with filtering capabilities
    const auto* sparse_vectors = query->GetSparseVectors();
    CHECK_ARGUMENT(query->GetNumElements() == 1, "num of query should be 1");
    auto sparse_query = sparse_vectors[0];

    // search parameter
    SINDISearchParameter search_param;
    search_param.FromJson(JsonType::parse(parameters));
    InnerSearchParam inner_param;

    inner_param.range_search_limit_size = static_cast<int>(limited_size);
    inner_param.radius = radius;

    FilterPtr ft = nullptr;
    if (filter != nullptr) {
        ft = std::make_shared<InnerIdWrapperFilter>(filter, *this->label_table_);
    }
    inner_param.is_inner_id_allowed = ft;

    auto computer = this->window_term_list_[0]->FactoryComputer(sparse_query, search_param);
    return search_impl<RANGE_SEARCH>(computer, inner_param);
}

void
SINDI::Serialize(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, cur_element_count_);

    uint32_t window_term_list_size = window_term_list_.size();
    StreamWriter::WriteObj(writer, window_term_list_size);
    for (const auto& window : window_term_list_) {
        window->Serialize(writer);
    }

    label_table_->Serialize(writer);

    if (use_reorder_) {
        rerank_flat_index_->Serialize(writer);
    }
}

void
SINDI::Deserialize(StreamReader& reader) {
    StreamReader::ReadObj(reader, cur_element_count_);

    uint32_t window_term_list_size = 0;
    StreamReader::ReadObj(reader, window_term_list_size);
    window_term_list_.resize(window_term_list_size);
    for (auto i = 0; i < window_term_list_.size(); i++) {
        if (i != 0) {
            window_term_list_[i] =
                std::make_shared<SparseTermDataCell>(doc_prune_ratio_, allocator_);
        }
        window_term_list_[i]->Deserialize(reader);
    }

    label_table_->Deserialize(reader);

    if (use_reorder_) {
        rerank_flat_index_->Deserialize(reader);
    }
}

}  // namespace vsag
