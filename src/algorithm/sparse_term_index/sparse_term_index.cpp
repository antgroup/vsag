
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

#include "sparse_term_index.h"

#include "utils/util_functions.h"

namespace vsag {
ParamPtr
SparseTermIndex::CheckAndMappingExternalParam(const JsonType& external_param,
                                              const IndexCommonParam& common_param) {
    auto ptr = std::make_shared<SparseTermIndexParameters>();
    ptr->FromJson(external_param);
    return ptr;
}

SparseTermIndex::SparseTermIndex(const SparseTermIndexParameterPtr& param,
                                 const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param),
      use_reorder_(param->use_reorder),
      window_size_(param->window_size),
      window_term_list_(common_param.allocator_.get()) {
    window_term_list_.resize(1, nullptr);
    this->window_term_list_[0] =
        std::make_shared<SparseTermDataCell>(param->query_prune_ratio,
                                             param->doc_prune_ratio,
                                             param->term_prune_ratio,
                                             common_param.allocator_.get());

    if (use_reorder_) {
        SparseIndexParameterPtr rerank_param = std::make_shared<SparseIndexParameters>();
        rerank_param->need_sort = true;
        rerank_flat_index_ = std::make_shared<SparseIndex>(rerank_param, common_param);
    }
}

std::vector<int64_t>
SparseTermIndex::Add(const DatasetPtr& base) {
    auto data_num = base->GetNumElements();
    CHECK_ARGUMENT(data_num > 0, "data_num is zero when add vectors");

    const auto* sparse_vectors = base->GetSparseVectors();
    const auto* ids = base->GetIds();

    for (int64_t i = 0; i < data_num; ++i) {
        auto cur_window = cur_element_count_ / window_size_;
        auto window_start_id = cur_window * window_size_;
        if (cur_window + 1 > window_term_list_.size()) {
            window_term_list_.resize(cur_window + 1);
            window_term_list_[cur_window] = this->window_term_list_[0]->Clone();
        }

        label_table_->Insert(cur_element_count_, ids[i]);
        const auto& sparse_vector = sparse_vectors[i];
        uint32_t inner_id = cur_element_count_ - window_start_id;

        window_term_list_[cur_window]->InsertVector(sparse_vector, inner_id);

        cur_element_count_++;
    }

    if (use_reorder_) {
        rerank_flat_index_->Add(base);
    }

    for (int64_t i = 0; i < window_term_list_.size(); i++) {
        window_term_list_[i]->TermPrune();
    }
    return {};
}

std::vector<int64_t>
SparseTermIndex::Build(const DatasetPtr& base) {
    return this->Add(base);
}

DatasetPtr
SparseTermIndex::KnnSearch(const DatasetPtr& query,
                           int64_t k,
                           const std::string& parameters,
                           const FilterPtr& filter) const {
    // Due to concerns about the performance of this index
    // We have not yet implemented search with filtering capabilities
    const auto* sparse_vectors = query->GetSparseVectors();
    CHECK_ARGUMENT(query->GetNumElements() == 1, "num of query should be 1");
    auto sparse_query = sparse_vectors[0];

    // search parameter
    SparseTermIndexParameters search_param;
    search_param.FromJson(JsonType::parse(parameters));
    auto n_candidate = search_param.n_candidate;
    if (n_candidate == DEFAULT_N_CANDIDATE) {
        n_candidate = k;
    } else {
        CHECK_ARGUMENT(
            k <= n_candidate,
            fmt::format("n_candidate({}) must be greater or equal to k({})", n_candidate, k));
    }

    // computer and heap
    auto computer = this->window_term_list_[0]->FactoryComputer(sparse_query);
    MaxHeap heap(this->allocator_);
    //    float cur_heap_top = std::numeric_limits<float>::max();

    // window iteration
    std::vector<float> dists(window_size_, 0.0);

    {  // pre-fill the heap
        auto term_list = this->window_term_list_[0];

        // compute
        term_list->Query(dists.data(), computer);

        // insert heap
        term_list->InsertHeapPreFill(dists.data(), computer, heap, n_candidate);
    }

    for (auto cur = 1; cur < window_term_list_.size(); cur++) {
        auto window_start_id = cur * window_size_;
        auto term_list = this->window_term_list_[cur];

        // compute
        term_list->Query(dists.data(), computer);

        // insert heap
        term_list->InsertHeapFull(dists.data(), computer, heap, window_start_id);
    }

    // rerank
    if (use_reorder_) {
        // high precision
        auto candidate_size = heap.size();
        auto high_precise_heap = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
        auto [sorted_ids, sorted_vals] = rerank_flat_index_->sort_sparse_vector(sparse_query);
        for (auto i = 0; i < candidate_size; i++) {
            auto inner_id = heap.top().second;
            auto high_precise_distance = rerank_flat_index_->CalDistanceByIdUnsafe(
                sorted_ids,
                sorted_vals,
                inner_id);  // TODO(ZXY): use flat to replace rerank_flat_index_
            auto label = label_table_->GetLabelById(inner_id);
            high_precise_heap->Push(high_precise_distance, label);
            if (high_precise_heap->Size() > k) {
                high_precise_heap->Pop();
            }
            heap.pop();
        }

        return rerank_flat_index_->collect_results(high_precise_heap);
    } else {
        // low precision
        auto [results, ret_dists, ret_ids] = CreateFastDataset(static_cast<int64_t>(k), allocator_);

        while (heap.size() > k) {
            heap.pop();
        }

        int cur_size = heap.size();

        for (int j = cur_size - 1; j >= 0; j--) {
            ret_dists[j] = -heap.top().first;
            ret_ids[j] = label_table_->GetLabelById(heap.top().second);
            heap.pop();
        }

        return results;
    }
}

}  // namespace vsag