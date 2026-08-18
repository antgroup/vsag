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

#include "test_index_common.h"

namespace fixtures {

void
TestIndex::TestConcurrentAddSearch(const TestIndex::IndexPtr& index,
                                   const TestDatasetPtr& dataset,
                                   const std::string& search_param,
                                   float expected_recall,
                                   bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_ADD_CONCURRENT) or
        not index->CheckFeature(vsag::SUPPORT_SEARCH_CONCURRENT) or
        not index->CheckFeature(vsag::SUPPORT_ADD_SEARCH_CONCURRENT)) {
        return;
    }
    fixtures::logger::LoggerReplacer _;

    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    auto topk = gt_topK;
    auto base_count = dataset->base_->GetNumElements();
    auto temp_count = static_cast<int64_t>(base_count * 0.8);
    auto dim = dataset->base_->GetDim();
    auto temp_dataset = vsag::Dataset::Make();
    temp_dataset->Dim(dim)
        ->Ids(dataset->base_->GetIds())
        ->NumElements(temp_count)
        ->Paths(dataset->base_->GetPaths())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->SparseVectors(dataset->base_->GetSparseVectors())
        ->Owner(false);
    index->Build(temp_dataset);
    fixtures::ThreadPool pool(5);
    std::vector<std::future<int>> futures;

    auto func = [&](uint64_t i) -> int {
        auto data_one = vsag::Dataset::Make();
        data_one->Dim(dim)
            ->Ids(dataset->base_->GetIds() + i)
            ->NumElements(1)
            ->Paths(dataset->base_->GetPaths() + i)
            ->Float32Vectors(dataset->base_->GetFloat32Vectors() + i * dim)
            ->SparseVectors(dataset->base_->GetSparseVectors() + i)
            ->Owner(false);

        auto add_res = index->Add(data_one);
        auto search_res = index->KnnSearch(data_one, topk, search_param);

        bool ret = 0;
        if (not add_res.has_value() or not search_res.has_value()) {
            return -1;
        }
        if (search_res.value()->GetIds()[0] == dataset->base_->GetIds()[i]) {
            ret = 1;
        }
        return ret;
    };

    for (uint64_t j = temp_count; j < base_count; ++j) {
        futures.emplace_back(pool.enqueue(func, j));
    }

    float query_size = static_cast<float>(base_count - temp_count);
    float recall = 0;
    for (auto& res : futures) {
        auto val = res.get();
        REQUIRE(val != -1);
        recall += val;
    }
    REQUIRE(recall / query_size >= expected_recall);
    REQUIRE(index->GetNumElements() == base_count);
}

void
TestIndex::TestConcurrentAdd(const TestIndex::IndexPtr& index,
                             const TestDatasetPtr& dataset,
                             bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_ADD_CONCURRENT)) {
        return;
    }
    fixtures::logger::LoggerReplacer _;

    auto base_count = dataset->base_->GetNumElements();
    auto temp_count = static_cast<int64_t>(base_count * 0.8);
    auto dim = dataset->base_->GetDim();
    auto temp_dataset = vsag::Dataset::Make();
    temp_dataset->Dim(dim)
        ->Ids(dataset->base_->GetIds())
        ->NumElements(temp_count)
        ->Paths(dataset->base_->GetPaths())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->SparseVectors(dataset->base_->GetSparseVectors())
        ->Owner(false);
    index->Build(temp_dataset);
    fixtures::ThreadPool pool(5);
    using RetType = tl::expected<std::vector<int64_t>, vsag::Error>;
    std::vector<std::future<RetType>> futures;

    auto func = [&](uint64_t i) -> RetType {
        auto data_one = vsag::Dataset::Make();
        data_one->Dim(dim)
            ->Ids(dataset->base_->GetIds() + i)
            ->NumElements(1)
            ->Paths(dataset->base_->GetPaths() + i)
            ->Float32Vectors(dataset->base_->GetFloat32Vectors() + i * dim)
            ->SparseVectors(dataset->base_->GetSparseVectors() + i)
            ->Owner(false);
        auto add_index = index->Add(data_one);
        return add_index;
    };

    for (uint64_t j = temp_count; j < base_count; ++j) {
        futures.emplace_back(pool.enqueue(func, j));
    }

    for (auto& res : futures) {
        auto val = res.get();
        REQUIRE(val.has_value() == expected_success);
    }
    REQUIRE(index->GetNumElements() == base_count);
}

void
TestIndex::TestConcurrentKnnSearch(const TestIndex::IndexPtr& index,
                                   const TestDatasetPtr& dataset,
                                   const std::string& search_param,
                                   float expected_recall,
                                   bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_SEARCH_CONCURRENT)) {
        return;
    }
    fixtures::logger::LoggerReplacer _;

    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    std::vector<float> search_results(query_count, 0.0f);
    using RetType = std::pair<tl::expected<DatasetPtr, vsag::Error>, uint64_t>;
    std::vector<std::future<RetType>> futures;
    auto topk = gt_topK;
    fixtures::ThreadPool pool(5);

    auto func = [&](uint64_t i) -> RetType {
        auto query = get_one_query(queries, i);
        auto res = index->KnnSearch(query, topk, search_param);
        return {res, i};
    };

    for (auto i = 0; i < query_count; ++i) {
        futures.emplace_back(pool.enqueue(func, i));
    }

    for (auto& res1 : futures) {
        auto [res, id] = res1.get();
        if (not expected_success) {
            if (res.has_value()) {
                REQUIRE(res.value()->GetDim() == 0);
            }
        } else {
            REQUIRE(res.has_value() == expected_success);
        }
        if (!expected_success) {
            return;
        }
        REQUIRE(res.value()->GetDim() == topk);
        auto result = res.value()->GetIds();
        auto gt = gts->GetIds() + gt_topK * id;
        auto val = Intersection(gt, gt_topK, result, topk);
        search_results[id] = static_cast<float>(val) / static_cast<float>(gt_topK);
    }

    auto cur_recall = std::accumulate(search_results.begin(), search_results.end(), 0.0f);
    if (cur_recall <= expected_recall * query_count) {
        WARN(fmt::format("cur_result({}) <= expected_recall * query_count({})",
                         cur_recall,
                         expected_recall * query_count));
    }
    REQUIRE(cur_recall > expected_recall * query_count * RECALL_THRESHOLD);
}

void
TestIndex::TestConcurrentAddSearchRemove(const TestIndex::IndexPtr& index,
                                         const TestDatasetPtr& dataset,
                                         const std::string& search_param,
                                         bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_ADD_SEARCH_DELETE_CONCURRENT)) {
        return;
    }
    fixtures::logger::LoggerReplacer _;

    auto base_count = dataset->base_->GetNumElements();
    auto temp_count = static_cast<int64_t>(base_count * 0.8);
    auto dim = dataset->base_->GetDim();
    auto temp_dataset = vsag::Dataset::Make();
    temp_dataset->Dim(dim)
        ->Ids(dataset->base_->GetIds())
        ->NumElements(temp_count)
        ->Paths(dataset->base_->GetPaths())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->SparseVectors(dataset->base_->GetSparseVectors())
        ->Owner(false);
    index->Build(temp_dataset);
    fixtures::ThreadPool pool(5);
    std::vector<std::future<bool>> futures;

    auto func = [&](uint64_t i) -> bool {
        auto data_one = vsag::Dataset::Make();
        data_one->Dim(dim)
            ->Ids(dataset->base_->GetIds() + i)
            ->NumElements(1)
            ->Paths(dataset->base_->GetPaths() + i)
            ->Float32Vectors(dataset->base_->GetFloat32Vectors() + i * dim)
            ->SparseVectors(dataset->base_->GetSparseVectors() + i)
            ->Owner(false);
        auto add_index = index->Add(data_one);
        auto search_index = index->KnnSearch(data_one, 1, search_param);
        auto remove_index = index->Remove(*(dataset->base_->GetIds() + i));
        return add_index.has_value() && search_index.has_value() && remove_index.has_value();
    };

    for (uint64_t j = temp_count; j < base_count; ++j) {
        futures.emplace_back(pool.enqueue(func, j));
    }

    for (auto& res : futures) {
        auto val = res.get();
        REQUIRE(val);
    }
}

}  // namespace fixtures
