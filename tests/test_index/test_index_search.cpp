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
TestIndex::TestKnnSearchCompare(const IndexPtr& index_weak,
                                const IndexPtr& index_strong,
                                const TestDatasetPtr& dataset,
                                const std::string& search_param,
                                bool expected_success) {
    if (not index_weak->CheckFeature(vsag::SUPPORT_KNN_SEARCH) or
        not index_strong->CheckFeature(vsag::SUPPORT_KNN_SEARCH)) {
        return;
    }

    double time_cost_weak = 0;
    double time_cost_strong = 0;

    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto topk = dataset->top_k;
    for (auto round = 0; round < 2; round++) {
        for (auto i = 0; i < query_count; ++i) {
            auto query = get_one_query(queries, i);

            if (round == 0) {
                auto st = std::chrono::high_resolution_clock::now();
                auto res = index_weak->KnnSearch(query, topk, search_param);
                auto ed = std::chrono::high_resolution_clock::now();
                time_cost_weak += std::chrono::duration<double>(ed - st).count();
            } else {
                auto st = std::chrono::high_resolution_clock::now();
                auto res = index_strong->KnnSearch(query, topk, search_param);
                auto ed = std::chrono::high_resolution_clock::now();
                time_cost_strong += std::chrono::duration<double>(ed - st).count();
            }
        }
    }
    if (expected_success) {
        REQUIRE(time_cost_weak > time_cost_strong);
    }
}

void
TestIndex::TestKnnSearch(const IndexPtr& index,
                         const TestDatasetPtr& dataset,
                         const std::string& search_param,
                         float expected_recall,
                         bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_KNN_SEARCH)) {
        return;
    }
    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    float cur_recall = 0.0f;
    auto topk = gt_topK;
    for (auto i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        auto res = index->KnnSearch(query, topk, search_param);
        if (not expected_success) {
            if (res.has_value()) {
                REQUIRE(res.value()->GetDim() == 0);
            }
            return;
        } else {
            REQUIRE(res.has_value() == true);
        }
        REQUIRE(res.value()->GetDim() == topk);
        auto result = res.value()->GetIds();
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK, result, topk);
        cur_recall += static_cast<float>(val) / static_cast<float>(gt_topK);
    }
    if (cur_recall <= expected_recall * query_count) {
        WARN(fmt::format("cur_result({}) <= expected_recall * query_count({})",
                         cur_recall,
                         expected_recall * query_count));
    }
    REQUIRE(cur_recall > expected_recall * query_count * RECALL_THRESHOLD);
}

void
TestIndex::TestRangeSearch(const IndexPtr& index,
                           const TestDatasetPtr& dataset,
                           const std::string& search_param,
                           float expected_recall,
                           int64_t limited_size,
                           bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_RANGE_SEARCH)) {
        return;
    }
    auto queries = dataset->range_query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->range_ground_truth_;
    auto gt_topK = gts->GetDim();
    const auto& radius = dataset->range_radius_;
    float cur_recall = 0.0f;
    for (auto i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        auto res = index->RangeSearch(query, radius[i], search_param, limited_size);
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
        if (limited_size > 0) {
            REQUIRE(res.value()->GetDim() <= limited_size);
        }
        auto res_dim = res.value()->GetDim();
        auto result = res.value()->GetIds();
        auto gt = gts->GetIds() + gt_topK * i;
        auto gt_count = std::max<int64_t>(0, gt_topK - 1);
        auto val = Intersection(gt, gt_count, result, res_dim);
        auto denominator = std::min(gt_count, res_dim);
        // Skip recall calculation if denominator is 0 to avoid NaN when RangeSearch returns
        // empty result.
        if (denominator > 0) {
            cur_recall += static_cast<float>(val) / static_cast<float>(denominator);
        }
    }
    if (cur_recall <= expected_recall * query_count) {
        WARN(fmt::format("cur_result({}) <= expected_recall * query_count({})",
                         cur_recall,
                         expected_recall * query_count));
    }
    REQUIRE(cur_recall > expected_recall * query_count * RECALL_THRESHOLD);
}

class FilterObj : public vsag::Filter {
public:
    FilterObj(std::function<bool(int64_t)> filter_func,
              std::function<bool(const char*)> ex_filter_func,
              float valid_ratio)
        : filter_func_(std::move(filter_func)),
          ex_filter_func_(std::move(ex_filter_func)),
          valid_ratio_(valid_ratio) {
    }

    bool
    CheckValid(int64_t id) const override {
        return not filter_func_(id);
    }

    bool
    CheckValid(const char* data) const override {
        if (not ex_filter_func_)
            return vsag::Filter::CheckValid(data);
        else
            return not ex_filter_func_(data);
    }

    float
    ValidRatio() const override {
        return valid_ratio_;
    }

private:
    std::function<bool(int64_t)> filter_func_{nullptr};
    std::function<bool(const char*)> ex_filter_func_{nullptr};
    float valid_ratio_{1.0F};
};

void
TestIndex::TestKnnSearchIter(const IndexPtr& index,
                             const TestDatasetPtr& dataset,
                             const std::string& search_param,
                             float expected_recall,
                             bool expected_success,
                             bool use_ex_filter) {
    if (not index->CheckFeature(vsag::SUPPORT_KNN_ITERATOR_FILTER_SEARCH)) {
        return;
    }
    if (use_ex_filter && not index->CheckFeature(vsag::SUPPORT_KNN_SEARCH_WITH_EX_FILTER)) {
        return;
    }
    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = use_ex_filter ? dataset->ex_filter_ground_truth_ : dataset->filter_ground_truth_;
    auto gt_topK = dataset->top_k;
    float cur_recall = 0.0f;
    auto topk = gt_topK;
    auto filter = std::make_shared<FilterObj>(
        dataset->filter_function_, dataset->ex_filter_function_, dataset->valid_ratio_);
    int64_t first_top = topk / 3;
    int64_t second_top = topk / 3;
    int64_t third_top = topk - first_top - second_top;
    std::vector<int64_t> ids(topk);
    for (auto i = 0; i < query_count; ++i) {
        vsag::IteratorContext* filter_ctx = nullptr;
        auto query = get_one_query(queries, i);
        auto res = index->KnnSearch(query, first_top, search_param, filter, filter_ctx, false);
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
        int64_t get_cnt = res.value()->GetDim();
        REQUIRE(res.value()->GetDim() == first_top);
        memcpy(ids.data(), res.value()->GetIds(), sizeof(int64_t) * first_top);
        auto res2 = index->KnnSearch(query, second_top, search_param, filter, filter_ctx, false);
        REQUIRE(res2.has_value() == expected_success);
        if (!expected_success) {
            return;
        }
        REQUIRE(res2.value()->GetDim() == second_top);
        memcpy(ids.data() + first_top, res2.value()->GetIds(), sizeof(int64_t) * second_top);
        auto res3 = index->KnnSearch(query, third_top, search_param, filter, filter_ctx, false);
        REQUIRE(res3.has_value() == expected_success);
        if (!expected_success) {
            return;
        }
        REQUIRE(res3.value()->GetDim() == third_top);
        memcpy(ids.data() + first_top + second_top,
               res3.value()->GetIds(),
               sizeof(int64_t) * third_top);
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK, ids.data(), topk);
        cur_recall += static_cast<float>(val) / static_cast<float>(gt_topK);
        delete filter_ctx;
    }
    if (cur_recall <= expected_recall * query_count) {
        WARN(fmt::format("cur_result({}) <= expected_recall * query_count({})",
                         cur_recall,
                         expected_recall * query_count));
    }
    REQUIRE(cur_recall > expected_recall * query_count * RECALL_THRESHOLD);
}

void
TestIndex::TestFilterSearch(const TestIndex::IndexPtr& index,
                            const TestDatasetPtr& dataset,
                            const std::string& search_param,
                            float expected_recall,
                            bool expected_success,
                            bool support_filter_obj) {
    if (not index->CheckFeature(vsag::SUPPORT_KNN_SEARCH_WITH_ID_FILTER)) {
        return;
    }
    auto queries = dataset->filter_query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->filter_ground_truth_;
    auto gt_topK = dataset->top_k;
    float cur_recall = 0.0f;
    auto topk = gt_topK;
    for (auto i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        tl::expected<DatasetPtr, vsag::Error> res;
        res = index->KnnSearch(query, topk, search_param, dataset->filter_function_);
        if (support_filter_obj) {
            auto filter = std::make_shared<FilterObj>(
                dataset->filter_function_, dataset->ex_filter_function_, 1.0F);
            auto obj_res = index->KnnSearch(query, topk, search_param, filter);
            if (expected_success) {
                for (int j = 0; j < topk; ++j) {
                    REQUIRE(std::abs(obj_res.value()->GetDistances()[j] -
                                     res.value()->GetDistances()[j]) <= 2e-6);
                }
            }
        }
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
        if (index->CheckFeature(vsag::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER)) {
            auto threshold = res.value()->GetDistances()[topk - 1] + 1e-5;
            auto range_result =
                index->RangeSearch(query, threshold, search_param, dataset->filter_function_);
            REQUIRE(range_result.has_value());
            REQUIRE(range_result.value()->GetDim() > 0);
            if (range_result.value()->GetDim() < topk - 1) {
                WARN(fmt::format("range search result dim {} is less than {}",
                                 range_result.value()->GetDim(),
                                 topk - 1));
            }
        }
        auto result = res.value()->GetIds();
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK, result, topk);
        cur_recall += static_cast<float>(val) / static_cast<float>(gt_topK);
    }
    if (cur_recall <= expected_recall * query_count) {
        WARN(fmt::format("cur_result({}) <= expected_recall * query_count({})",
                         cur_recall,
                         expected_recall * query_count));
    }
    REQUIRE(cur_recall > expected_recall * query_count * RECALL_THRESHOLD);
}

void
TestIndex::TestSearchAllocator(const TestIndex::IndexPtr& index,
                               const TestDatasetPtr& dataset,
                               const std::string& search_param,
                               float expected_recall,
                               bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_KNN_SEARCH)) {
        return;
    }
    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    float cur_recall = 0.0f;
    auto topk = gt_topK;
    class ExampleAllocator : public vsag::Allocator {
    public:
        std::string
        Name() override {
            return "example-allocator";
        }

        void*
        Allocate(uint64_t size) override {
            auto addr = (void*)malloc(size);
            sizes_[addr] = size;
            return addr;
        }

        void
        Deallocate(void* p) override {
            if (sizes_.find(p) == sizes_.end())
                return;
            sizes_.erase(p);
            return free(p);
        }

        void*
        Reallocate(void* p, uint64_t size) override {
            auto addr = (void*)realloc(p, size);
            sizes_.erase(p);
            sizes_[addr] = size;
            return addr;
        }

    private:
        std::unordered_map<void*, uint64_t> sizes_;
    };

    for (auto i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        ExampleAllocator allocator;
        vsag::SearchParam search_params(false, search_param, nullptr, &allocator);
        auto res = index->KnnSearch(query, topk, search_params);
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
        auto dis = res.value()->GetDistances();
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK, result, topk);
        cur_recall += static_cast<float>(val) / static_cast<float>(gt_topK);
        allocator.Deallocate((void*)result);
        allocator.Deallocate((void*)dis);
    }
    if (cur_recall <= expected_recall * query_count) {
        WARN(fmt::format("cur_result({}) <= expected_recall * query_count({})",
                         cur_recall,
                         expected_recall * query_count));
    }
    REQUIRE(cur_recall > expected_recall * query_count * RECALL_THRESHOLD);
}

void
TestIndex::TestSearchWithDirtyVector(const TestIndex::IndexPtr& index,
                                     const TestDatasetPtr& dataset,
                                     const std::string& search_param,
                                     bool expected_success) {
    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    auto topk = gt_topK;
    int valid_query_count = static_cast<int64_t>(query_count * 0.9);
    for (auto i = 0; i < valid_query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        auto res = index->KnnSearch(query, topk, search_param);
        REQUIRE(res.has_value() == expected_success);
        if (!expected_success) {
            return;
        }
        REQUIRE(res.value()->GetDim() == topk);
    }

    const auto& radius = dataset->range_radius_;
    for (auto i = 0; i < valid_query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        if (std::isnan(radius[i])) {
            continue;
        }
        auto res = index->RangeSearch(query, radius[i], search_param);
        REQUIRE(res.has_value() == expected_success);
    }

    for (auto i = valid_query_count; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        auto res = index->KnnSearch(query, topk, search_param);
        REQUIRE(res.has_value() == expected_success);
    }
}

void
TestIndex::TestKnnSearchExFilter(const IndexPtr& index,
                                 const TestDatasetPtr& dataset,
                                 const std::string& search_param,
                                 float expected_recall,
                                 bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_KNN_SEARCH_WITH_EX_FILTER)) {
        return;
    }
    auto queries = dataset->filter_query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ex_filter_ground_truth_;
    auto gt_topK = dataset->top_k;
    auto extra_info_size = dataset->base_->GetExtraInfoSize();
    float cur_recall = 0.0f;
    auto topk = gt_topK;
    auto f = std::make_shared<FilterObj>(dataset->filter_function_, nullptr, dataset->valid_ratio_);
    auto filter = std::make_shared<FilterObj>(
        dataset->filter_function_, dataset->ex_filter_function_, dataset->valid_ratio_);
    for (auto i = 0; i < query_count; ++i) {
        auto query_recall = 0.0f;
        auto query = get_one_query(queries, i);
        auto res = index->KnnSearch(query, topk, search_param, filter);
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
        REQUIRE(res.has_value() == true);
        REQUIRE(res.value()->GetDim() == topk);
        auto result = res.value()->GetIds();
        if (extra_info_size > 0) {
            const char* extra_infos = res.value()->GetExtraInfos();
            REQUIRE(f->CheckValid(extra_infos) == true);
            REQUIRE(extra_infos != nullptr);
            int64_t num = res.value()->GetNumElements();
            for (int j = 0; j < num; ++j) {
                REQUIRE((extra_infos + j * extra_info_size) != nullptr);
            }
        }
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK, result, topk);
        cur_recall += static_cast<float>(val) / static_cast<float>(gt_topK);
    }
    if (cur_recall <= expected_recall * query_count) {
        WARN(fmt::format("cur_result({}) <= expected_recall * query_count({})",
                         cur_recall,
                         expected_recall * query_count));
    }
    REQUIRE(cur_recall > expected_recall * query_count * RECALL_THRESHOLD);
}

void
TestIndex::TestSearchOvertime(const IndexPtr& index,
                              const TestDatasetPtr& dataset,
                              const std::string& search_param) {
    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->SparseVectors(queries->GetSparseVectors() + i)
            ->Paths(queries->GetPaths() + i)
            ->Owner(false);
        auto res = index->KnnSearch(query, 10, search_param);
        REQUIRE(res.has_value());
        auto result = res.value();
        REQUIRE(result->GetStatistics() != "{}");
        auto stats = result->GetStatistics({"is_timeout"});
        REQUIRE(stats.size() == 1);
        bool has_timeout_result = (stats[0] == "true" or stats[0] == "false");
        REQUIRE(has_timeout_result);
    }
}

}  // namespace fixtures
