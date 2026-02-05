
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

#include "test_index.h"

#include "fixtures/memory_record_allocator.h"
#include "fixtures/test_logger.h"
#include "fixtures/test_reader.h"
#include "fixtures/thread_pool.h"
#include "impl/allocator/default_allocator.h"
#include "index/hnsw.h"
#include "simd/fp32_simd.h"
#include "vsag/engine.h"
#include "vsag/resource.h"
#include "vsag/search_param.h"

namespace fixtures {
static int64_t
Intersection(const int64_t* x, int64_t x_count, const int64_t* y, int64_t y_count) {
    std::unordered_set<int64_t> set_x(x, x + x_count);
    int result = 0;

    for (int i = 0; i < y_count; ++i) {
        if (set_x.count(y[i])) {
            ++result;
        }
    }
    return result;
}

vsag::DatasetPtr
get_one_query(const vsag::DatasetPtr& queries, int i) {
    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(queries->GetDim())
        ->Float32Vectors(queries->GetFloat32Vectors() + i * queries->GetDim())
        ->SparseVectors(queries->GetSparseVectors() + i)
        ->Owner(false);
    if (queries->GetPaths() != nullptr) {
        query->Paths(queries->GetPaths() + i);
    }
    return query;
}

void
TestIndex::TestBuildIndex(const IndexPtr& index,
                          const TestDatasetPtr& dataset,
                          bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_BUILD)) {
        return;
    }
    auto build_index = index->Build(dataset->base_);
    if (expected_success) {
        REQUIRE(build_index.has_value());
        // check the number of vectors in index
        REQUIRE(index->GetNumElements() == dataset->base_->GetNumElements());
    } else {
        REQUIRE(build_index.has_value() == expected_success);
    }
}

void
TestIndex::TestAddIndex(const IndexPtr& index,
                        const TestDatasetPtr& dataset,
                        bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_ADD_FROM_EMPTY)) {
        return;
    }
    auto add_index = index->Add(dataset->base_);
    if (expected_success) {
        REQUIRE(add_index.has_value());
        // check the number of vectors in index
        REQUIRE(index->GetNumElements() == dataset->base_->GetNumElements());
    } else {
        REQUIRE(not add_index.has_value());
    }
}

void
TestIndex::TestUpdateId(const IndexPtr& index,
                        const TestDatasetPtr& dataset,
                        const std::string& search_param,
                        bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_UPDATE_ID_CONCURRENT)) {
        return;
    }
    auto ids = dataset->base_->GetIds();
    auto num_vectors = dataset->base_->GetNumElements();
    auto dim = dataset->base_->GetDim();
    auto gt_topK = dataset->top_k;
    auto base = dataset->base_->GetFloat32Vectors();

    std::unordered_map<int64_t, int64_t> update_id_map;
    std::unordered_map<int64_t, int64_t> reverse_id_map;
    int64_t max_id = num_vectors;
    for (int i = 0; i < num_vectors; i++) {
        if (ids[i] > max_id) {
            max_id = ids[i];
        }
    }
    for (int i = 0; i < num_vectors; i++) {
        update_id_map[ids[i]] = ids[i] + 2 * max_id;
    }

    std::vector<int> correct_num = {0, 0};
    for (int round = 0; round < 2; round++) {
        // round 0 for update, round 1 for validate update results
        for (int i = 0; i < num_vectors; i++) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)
                ->Dim(dim)
                ->Float32Vectors(base + i * dim)
                ->SparseVectors(dataset->base_->GetSparseVectors() + i)
                ->Owner(false);

            auto result = index->KnnSearch(query, gt_topK, search_param);
            REQUIRE(result.has_value());

            if (round == 0) {
                if (result.value()->GetIds()[0] == ids[i]) {
                    correct_num[round] += 1;
                }

                auto succ_update_res = index->UpdateId(ids[i], update_id_map[ids[i]]);
                REQUIRE(succ_update_res.has_value());
                if (expected_success) {
                    if (index->CheckFeature(vsag::IndexFeature::SUPPORT_CHECK_ID_EXIST)) {
                        REQUIRE(index->CheckIdExist(ids[i]) == false);
                        REQUIRE(index->CheckIdExist(update_id_map[ids[i]]) == true);
                    }
                    REQUIRE(succ_update_res.value());
                }

                // old id don't exist
                auto failed_old_res = index->UpdateId(ids[i], update_id_map[ids[i]]);
                REQUIRE(not failed_old_res.has_value());

                // same id
                auto succ_same_res = index->UpdateId(update_id_map[ids[i]], update_id_map[ids[i]]);
                REQUIRE(succ_same_res.has_value());
                REQUIRE(succ_same_res.value());
            } else {
                if (result.value()->GetIds()[0] == update_id_map[ids[i]]) {
                    correct_num[round] += 1;
                }
            }
        }

        for (int i = 0; i < num_vectors; i++) {
            if (round == 0) {
                // new id is used
                auto failed_new_res =
                    index->UpdateId(update_id_map[ids[i]], update_id_map[ids[num_vectors - i - 1]]);
                REQUIRE(not failed_new_res.has_value());
            }
        }
    }

    REQUIRE(correct_num[0] == correct_num[1]);
}

void
TestIndex::TestUpdateVectorSparse(const IndexPtr& index,
                                  const TestDatasetPtr& dataset,
                                  bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_UPDATE_VECTOR_CONCURRENT)) {
        return;
    }
    auto ids = dataset->base_->GetIds();
    auto num_vectors = dataset->base_->GetNumElements();
    auto base = dataset->base_->GetSparseVectors();

    for (int i = 0; i < num_vectors; i++) {
        auto far_base = vsag::Dataset::Make();
        auto close_base = vsag::Dataset::Make();
        close_base->NumElements(1)->SparseVectors(base + i)->Owner(false);
        far_base->NumElements(1)
            ->SparseVectors(base + ((i + num_vectors / 2) % num_vectors))
            ->Owner(false);

        // [step 1] success case with <close> vector
        auto dist_before = index->CalcDistanceById(close_base, ids[i]).value();
        auto update_res = index->UpdateVector(ids[i], close_base);
        auto dist_after = index->CalcDistanceById(close_base, ids[i]).value();
        REQUIRE(update_res.has_value());
        REQUIRE(update_res.value());
        REQUIRE(std::abs(dist_before - dist_after) < 1e-3);

        // [step 2] update with <far> vector
        auto far_update_res = index->UpdateVector(ids[i], far_base);
        REQUIRE(far_update_res.has_value());
        REQUIRE_FALSE(far_update_res.value());
    }
}

void
TestIndex::TestUpdateVector(const IndexPtr& index,
                            const TestDatasetPtr& dataset,
                            const std::string& search_param,
                            bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_UPDATE_VECTOR_CONCURRENT)) {
        return;
    }
    auto ids = dataset->base_->GetIds();
    auto num_vectors = dataset->base_->GetNumElements();
    auto dim = dataset->base_->GetDim();
    auto gt_topK = dataset->top_k;
    auto base = dataset->base_->GetFloat32Vectors();

    int64_t max_id = num_vectors;
    for (int i = 0; i < num_vectors; i++) {
        if (ids[i] > max_id) {
            max_id = ids[i];
        }
    }

    uint32_t count_fail_close_update = 0;
    std::vector<int> correct_num = {0, 0};
    uint32_t success_far_updated = 0, failed_far_updated = 0;
    std::mt19937 rng(42);
    for (int round = 0; round < 2; round++) {
        // round 0 for test original recall
        // round 1 for test updated  recall
        for (int i = 0; i < num_vectors; i++) {
            auto close_base = vsag::Dataset::Make();
            auto far_base = vsag::Dataset::Make();
            auto original_base = vsag::Dataset::Make();
            original_base->NumElements(1)->Dim(dim)->Float32Vectors(base + i * dim)->Owner(false);

            if (round == 1) {
                // [step 0] prepare data
                std::vector<float> close_vec(dim);
                std::vector<float> far_vec(dim);
                std::normal_distribution<float> close_distribution(0.0F, 0.001F);
                std::normal_distribution<float> far_distribution(0.0F, 1.0F);
                for (int d = 0; d < dim; d++) {
                    close_vec[d] = base[i * dim + d] + close_distribution(rng);
                    far_vec[d] = base[i * dim + d] + far_distribution(rng);
                }

                close_base->NumElements(1)
                    ->Dim(dim)
                    ->Float32Vectors(close_vec.data())
                    ->Owner(false);

                far_base->NumElements(1)->Dim(dim)->Float32Vectors(far_vec.data())->Owner(false);

                // [step 0] prepare dist
                float dist_original_original = 0;
                float dist_original_close = 0;
                float dist_original_far = 0;
                float dist_original_original_recovery = 0;

                // [step 1] success case with <close> vector
                dist_original_original = *index->CalcDistanceById(base + i * dim, ids[i]);
                auto close_update_res = index->UpdateVector(ids[i], close_base);
                dist_original_close = *index->CalcDistanceById(base + i * dim, ids[i]);
                if (not close_update_res.has_value() or not close_update_res.value()) {
                    count_fail_close_update++;
                }

                // [step 2] update with <far> vector
                auto far_update_res = index->UpdateVector(ids[i], far_base);
                REQUIRE(far_update_res.has_value());
                if (far_update_res.value()) {
                    // note that the update should be failed, but for some cases, it success
                    success_far_updated++;
                } else {
                    failed_far_updated++;
                }

                // [step 3] force update with <far> vector
                auto force_far_update_res = index->UpdateVector(ids[i], far_base, true);
                REQUIRE(force_far_update_res.has_value());
                REQUIRE(force_far_update_res.value());
                dist_original_far = *index->CalcDistanceById(base + i * dim, ids[i]);
                if (expected_success) {
                    REQUIRE(dist_original_close <= dist_original_far);
                }

                // [step 4] update back with <original> vector
                auto force_original_update_res = index->UpdateVector(ids[i], original_base, true);
                REQUIRE(force_original_update_res.has_value());
                REQUIRE(force_original_update_res.value());
                dist_original_original_recovery = *index->CalcDistanceById(base + i * dim, ids[i]);
                REQUIRE(std::abs(dist_original_original - dist_original_original_recovery) < 1e-3);

                // [fail case] old id don't exist
                auto failed_old_res = index->UpdateVector(ids[i] + 2 * max_id, close_base);
                REQUIRE(not failed_old_res.has_value());
            }

            // [step 0] test recall
            auto result = index->KnnSearch(original_base, gt_topK, search_param);
            REQUIRE(result.has_value());

            if (result.value()->GetIds()[0] == ids[i]) {
                correct_num[round] += 1;
            }
        }
    }

    REQUIRE(count_fail_close_update < 10);
    REQUIRE(correct_num[0] == correct_num[1]);
    if (expected_success) {
        REQUIRE(success_far_updated < failed_far_updated);
    }
}

void
TestIndex::TestContinueAdd(const IndexPtr& index,
                           const TestDatasetPtr& dataset,
                           bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_ADD_AFTER_BUILD)) {
        return;
    }
    auto base_count = dataset->base_->GetNumElements();
    int64_t temp_count = std::max<int64_t>(1, dataset->base_->GetNumElements() / 2);
    auto dim = dataset->base_->GetDim();
    auto temp_dataset = vsag::Dataset::Make();
    temp_dataset->Dim(dim)
        ->Ids(dataset->base_->GetIds())
        ->NumElements(temp_count)
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Paths(dataset->base_->GetPaths())
        ->SparseVectors(dataset->base_->GetSparseVectors())
        ->Owner(false);
    index->Build(temp_dataset);
    for (uint64_t j = temp_count; j < base_count; ++j) {
        auto data_one = vsag::Dataset::Make();
        data_one->Dim(dim)
            ->Ids(dataset->base_->GetIds() + j)
            ->NumElements(1)
            ->Float32Vectors(dataset->base_->GetFloat32Vectors() + j * dim)
            ->Paths(dataset->base_->GetPaths() + j)
            ->SparseVectors(dataset->base_->GetSparseVectors() + j)
            ->Owner(false);
        auto add_index = index->Add(data_one);
        if (expected_success) {
            REQUIRE(add_index.has_value());
            // check the number of vectors in index
            REQUIRE(index->GetNumElements() == (j + 1));
        } else {
            REQUIRE(not add_index.has_value());
        }
    }
}

void
TestIndex::TestTrainAndAdd(const TestIndex::IndexPtr& index,
                           const TestDatasetPtr& dataset,
                           bool expected_success) {
    auto base_count = dataset->base_->GetNumElements();
    int64_t temp_count =
        std::max<int64_t>(1, static_cast<int64_t>(dataset->base_->GetNumElements() * 0.8));
    auto dim = dataset->base_->GetDim();
    auto temp_dataset = vsag::Dataset::Make();
    temp_dataset->Dim(dim)
        ->Ids(dataset->base_->GetIds())
        ->NumElements(temp_count)
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Paths(dataset->base_->GetPaths())
        ->SparseVectors(dataset->base_->GetSparseVectors())
        ->Owner(false);
    index->Train(dataset->base_);
    auto result_add = index->Add(temp_dataset);
    REQUIRE(result_add.has_value());
    for (uint64_t j = temp_count; j < base_count; ++j) {
        auto data_one = vsag::Dataset::Make();
        data_one->Dim(dim)
            ->Ids(dataset->base_->GetIds() + j)
            ->NumElements(1)
            ->Float32Vectors(dataset->base_->GetFloat32Vectors() + j * dim)
            ->Paths(dataset->base_->GetPaths() + j)
            ->SparseVectors(dataset->base_->GetSparseVectors() + j)
            ->Owner(false);
        auto add_index = index->Add(data_one);
        if (expected_success) {
            REQUIRE(add_index.has_value());
            // check the number of vectors in index
            REQUIRE(index->GetNumElements() == (j + 1));
        } else {
            REQUIRE(not add_index.has_value());
        }
    }
}

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
        auto result = res.value()->GetIds();
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK - 1, result, res.value()->GetDim());
        cur_recall += static_cast<float>(val) /
                      static_cast<float>(std::min(gt_topK - 1, res.value()->GetDim()));
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
TestIndex::TestCalcDistanceById(const IndexPtr& index,
                                const TestDatasetPtr& dataset,
                                float error,
                                bool expected_success,
                                bool is_sparse) {
    if (not index->CheckFeature(vsag::SUPPORT_CAL_DISTANCE_BY_ID)) {
        return;
    }
    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    for (auto i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        for (auto j = 0; j < gt_topK; ++j) {
            auto id = gts->GetIds()[i * gt_topK + j];
            auto dist = gts->GetDistances()[i * gt_topK + j];
            tl::expected<float, vsag::Error> result;
            if (is_sparse) {
                result = index->CalcDistanceById(query, id);
            } else {
                result = index->CalcDistanceById(query->GetFloat32Vectors(), id);
            }
            if (not expected_success) {
                continue;
            }
            REQUIRE(result.has_value());
            float estimate_dist = result.value();
            REQUIRE(std::abs(dist - estimate_dist) < error);
        }
    }
}

void
TestIndex::TestBatchCalcDistanceById(const IndexPtr& index,
                                     const TestDatasetPtr& dataset,
                                     float error,
                                     bool expected_success,
                                     bool is_sparse,
                                     bool is_old_index) {
    if (not index->CheckFeature(vsag::SUPPORT_CAL_DISTANCE_BY_ID)) {
        return;
    }
    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    for (auto i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        tl::expected<DatasetPtr, vsag::Error> result;
        if (is_sparse) {
            result = index->CalDistanceById(query, gts->GetIds() + (i * gt_topK), gt_topK);
        } else {
            result = index->CalDistanceById(
                query->GetFloat32Vectors(), gts->GetIds() + (i * gt_topK), gt_topK);
            if (is_old_index) {
                // for old index (hnsw and diskann)
                REQUIRE_THROWS(index->CalDistanceById(query, gts->GetIds() + (i * gt_topK), gt_topK)
                                   .has_value());
            } else {
                // for new index (from inner_index_interface)
                REQUIRE_FALSE(index->CalDistanceById(query, gts->GetIds() + (i * gt_topK), gt_topK)
                                  .has_value());
            }
        }
        if (not expected_success) {
            return;
        }
        for (auto j = 0; j < gt_topK; ++j) {
            REQUIRE(std::abs(gts->GetDistances()[i * gt_topK + j] -
                             result.value()->GetDistances()[j]) < error);
        }
    }
    SECTION("test non-existing id") {
        int64_t test_num = 10;
        std::vector<int64_t> no_exist_ids(test_num);
        for (int i = 0; i < test_num; ++i) {
            no_exist_ids[i] = -i - 1;
        }
        tl::expected<DatasetPtr, vsag::Error> result;
        queries->NumElements(1);
        if (is_sparse) {
            result = index->CalDistanceById(queries, no_exist_ids.data(), test_num);
        } else {
            result =
                index->CalDistanceById(queries->GetFloat32Vectors(), no_exist_ids.data(), test_num);
        }
        for (int i = 0; i < test_num; ++i) {
            fixtures::dist_t dist = result.value()->GetDistances()[i];
            REQUIRE(dist == -1);
        }
        queries->NumElements(query_count);
    }
}

void
TestIndex::TestGetMinAndMaxId(const IndexPtr& index,
                              const TestDatasetPtr& dataset,
                              bool expected_success) {
    auto base_count = dataset->base_->GetNumElements();
    auto dim = dataset->base_->GetDim();
    auto get_min_max_res = index->GetMinAndMaxId();
    if (not expected_success) {
        REQUIRE_FALSE(get_min_max_res.has_value());
        return;
    }
    REQUIRE(get_min_max_res.has_value() == (index->GetNumElements() > 0));
    int64_t res_max_id = INT64_MIN;
    int64_t res_min_id = INT64_MAX;
    for (uint64_t j = 0; j < base_count; ++j) {
        res_max_id =
            res_max_id > dataset->base_->GetIds()[j] ? res_max_id : dataset->base_->GetIds()[j];
        res_min_id =
            res_min_id < dataset->base_->GetIds()[j] ? res_min_id : dataset->base_->GetIds()[j];
    }
    get_min_max_res = index->GetMinAndMaxId();
    REQUIRE(get_min_max_res.has_value() == true);
    int64_t min_id = get_min_max_res.value().first;
    int64_t max_id = get_min_max_res.value().second;

    REQUIRE(min_id == res_min_id);
    REQUIRE(max_id == res_max_id);
}

void
TestIndex::TestSerializeFile(const IndexPtr& index_from,
                             const IndexPtr& index_to,
                             const TestDatasetPtr& dataset,
                             const std::string& search_param,
                             bool expected_success) {
    if (not index_from->CheckFeature(vsag::SUPPORT_SERIALIZE_FILE) or
        not index_to->CheckFeature(vsag::SUPPORT_DESERIALIZE_FILE)) {
        return;
    }
    auto dir = fixtures::TempDir("serialize");
    auto path = dir.GenerateRandomFile();
    std::ofstream outfile(path, std::ios::out | std::ios::binary);
    auto serialize_index = index_from->Serialize(outfile);
    REQUIRE(serialize_index.has_value() == expected_success);
    outfile.close();

    std::ifstream infile(path, std::ios::in | std::ios::binary);
    auto deserialize_index = index_to->Deserialize(infile);
    REQUIRE(deserialize_index.has_value() == expected_success);
    infile.close();
    if (index_to->GetNumElements() == 0) {
        return;
    }

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto topk = 10;
    for (auto i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        auto res_from = index_from->KnnSearch(query, topk, search_param);
        auto res_to = index_to->KnnSearch(query, topk, search_param);
        REQUIRE(res_from.has_value());
        REQUIRE(res_to.has_value());
        REQUIRE(res_from.value()->GetDim() == res_to.value()->GetDim());
        int64_t result_count = res_from.value()->GetDim();
        for (int64_t j = 0; j < result_count; ++j) {
            REQUIRE(std::abs(res_from.value()->GetDistances()[j] -
                             res_to.value()->GetDistances()[j]) <= 2e-6);
        }
    }
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
TestIndex::TestSerializeBinarySet(const IndexPtr& index_from,
                                  const IndexPtr& index_to,
                                  const TestDatasetPtr& dataset,
                                  const std::string& search_param,
                                  bool expected_success) {
    if (not index_from->CheckFeature(vsag::SUPPORT_SERIALIZE_BINARY_SET) or
        not index_to->CheckFeature(vsag::SUPPORT_DESERIALIZE_BINARY_SET)) {
        return;
    }
    auto serialize_binary = index_from->Serialize();
    REQUIRE(serialize_binary.has_value() == expected_success);

    auto deserialize_index = index_to->Deserialize(serialize_binary.value());
    REQUIRE(deserialize_index.has_value() == expected_success);
    if (index_to->GetNumElements() == 0) {
        return;
    }

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto topk = 10;
    for (auto i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        auto res_from = index_from->KnnSearch(query, topk, search_param);
        auto res_to = index_to->KnnSearch(query, topk, search_param);
        REQUIRE(res_from.has_value());
        REQUIRE(res_to.has_value());
        REQUIRE(res_from.value()->GetDim() == res_to.value()->GetDim());
        int64_t result_count = res_from.value()->GetDim();
        for (int64_t j = 0; j < result_count; ++j) {
            REQUIRE(res_to.value()->GetIds()[j] == res_from.value()->GetIds()[j]);
        }
    }
}

void
TestIndex::TestSerializeReaderSet(const IndexPtr& index_from,
                                  const IndexPtr& index_to,
                                  const TestDatasetPtr& dataset,
                                  const std::string& search_param,
                                  const std::string& index_name,
                                  bool expected_success) {
    if (not index_from->CheckFeature(vsag::SUPPORT_SERIALIZE_BINARY_SET) or
        not index_to->CheckFeature(vsag::SUPPORT_DESERIALIZE_READER_SET)) {
        return;
    }
    vsag::ReaderSet rs;
    auto serialize_binary = index_from->Serialize();
    REQUIRE(serialize_binary.has_value() == expected_success);
    auto binary_set = serialize_binary.value();
    for (const auto& key : binary_set.GetKeys()) {
        rs.Set(key, std::make_shared<TestReader>(binary_set.Get(key)));
    }
    REQUIRE(rs.Get("this_is_a_wrong_key") == nullptr);
    auto deserialize_index = index_to->Deserialize(rs);
    REQUIRE(deserialize_index.has_value() == expected_success);
    if (index_to->GetNumElements() == 0) {
        return;
    }

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto topk = 10;
    for (auto i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        auto res_from = index_from->KnnSearch(query, topk, search_param);
        auto res_to = index_to->KnnSearch(query, topk, search_param);
        REQUIRE(res_from.has_value());
        REQUIRE(res_to.has_value());
        REQUIRE(res_from.value()->GetDim() == res_to.value()->GetDim());
        int64_t result_count = res_from.value()->GetDim();
        for (int64_t j = 0; j < result_count; ++j) {
            REQUIRE(res_to.value()->GetIds()[j] == res_from.value()->GetIds()[j]);
        }
    }
}

void
TestIndex::TestSerializeWriteFunc(const IndexPtr& index_from,
                                  const IndexPtr& index_to,
                                  const TestDatasetPtr& dataset,
                                  const std::string& search_param,
                                  bool expected_success) {
    if (not index_from->CheckFeature(vsag::SUPPORT_SERIALIZE_WRITE_FUNC)) {
        return;
    }
    auto dir = fixtures::TempDir("serialize");
    auto path = dir.GenerateRandomFile();
    std::ofstream outfile(path, std::ios::out | std::ios::binary);
    vsag::WriteFuncType write_func =
        [&outfile](vsag::OffsetType offset, vsag::SizeType size, const void* data) -> void {
        outfile.seekp(offset);
        outfile.write(reinterpret_cast<const char*>(data), size);
    };
    auto serialize_index = index_from->Serialize(write_func);
    REQUIRE(serialize_index.has_value() == expected_success);
    outfile.close();

    std::ifstream infile(path, std::ios::in | std::ios::binary);
    auto deserialize_index = index_to->Deserialize(infile);
    REQUIRE(deserialize_index.has_value() == expected_success);
    infile.close();
    if (index_to->GetNumElements() == 0) {
        return;
    }

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto topk = 10;
    for (auto i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        auto res_from = index_from->KnnSearch(query, topk, search_param);
        auto res_to = index_to->KnnSearch(query, topk, search_param);
        REQUIRE(res_from.has_value());
        REQUIRE(res_to.has_value());
        REQUIRE(res_from.value()->GetDim() == res_to.value()->GetDim());
        int64_t result_count = res_from.value()->GetDim();
        for (int64_t j = 0; j < result_count; ++j) {
            REQUIRE(res_to.value()->GetIds()[j] == res_from.value()->GetIds()[j]);
        }
    }
}

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
TestIndex::TestConcurrentDestruct(TestIndex::IndexPtr& index,
                                  const TestDatasetPtr& dataset,
                                  const std::string& search_param) {
    std::vector<std::future<bool>> futures;
    fixtures::ThreadPool pool(32);
    std::shared_mutex index_mutex;

    auto func = [&](uint64_t i) -> bool {
        auto base = vsag::Dataset::Make();
        base->NumElements(1)
            ->Ids(dataset->base_->GetIds() + i)
            ->Dim(dataset->base_->GetDim())
            ->Float32Vectors(dataset->base_->GetFloat32Vectors() + i * dataset->base_->GetDim())
            ->Owner(false);

        if (i == (dataset->base_->GetNumElements() * 3) / 4) {
            std::unique_lock status_lock(index_mutex);
            std::dynamic_pointer_cast<vsag::HNSW>(index)->SetStatus(vsag::VSAGIndexStatus::ALIVE);
            index.reset();
            return true;
        }

        std::shared_lock status_lock(index_mutex);
        if (not index) {
            return false;
        }

        switch (random() % 22) {
            case 0:
                return index->Build(base).has_value();
            case 1:
                return index->Add(base).has_value();
            case 2:
                return index->Remove(*base->GetIds()).has_value();
            case 3:
                return index->UpdateId(*base->GetIds(), *base->GetIds() + 1).has_value();
            case 4:
                return index->UpdateVector(*base->GetIds(), base).has_value();
            case 5:
                return index->KnnSearch(base, 100, search_param).has_value();
            case 6:
                return index->RangeSearch(base, 100, search_param).has_value();
            case 7:
                return index->Feedback(base, 100, search_param).has_value();
            case 8:
                return index->Pretrain({*base->GetIds()}, 100, search_param).has_value();
            case 9:
                return index->CalcDistanceById(base->GetFloat32Vectors(), *base->GetIds())
                    .has_value();
            case 10:
                return index->CalDistanceById(base->GetFloat32Vectors(), base->GetIds(), 1)
                    .has_value();
            case 11:
                return index->GetMinAndMaxId().has_value();
            case 12:
                return index->Serialize().has_value();
            case 13: {
                std::ostringstream oss;
                std::ostream& out = oss;
                return index->Serialize(out).has_value();
            }
            case 14: {
                vsag::BinarySet bs;
                return index->Deserialize(bs).has_value();
            }
            case 15: {
                vsag::ReaderSet rs;
                return index->Deserialize(rs).has_value();
            }
            case 16: {
                std::istringstream iss;
                std::istream& in = iss;
                return index->Deserialize(in).has_value();
            }
            case 17:
                return index->Merge({}).has_value();
            case 18:
                return index->CheckFeature(vsag::IndexFeature::SUPPORT_BUILD);
            case 19:
                return index->CheckIdExist(*base->GetIds());
            case 20:
                return index->GetMemoryUsage() > 0;
            case 21:
                std::dynamic_pointer_cast<vsag::HNSW>(index)->SetStatus(
                    vsag::VSAGIndexStatus::DESTROYED);
                return true;
            default:
                std::dynamic_pointer_cast<vsag::HNSW>(index)->SetStatus(
                    vsag::VSAGIndexStatus::ALIVE);
                return true;
        }
    };

    for (uint64_t i = 0; i < dataset->base_->GetNumElements(); i++) {
        futures.emplace_back(pool.enqueue(func, i));
    }
}

void
TestIndex::TestContinueAddIgnoreRequire(const TestIndex::IndexPtr& index,
                                        const TestDatasetPtr& dataset,
                                        float build_ratio) {
    auto base_count = dataset->base_->GetNumElements();
    int64_t temp_count = static_cast<int64_t>(base_count * build_ratio);
    auto dim = dataset->base_->GetDim();
    auto temp_dataset = vsag::Dataset::Make();
    temp_dataset->Dim(dim)
        ->Ids(dataset->base_->GetIds())
        ->NumElements(temp_count)
        ->Paths(dataset->base_->GetPaths())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);
    index->Build(temp_dataset);
    for (uint64_t j = temp_count; j < base_count; ++j) {
        auto data_one = vsag::Dataset::Make();
        data_one->Dim(dim)
            ->Ids(dataset->base_->GetIds() + j)
            ->NumElements(1)
            ->Paths(dataset->base_->GetPaths() + j)
            ->Float32Vectors(dataset->base_->GetFloat32Vectors() + j * dim)
            ->Owner(false);
        auto add_index = index->Add(data_one);
    }
}
void
TestIndex::TestDuplicateAdd(const TestIndex::IndexPtr& index, const TestDatasetPtr& dataset) {
    auto double_dataset = dataset->base_->DeepCopy();
    double_dataset->Append(dataset->base_);
    uint64_t base_count = dataset->base_->GetNumElements();

    auto check_func = [&](std::vector<int64_t>& failed_ids) -> void {
        REQUIRE(failed_ids.size() == base_count);
        std::sort(failed_ids.begin(), failed_ids.end());
        for (uint64_t i = 0; i < base_count; ++i) {
            REQUIRE(failed_ids[i] == dataset->base_->GetIds()[i]);
        }
    };

    // add once with duplicate;
    auto add_index = index->Build(double_dataset);
    REQUIRE(add_index.has_value());
    check_func(add_index.value());

    // add twice with duplicate;
    auto add_index_2 = index->Add(dataset->base_);
    REQUIRE(add_index_2.has_value());
    check_func(add_index_2.value());
}

void
TestIndex::TestEstimateMemory(const std::string& index_name,
                              const std::string& build_param,
                              const TestDatasetPtr& dataset) {
    auto allocator = std::make_shared<fixtures::MemoryRecordAllocator>();
    {
        vsag::Resource resource(allocator.get(), nullptr);
        vsag::Engine engine(&resource);
        auto index1 = engine.CreateIndex(index_name, build_param).value();
        REQUIRE(index1->GetNumElements() == 0);
        if (index1->CheckFeature(vsag::SUPPORT_ESTIMATE_MEMORY)) {
            auto data_size = dataset->base_->GetNumElements();
            auto estimate_memory = index1->EstimateMemory(data_size);
            auto build_index = index1->Build(dataset->base_);
            auto real_memory = allocator->GetCurrentMemory();
            if (estimate_memory <= static_cast<uint64_t>(real_memory * 0.8) or
                estimate_memory >= static_cast<uint64_t>(real_memory * 1.2)) {
                WARN(fmt::format("estimate_memory({}) is not in range [{}, {}]",
                                 estimate_memory,
                                 static_cast<uint64_t>(real_memory * 0.8),
                                 static_cast<uint64_t>(real_memory * 1.2)));
            }

            REQUIRE(estimate_memory >= static_cast<uint64_t>(real_memory * 0.1));
            REQUIRE(estimate_memory <= static_cast<uint64_t>(real_memory * 5.0));
        }
    }
}

void
TestIndex::TestGetMemoryUsage(const std::string& index_name,
                              const std::string& build_param,
                              const TestDatasetPtr& dataset) {
    auto allocator = std::make_shared<fixtures::MemoryRecordAllocator>();
    {
        vsag::Resource resource(allocator.get(), nullptr);
        vsag::Engine engine(&resource);
        auto index1 = engine.CreateIndex(index_name, build_param).value();
        REQUIRE(index1->GetNumElements() == 0);
        auto index2 = vsag::Factory::CreateIndex(index_name, build_param).value();
        REQUIRE(index2->GetNumElements() == 0);
        fixtures::TempDir dir("index");
        auto path = dir.GenerateRandomFile();
        if (index1->CheckFeature(vsag::SUPPORT_GET_MEMORY_USAGE)) {
            auto data_size = dataset->base_->GetNumElements();
            auto build_index = index2->Build(dataset->base_);
            REQUIRE(build_index.has_value());
            std::ofstream outf(path, std::ios::binary);
            index2->Serialize(outf);
            outf.close();
            std::ifstream inf(path, std::ios::binary);
            index1->Deserialize(inf);
            auto real_memory = allocator->GetCurrentMemory();
            auto get_memory = index1->GetMemoryUsage();

            if (get_memory <= static_cast<uint64_t>(real_memory * 0.8) or
                get_memory >= static_cast<uint64_t>(real_memory * 1.2)) {
                WARN(fmt::format("get_memory({}) is not in range [{}, {}]",
                                 get_memory,
                                 static_cast<uint64_t>(real_memory * 0.8),
                                 static_cast<uint64_t>(real_memory * 1.2)));
            }

            REQUIRE(get_memory >= static_cast<uint64_t>(real_memory * 0.2));
            REQUIRE(get_memory <= static_cast<uint64_t>(real_memory * 3.2));
            inf.close();
        }
    }
}

void
TestIndex::TestCheckIdExist(const TestIndex::IndexPtr& index,
                            const TestDatasetPtr& dataset,
                            bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_CHECK_ID_EXIST)) {
        return;
    }
    auto data_count = dataset->base_->GetNumElements();
    auto* ids = dataset->base_->GetIds();
    int N = 10;
    for (int i = 0; i < N; ++i) {
        auto good_id = ids[random() % data_count];
        REQUIRE(index->CheckIdExist(good_id) == expected_success);
    }
    std::unordered_set<int64_t> exist_ids(ids, ids + data_count);
    int bad_id = 97;
    while (N > 0) {
        for (; bad_id < data_count * N; ++bad_id) {
            if (exist_ids.count(bad_id) == 0) {
                break;
            }
        }
        REQUIRE(index->CheckIdExist(bad_id) == false);
        --N;
    }
}
TestIndex::IndexPtr
TestIndex::TestMergeIndex(const std::string& name,
                          const std::string& build_param,
                          const TestDatasetPtr& dataset,
                          int32_t split_num,
                          bool expect_success) {
    auto create_index_result = vsag::Factory::CreateIndex(name, build_param);
    REQUIRE(create_index_result.has_value() == expect_success);
    auto index = create_index_result.value();
    if (not index->CheckFeature(vsag::SUPPORT_MERGE_INDEX)) {
        return nullptr;
    }

    auto& raw_data = dataset->base_;
    std::vector<vsag::DatasetPtr> sub_datasets;
    int64_t all_data_num = raw_data->GetNumElements();
    int64_t data_dim = raw_data->GetDim();
    const float* vectors = raw_data->GetFloat32Vectors();  // shape = (all_data_num, data_dim)
    const int64_t* ids = raw_data->GetIds();               // shape = (all_data_num)

    int64_t subset_size = all_data_num / split_num;
    int64_t remaining = all_data_num % split_num;

    int64_t start_index = 0;

    for (int64_t i = 0; i < split_num; ++i) {
        int64_t current_subset_size = subset_size + (i < remaining ? 1 : 0);
        auto subset = vsag::Dataset::Make();
        subset->Float32Vectors(vectors + start_index * data_dim);
        subset->Ids(ids + start_index);
        subset->NumElements(current_subset_size);
        subset->Dim(data_dim);
        subset->Owner(false);
        sub_datasets.push_back(subset);
        start_index += current_subset_size;
    }

    std::vector<vsag::MergeUnit> merge_units;
    for (auto sub_dataset : sub_datasets) {
        auto new_index_result = vsag::Factory::CreateIndex(name, build_param);
        REQUIRE(new_index_result.has_value() == expect_success);
        auto new_index = new_index_result.value();
        new_index->Build(sub_dataset);
        vsag::IdMapFunction id_map = [](int64_t id) -> std::tuple<bool, int64_t> {
            return std::make_tuple(true, id);
        };
        merge_units.push_back({new_index, id_map});
    }
    auto merge_result = index->Merge(merge_units);
    REQUIRE(merge_result.has_value());
    return index;
}

TestIndex::IndexPtr
TestIndex::TestMergeIndexWithSameModel(const TestIndex::IndexPtr& model,
                                       const TestDatasetPtr& dataset,
                                       int32_t split_num,
                                       bool expect_success) {
    if (not model->CheckFeature(vsag::SUPPORT_MERGE_INDEX)) {
        return nullptr;
    }
    if (not model->CheckFeature(vsag::SUPPORT_CLONE)) {
        return nullptr;
    }
    auto& raw_data = dataset->base_;
    std::vector<vsag::DatasetPtr> sub_datasets;
    int64_t all_data_num = raw_data->GetNumElements();
    int64_t data_dim = raw_data->GetDim();
    const float* vectors = raw_data->GetFloat32Vectors();  // shape = (all_data_num, data_dim)
    const int64_t* ids = raw_data->GetIds();               // shape = (all_data_num)
    int64_t subset_size = all_data_num / split_num;
    int64_t remaining = all_data_num % split_num;

    int64_t start_index = 0;

    for (int64_t i = 0; i < split_num; ++i) {
        int64_t current_subset_size = subset_size + (i < remaining ? 1 : 0);
        auto subset = vsag::Dataset::Make();
        subset->Float32Vectors(vectors + start_index * data_dim);
        subset->Ids(ids + start_index);
        subset->NumElements(current_subset_size);
        subset->Dim(data_dim);
        subset->Owner(false);
        sub_datasets.push_back(subset);
        start_index += current_subset_size;
    }
    std::vector<vsag::MergeUnit> merge_units;
    for (auto sub_dataset : sub_datasets) {
        auto new_index_result = model->Clone();
        REQUIRE(new_index_result.has_value() == expect_success);
        auto new_index = new_index_result.value();
        auto result_new = new_index->Add(sub_dataset);
        REQUIRE(result_new.has_value());
        vsag::IdMapFunction id_map = [](int64_t id) -> std::tuple<bool, int64_t> {
            return std::make_tuple(true, id);
        };
        merge_units.push_back({new_index, id_map});
    }
    auto index_result = model->Clone();
    REQUIRE(index_result.has_value() == expect_success);
    auto index = index_result.value();
    auto merge_result = index->Merge(merge_units);
    REQUIRE(merge_result.has_value());
    return index;
}

void
TestIndex::TestGetExtraInfoById(const TestIndex::IndexPtr& index,
                                const TestDatasetPtr& dataset,
                                int64_t extra_info_size) {
    if (not index->CheckFeature(vsag::SUPPORT_GET_EXTRA_INFO_BY_ID)) {
        return;
    }
    int64_t count = dataset->count_;
    std::vector<int64_t> ids(count);
    memcpy(ids.data(), dataset->base_->GetIds(), count * sizeof(int64_t));
    std::shuffle(ids.begin(), ids.end(), std::default_random_engine());
    std::vector<char> extra_infos(count * extra_info_size);
    auto result = index->GetExtraInfoByIds(ids.data(), count, extra_infos.data());
    REQUIRE(result.has_value());
    for (int64_t i = 0; i < count; ++i) {
        REQUIRE(memcmp(extra_infos.data() + i * extra_info_size,
                       dataset->base_->GetExtraInfos() +
                           (ids[i] >> dataset->id_shift) * extra_info_size,
                       extra_info_size) == 0);
    }
}

void
TestIndex::TestUpdateExtraInfo(const TestIndex::IndexPtr& index,
                               const TestDatasetPtr& dataset,
                               int64_t extra_info_size) {
    if (not index->CheckFeature(vsag::SUPPORT_UPDATE_EXTRA_INFO_CONCURRENT)) {
        return;
    }
    int64_t count = dataset->count_;
    std::vector<int64_t> ids(count);
    memcpy(ids.data(), dataset->base_->GetIds(), count * sizeof(int64_t));
    std::vector<char> extra_infos(extra_info_size * count);
    {
        auto result = index->GetExtraInfoByIds(ids.data(), count, extra_infos.data());
        REQUIRE(result.has_value());
    }

    std::vector<char> empty_extra_info(extra_info_size);

    for (int64_t i = 0; i < count; ++i) {
        auto extra_info_dataset = vsag::Dataset::Make();
        extra_info_dataset->ExtraInfos(empty_extra_info.data())
            ->NumElements(1)
            ->Owner(false)
            ->ExtraInfoSize(extra_info_size)
            ->Ids(ids.data() + i);
        auto result = index->UpdateExtraInfo(extra_info_dataset);
        REQUIRE(result.has_value());
        REQUIRE(result.value());
    }

    {
        int64_t invalid_label = 1000000001;
        auto extra_info_dataset = vsag::Dataset::Make();
        extra_info_dataset->ExtraInfos(empty_extra_info.data())
            ->NumElements(1)
            ->Owner(false)
            ->ExtraInfoSize(extra_info_size)
            ->Ids(&invalid_label);
        auto result = index->UpdateExtraInfo(extra_info_dataset);
        REQUIRE(result.has_value());
        REQUIRE(!result.value());
    }

    for (int64_t i = 0; i < count; ++i) {
        auto extra_info_dataset = vsag::Dataset::Make();
        extra_info_dataset->ExtraInfos(extra_infos.data() + i * extra_info_size)
            ->NumElements(1)
            ->Owner(false)
            ->ExtraInfoSize(extra_info_size)
            ->Ids(ids.data() + i);
        auto result = index->UpdateExtraInfo(extra_info_dataset);
        REQUIRE(result.has_value());
        REQUIRE(result.value());
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
TestIndex::TestClone(const TestIndex::IndexPtr& index,
                     const TestDatasetPtr& dataset,
                     const std::string& search_param) {
    if (not index->CheckFeature(vsag::SUPPORT_CLONE)) {
        return;
    }
    auto index_clone_result = index->Clone();
    REQUIRE(index_clone_result.has_value() == true);
    auto& index_clone = index_clone_result.value();

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto topk = 10;
    for (auto i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        auto res_from = index->KnnSearch(query, topk, search_param);
        auto res_to = index_clone->KnnSearch(query, topk, search_param);
        REQUIRE(res_from.has_value());
        REQUIRE(res_to.has_value());
        REQUIRE(res_from.value()->GetDim() == res_to.value()->GetDim());
        int64_t result_count = res_from.value()->GetDim();
        for (int64_t j = 0; j < result_count; ++j) {
            REQUIRE(res_to.value()->GetIds()[j] == res_from.value()->GetIds()[j]);
        }
    }
}

void
TestIndex::TestExportModel(const TestIndex::IndexPtr& index,
                           const TestIndex::IndexPtr& index2,
                           const TestDatasetPtr& dataset,
                           const std::string& search_param) {
    if (not index->CheckFeature(vsag::SUPPORT_EXPORT_MODEL)) {
        return;
    }
    auto index_model_result = index->ExportModel();
    REQUIRE(index_model_result.has_value() == true);
    auto index_model = index_model_result.value();
    fixtures::test_serializion_file(*index_model, *index2, "export_model_test");
    index_model = index2;
    tl::expected<std::vector<int64_t>, vsag::Error> add_index;
    if (index->CheckFeature(vsag::SUPPORT_ADD_AFTER_BUILD)) {
        add_index = index_model->Add(dataset->base_);
        REQUIRE(add_index.has_value());
    } else if (index->CheckFeature(vsag::SUPPORT_BUILD)) {
        add_index = index_model->Build(dataset->base_);
        REQUIRE(add_index.has_value());
    } else {
        return;
    }

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    float recall1 = 0.0F;
    float recall2 = 0.0F;
    auto topk = gt_topK;
    for (auto i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        auto res1 = index->KnnSearch(query, topk, search_param);
        REQUIRE(res1.has_value());
        auto result1 = res1.value()->GetIds();
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK, result1, topk);
        recall1 += static_cast<float>(val) / static_cast<float>(gt_topK);

        auto res2 = index_model->KnnSearch(query, topk, search_param);
        REQUIRE(res2.has_value());
        auto result2 = res2.value()->GetIds();
        val = Intersection(gt, gt_topK, result2, topk);
        recall2 += static_cast<float>(val) / static_cast<float>(gt_topK);
    }

    REQUIRE(std::abs(recall1 - recall2) < 0.01F * query_count);
}

void
TestIndex::TestMarkRemoveIndex(const TestIndex::IndexPtr& index,
                               const TestDatasetPtr& dataset,
                               const std::string& search_param,
                               bool expected_success) {
    if (index->GetIndexType() != vsag::IndexType::HNSW) {
        auto train_result = index->Train(dataset->base_);
        REQUIRE(train_result.has_value());
    }

    auto base_num = dataset->base_->GetNumElements();
    auto base_dim = dataset->base_->GetDim();
    auto vectors = dataset->base_->GetFloat32Vectors();
    auto ids = dataset->base_->GetIds();

    // step 1: add base data to index
    auto add_results = index->Add(dataset->base_);
    REQUIRE(add_results.has_value());
    REQUIRE(add_results.value().empty());

    // step 2: verify initial state
    REQUIRE(index->GetNumElements() == base_num);
    REQUIRE(index->GetNumberRemoved() == 0);

    // step 3: test mark remove operation
    {
        // test mark remove operation with invalid id
        auto wrong_result = index->Remove(-1, vsag::RemoveMode::MARK_REMOVE);
        if (index->GetIndexType() == vsag::IndexType::HNSW) {
            REQUIRE(wrong_result.has_value());
            REQUIRE_FALSE(wrong_result.value());
        } else {
            REQUIRE(wrong_result.has_value());
            REQUIRE(wrong_result.value() == 0);
        }

        // delete half of the base data
        int64_t remove_count = base_num / 2;
        std::vector<int64_t> remove_ids;
        remove_ids.assign(ids, ids + remove_count);
        auto remove_result = index->Remove(remove_ids, vsag::RemoveMode::MARK_REMOVE);

        // verify number of removed elements and num elements after mark remove
        REQUIRE(remove_result.has_value());
        REQUIRE(remove_result.value());
        REQUIRE(index->GetNumElements() == base_num - remove_count);
        REQUIRE(index->GetNumberRemoved() == remove_count);

        // test mark remove operation with duplicate id
        auto duplicate_remove = index->Remove(remove_ids, vsag::RemoveMode::MARK_REMOVE);
        REQUIRE(duplicate_remove.has_value());
        REQUIRE(duplicate_remove.value() == 0);

        // test search operation after mark remove
        for (int64_t i = 0; i < base_num; ++i) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)
                ->Dim(base_dim)
                ->Float32Vectors(vectors + i * base_dim)
                ->Owner(false);

            int64_t k = 10;
            auto search_result = index->KnnSearch(query, k, search_param);
            REQUIRE(search_result.has_value());

            // verify removed ids are not in search results
            if (i < remove_count) {
                bool found_removed_id = false;
                for (int64_t j = 0; j < search_result.value()->GetDim(); ++j) {
                    if (search_result.value()->GetIds()[j] == ids[i]) {
                        found_removed_id = true;
                        break;
                    }
                }
                REQUIRE_FALSE(found_removed_id);
                query->Ids(ids + i);
                auto add_result = index->Add(query);
                REQUIRE(add_result.has_value());
            }
        }
    }
}

void
TestIndex::TestRemoveIndex(const TestIndex::IndexPtr& index,
                           const TestDatasetPtr& dataset,
                           bool expected_success) {
    if (index->GetIndexType() != vsag::IndexType::HNSW) {
        auto train_result = index->Train(dataset->base_);
        REQUIRE(train_result.has_value());
    }
    auto base_num = dataset->base_->GetNumElements();
    auto base_dim = dataset->base_->GetDim();
    for (int64_t i = 0; i < base_num; ++i) {
        auto new_data = vsag::Dataset::Make();
        int64_t id = base_num + i;
        new_data->NumElements(1)
            ->Dim(base_dim)
            ->Ids(&id)
            ->Float32Vectors(dataset->base_->GetFloat32Vectors() + i * base_dim)
            ->Owner(false);
        auto add_results = index->Add(new_data);
        REQUIRE(add_results.has_value());
    }
    for (int64_t i = 0; i < base_num; ++i) {
        auto new_data = vsag::Dataset::Make();
        new_data->NumElements(1)
            ->Dim(base_dim)
            ->Ids(dataset->base_->GetIds() + i)
            ->Float32Vectors(dataset->base_->GetFloat32Vectors() + i * base_dim)
            ->Owner(false);
        auto add_results = index->Add(new_data);
        REQUIRE(add_results.has_value());

        auto remove_results = index->Remove(i + base_num);
        REQUIRE(index->GetNumberRemoved() == i + 1);
        REQUIRE(remove_results.has_value());
        remove_results = index->Remove(i + base_num);
        if (index->GetIndexType() != vsag::IndexType::HNSW) {
            REQUIRE_FALSE(remove_results.has_value());
        } else {
            REQUIRE(remove_results.has_value());
            REQUIRE_FALSE(remove_results.value());
        }
        REQUIRE(index->GetNumElements() == dataset->base_->GetNumElements());
    }
}

void
TestIndex::TestRecoverRemoveIndex(const IndexPtr& index,
                                  const TestDatasetPtr& dataset,
                                  const std::string& search_parameters) {
    auto base_num = dataset->base_->GetNumElements();
    auto base_dim = dataset->base_->GetDim();
    auto vectors = dataset->base_->GetFloat32Vectors();
    auto ids = dataset->base_->GetIds();

    // [step 0] build
    if (index->GetIndexType() != vsag::IndexType::HNSW) {
        auto train_result = index->Train(dataset->base_);
        REQUIRE(train_result.has_value());
    }
    auto add_results = index->Add(dataset->base_);
    REQUIRE(add_results.has_value());
    REQUIRE(add_results.value().size() == 0);

    // [step 0] test original recall
    int correct = 0;
    for (int i = 0; i < base_num; i++) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(base_dim)->Float32Vectors(vectors + i * base_dim)->Owner(false);

        int64_t k = 10;
        auto result = index->KnnSearch(query, k, search_parameters);
        REQUIRE(result.has_value());
        if (result.value()->GetIds()[0] == ids[i]) {
            correct += 1;
        }
    }
    float recall_before = ((float)correct) / base_num;

    {
        auto wrong_result = index->Remove(-1);
        if (index->GetIndexType() == vsag::IndexType::HNSW) {
            REQUIRE(wrong_result.has_value());
            REQUIRE_FALSE(wrong_result.value());
        } else {
            REQUIRE_FALSE(wrong_result.has_value());
        }
    }

    {  // Test of removing and then inserting for successful recovery
        // [step 1] remove half data
        for (int i = 0; i < base_num / 2; ++i) {
            REQUIRE(index->GetNumElements() == base_num - i);
            REQUIRE(index->GetNumberRemoved() == i);
            auto result = index->Remove(ids[i]);
            REQUIRE(result.has_value());
            REQUIRE(result.value());
        }
        REQUIRE(index->GetNumElements() == base_num / 2);
        REQUIRE(index->GetNumberRemoved() == base_num / 2);

        // [step 2] test recall of half data
        correct = 0;
        for (int i = 0; i < base_num; i++) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)
                ->Dim(base_dim)
                ->Float32Vectors(vectors + i * base_dim)
                ->Owner(false);

            int64_t k = 10;
            auto result = index->KnnSearch(query, k, search_parameters);
            REQUIRE(result.has_value());
            if (i < base_num / 2) {
                REQUIRE(result.value()->GetIds()[0] != ids[i]);
            } else {
                if (result.value()->GetIds()[0] == ids[i]) {
                    correct += 1;
                }
            }
        }
        float recall_removed = ((float)correct) / (base_num / 2);
        REQUIRE(recall_removed >= 0.90);

        // [step 3] add <original data> into index again for recovery
        correct = 0;
        auto half_valid_dataset = vsag::Dataset::Make();
        half_valid_dataset->NumElements(base_num)
            ->Dim(base_dim)
            ->Float32Vectors(vectors)
            ->Ids(ids)
            ->Owner(false);
        auto result3 = index->Add(half_valid_dataset);
        REQUIRE(result3.value().size() == base_num / 2);  // caused by half data already present
        REQUIRE(index->GetNumElements() ==
                base_num);                        // successfully restored to the original index
        REQUIRE(index->GetNumberRemoved() == 0);  // [key point] remove operation was rolled back

        // [step 4] test recall of recovery
        for (int i = 0; i < base_num; i++) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)
                ->Dim(base_dim)
                ->Float32Vectors(vectors + i * base_dim)
                ->Owner(false);

            int64_t k = 10;
            auto result = index->KnnSearch(query, k, search_parameters);
            REQUIRE(result.has_value());
            if (result.value()->GetIds()[0] == ids[i]) {
                correct += 1;
            }
        }
        float recall_after = ((float)correct) / base_num;
        REQUIRE(std::abs(recall_before - recall_after) < 0.01);
    }

    {  // Test of removing and then inserting for failed recovery
        // [step 1] remove half data
        for (int i = 0; i < base_num / 2; ++i) {
            REQUIRE(index->GetNumElements() == base_num - i);
            REQUIRE(index->GetNumberRemoved() == i);
            auto result = index->Remove(ids[i]);
            REQUIRE(result.has_value());
            REQUIRE(result.value());
        }
        REQUIRE(index->GetNumElements() == base_num / 2);
        REQUIRE(index->GetNumberRemoved() == base_num / 2);

        // [step 2] add <reverse data> into index again try to recovery but failed, execute directly insertion
        auto half_valid_dataset = vsag::Dataset::Make();
        std::vector<int64_t> alter_ids(base_num);
        for (int i = 0; i < base_num; i++) {
            alter_ids[i] = ids[base_num - i - 1];
        }
        half_valid_dataset->NumElements(base_num)
            ->Dim(base_dim)
            ->Float32Vectors(vectors)
            ->Ids(alter_ids.data())
            ->Owner(false);
        auto result2 = index->Add(half_valid_dataset);
        REQUIRE(result2.value().size() == base_num / 2);  // caused by half data already present
        // 1.0: caused by directly inserting half vectors (0.5 + 0.5 = 1.0)
        // 0.05: caused by small amount of incorrect recovery vectors
        REQUIRE(std::abs(index->GetNumElements() - base_num) < base_num * 0.01);
        REQUIRE(std::abs(index->GetNumberRemoved() - base_num / 2) <
                base_num * 0.05);  // [key point] no impact on the removed data
    }

    // [clean] recovery index
    for (int i = 0; i < base_num; i++) {
        auto original_base = vsag::Dataset::Make();
        original_base->NumElements(1)
            ->Dim(base_dim)
            ->Float32Vectors(vectors + i * base_dim)
            ->Ids(ids + i)
            ->Owner(false);
        auto update_recovery_result = index->UpdateVector(ids[i], original_base, true);
        if (not update_recovery_result) {
            auto add_result = index->Add(original_base);
        }
    }
    REQUIRE(index->GetNumElements() == base_num);
}

template <class T>
std::string
create_attr_string(const std::string& name, const std::vector<T>& values) {
    if (values.size() == 1) {
        std::stringstream ss;
        if constexpr (std::is_same_v<T, std::string>) {
            ss << name << " = \"" << values[0] << "\"";
        } else {
            ss << name << " = " << std::to_string(values[0]);
        }
        return ss.str();
    }
    std::ostringstream oss;
    for (uint64_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            oss << "|";
        }
        if constexpr (std::is_same_v<T, std::string>) {
            oss << values[i];
        } else {
            oss << std::to_string(values[i]);
        }
    }
    return "multi_in(" + name + ", \"" + oss.str() + "\", \"|\")";
}

std::string
trans_attr_to_string(const vsag::Attribute& attr) {
    using namespace vsag;
    auto name = attr.name_;
    auto type = attr.GetValueType();
    if (type == AttrValueType::STRING) {
        const auto temp = dynamic_cast<const AttributeValue<std::string>*>(&attr);
        auto values = temp->GetValue();
        return create_attr_string(name, values);
    } else if (type == AttrValueType::UINT8) {
        const auto temp = dynamic_cast<const AttributeValue<uint8_t>*>(&attr);
        auto values = temp->GetValue();
        return create_attr_string(name, values);
    } else if (type == AttrValueType::UINT16) {
        const auto temp = dynamic_cast<const AttributeValue<uint16_t>*>(&attr);
        auto values = temp->GetValue();
        return create_attr_string(name, values);
    } else if (type == AttrValueType::UINT32) {
        const auto temp = dynamic_cast<const AttributeValue<uint32_t>*>(&attr);
        auto values = temp->GetValue();
        return create_attr_string(name, values);
    } else if (type == AttrValueType::UINT64) {
        const auto temp = dynamic_cast<const AttributeValue<uint64_t>*>(&attr);
        auto values = temp->GetValue();
        return create_attr_string(name, values);
    } else if (type == AttrValueType::INT8) {
        const auto temp = dynamic_cast<const AttributeValue<int8_t>*>(&attr);
        auto values = temp->GetValue();
        return create_attr_string(name, values);
    } else if (type == AttrValueType::INT16) {
        const auto temp = dynamic_cast<const AttributeValue<int16_t>*>(&attr);
        auto values = temp->GetValue();
        return create_attr_string(name, values);
    } else if (type == AttrValueType::INT32) {
        const auto temp = dynamic_cast<const AttributeValue<int32_t>*>(&attr);
        auto values = temp->GetValue();
        return create_attr_string(name, values);
    } else if (type == AttrValueType::INT64) {
        const auto temp = dynamic_cast<const AttributeValue<int64_t>*>(&attr);
        auto values = temp->GetValue();
        return create_attr_string(name, values);
    }
    return "";
}

template <typename T>
static vsag::Attribute*
mock_value(const vsag::AttributeValue<T>* attr) {
    auto result = new vsag::AttributeValue<T>();
    result->name_ = attr->name_;
    auto old_values = std::unordered_set<T>(attr->GetValue().begin(), attr->GetValue().end());
    T random_new_value;
    if constexpr (std::is_same_v<T, std::string>) {
        random_new_value = "random_string";
    } else {
        random_new_value = static_cast<T>(rand());
        while (old_values.count(random_new_value)) {
            random_new_value = static_cast<T>(rand());
        }
    }
    result->GetValue().emplace_back(random_new_value);
    return result;
}

static void
mock_attrset(vsag::Attribute& attr, vsag::AttributeSet& old_attrs, vsag::AttributeSet& new_attrs) {
    using namespace vsag;
    old_attrs.attrs_.emplace_back(&attr);
    auto name = attr.name_;
    auto type = attr.GetValueType();
    if (type == AttrValueType::STRING) {
        const auto temp = dynamic_cast<const AttributeValue<std::string>*>(&attr);
        new_attrs.attrs_.emplace_back(mock_value<std::string>(temp));
    } else if (type == AttrValueType::UINT8) {
        const auto temp = dynamic_cast<const AttributeValue<uint8_t>*>(&attr);
        new_attrs.attrs_.emplace_back(mock_value<uint8_t>(temp));
    } else if (type == AttrValueType::UINT16) {
        const auto temp = dynamic_cast<const AttributeValue<uint16_t>*>(&attr);
        new_attrs.attrs_.emplace_back(mock_value<uint16_t>(temp));
    } else if (type == AttrValueType::UINT32) {
        const auto temp = dynamic_cast<const AttributeValue<uint32_t>*>(&attr);
        new_attrs.attrs_.emplace_back(mock_value<uint32_t>(temp));
    } else if (type == AttrValueType::UINT64) {
        const auto temp = dynamic_cast<const AttributeValue<uint64_t>*>(&attr);
        new_attrs.attrs_.emplace_back(mock_value<uint64_t>(temp));
    } else if (type == AttrValueType::INT8) {
        const auto temp = dynamic_cast<const AttributeValue<int8_t>*>(&attr);
        new_attrs.attrs_.emplace_back(mock_value<int8_t>(temp));
    } else if (type == AttrValueType::INT16) {
        const auto temp = dynamic_cast<const AttributeValue<int16_t>*>(&attr);
        new_attrs.attrs_.emplace_back(mock_value<int16_t>(temp));
    } else if (type == AttrValueType::INT32) {
        const auto temp = dynamic_cast<const AttributeValue<int32_t>*>(&attr);
        new_attrs.attrs_.emplace_back(mock_value<int32_t>(temp));
    } else if (type == AttrValueType::INT64) {
        const auto temp = dynamic_cast<const AttributeValue<int64_t>*>(&attr);
        new_attrs.attrs_.emplace_back(mock_value<int64_t>(temp));
    }
}

static void
release_attrset(vsag::AttributeSet& attrset) {
    for (auto* attr : attrset.attrs_) {
        delete attr;
    }
}

void
TestIndex::TestWithAttr(const IndexPtr& index,
                        const TestDatasetPtr& dataset,
                        const std::string& search_param,
                        bool with_update) {
    using namespace vsag;
    auto attrsets = dataset->base_->GetAttributeSets();
    auto* query_vec = dataset->base_->GetFloat32Vectors();
    auto count = std::min<int64_t>(dataset->base_->GetNumElements(), 200);
    auto dim = dataset->base_->GetDim();
    const auto* labels = dataset->base_->GetIds();

    for (int i = 0; i < count; ++i) {
        SearchRequest req;
        auto query = vsag::Dataset::Make();
        query->Float32Vectors(query_vec + i * dim)->Dim(dim)->Owner(false)->NumElements(1);
        auto attrset = attrsets[i].attrs_;
        int j1 = random() % attrset.size();
        int j2 = random() % attrset.size();
        req.topk_ = 10;
        req.filter_ = nullptr;
        req.params_str_ = search_param;
        req.enable_attribute_filter_ = true;
        req.query_ = query;
        req.attribute_filter_str_ = "(" + trans_attr_to_string(*attrset[j2]) + ") AND (" +
                                    trans_attr_to_string(*attrset[j1]) + ")";
        auto the_id = dataset->base_->GetIds()[i];
        auto result = index->SearchWithRequest(req);
        REQUIRE(result.has_value());
        auto ids = result.value()->GetIds();
        auto result_count = result.value()->GetDim();
        std::unordered_set<int64_t> sets(ids, ids + result_count);
        REQUIRE(sets.find(the_id) != sets.end());
        if (not with_update) {
            continue;
        }
        AttributeSet new_attrs;
        AttributeSet old_attrs;
        mock_attrset(*attrset[j1], old_attrs, new_attrs);

        auto test_func = [&]() -> void {
            auto result1 = index->SearchWithRequest(req);
            REQUIRE(result1.has_value());
            auto* ids = result1.value()->GetIds();
            auto result_count = result1.value()->GetDim();
            if (result_count != 0) {
                std::unordered_set<int64_t> sets1(ids, ids + result_count);
                REQUIRE(sets1.find(the_id) == sets1.end());
            }

            req.attribute_filter_str_ = "(" + trans_attr_to_string(*new_attrs.attrs_[0]) + ")";

            auto result2 = index->SearchWithRequest(req);
            REQUIRE(result2.has_value());
            ids = result2.value()->GetIds();
            result_count = result2.value()->GetDim();
            std::unordered_set<int64_t> sets1(ids, ids + result_count);
            REQUIRE(sets1.find(the_id) != sets1.end());
        };

        if (i % 2 == 0) {
            index->UpdateAttribute(labels[i], new_attrs, old_attrs);
            test_func();
        } else {
            index->UpdateAttribute(labels[i], new_attrs);
            test_func();
        }
        release_attrset(new_attrs);
    }
}

void
TestIndex::TestGetRawVectorByIds(const IndexPtr& index,
                                 const TestDatasetPtr& dataset,
                                 bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_GET_RAW_VECTOR_BY_IDS)) {
        return;
    }

    // get with not existed id
    {
        int64_t non_exist_id = -9999999;
        auto failed_res = index->GetRawVectorByIds(&non_exist_id, 1);
        REQUIRE(not failed_res.has_value());
    }

    auto count = static_cast<int64_t>(dataset->count_);
    vsag::IndexDetailInfo info;
    auto data_type = index->GetDetailDataByName("data_type", info).value()->GetDataScalarString();

    for (bool use_specific_allocator : {true, false}) {
        // specific_allocator
        vsag::DefaultAllocator allocator;
        void* mem = nullptr;

        // common case
        auto vectors = index->GetRawVectorByIds(dataset->base_->GetIds(), count);
        REQUIRE(vectors.has_value());

        if (use_specific_allocator) {
            vectors = index->GetRawVectorByIds(dataset->base_->GetIds(), count, &allocator);
            // create a delegate task (via Dataset Owner mechanism) to check and release the memory which allocated from the external allocator.
            vectors.value()->Owner(true, &allocator);
        }

        if (data_type == vsag::DATATYPE_SPARSE) {
            mem = (void*)vectors.value()->GetSparseVectors();
            for (int i = 0; i < count; i++) {
                // get single data
                auto single_dataset = vsag::Dataset::Make();
                auto sparse_vectors = vectors.value()->GetSparseVectors() + i;
                single_dataset->SparseVectors(sparse_vectors)->NumElements(1)->Owner(false);
                if (not expected_success) {
                    return;
                }

                // self distance
                auto dists_res =
                    index->CalDistanceById(single_dataset, dataset->base_->GetIds() + i, 1);
                REQUIRE(dists_res.has_value());
                auto dist = dists_res.value()->GetDistances()[0];

                // ground truth distance
                float gt_dist = 0;
                for (int j = 0; j < sparse_vectors->len_; j++) {
                    gt_dist += sparse_vectors->vals_[j] * sparse_vectors->vals_[j];
                }
                gt_dist = 1 - gt_dist;
                REQUIRE(std::abs(gt_dist - dist) < 1e-3);
            }
        } else if (data_type == vsag::DATATYPE_FLOAT32) {
            auto float_vectors = vectors.value()->GetFloat32Vectors();
            mem = (void*)float_vectors;
            auto dim = dataset->base_->GetDim();
            if (not expected_success) {
                return;
            }
            for (int i = 0; i < count; ++i) {
                REQUIRE(std::memcmp(float_vectors + i * dim,
                                    dataset->base_->GetFloat32Vectors() + i * dim,
                                    dim * sizeof(float)) == 0);
            }
        } else if (data_type == vsag::DATATYPE_INT8) {
            auto int8_vectors = vectors.value()->GetInt8Vectors();
            mem = (void*)int8_vectors;
            auto dim = dataset->base_->GetDim();
            if (not expected_success) {
                return;
            }
            for (int i = 0; i < count; ++i) {
                REQUIRE(std::memcmp(int8_vectors + i * dim,
                                    dataset->base_->GetInt8Vectors() + i * dim,
                                    dim) == 0);
            }
        } else {
            throw std::invalid_argument("Invalid data type: " + data_type);
        }
    }
}
void
TestIndex::TestBuildDuplicateIndex(const IndexPtr& index,
                                   const TestDatasetPtr& dataset,
                                   const std::string& duplicate_pos,
                                   bool expect_success) {
    index->Train(dataset->base_);
    if (duplicate_pos == "prefix") {
        auto result = index->Build(dataset->base_);
        REQUIRE(result.has_value() == expect_success);
        for (int64_t i = dataset->base_->GetNumElements(); i < 2 * dataset->base_->GetNumElements();
             ++i) {
            auto new_data = vsag::Dataset::Make();
            new_data->NumElements(1)
                ->Dim(dataset->base_->GetDim())
                ->Ids(&i)
                ->SparseVectors(dataset->base_->GetSparseVectors())
                ->Paths(dataset->base_->GetPaths())
                ->Float32Vectors(dataset->base_->GetFloat32Vectors())
                ->Owner(false);
            auto add_result = index->Add(new_data);
            REQUIRE(add_result.has_value() == expect_success);
        }
    } else if (duplicate_pos == "suffix") {
        for (int64_t i = dataset->base_->GetNumElements(); i < 2 * dataset->base_->GetNumElements();
             ++i) {
            auto new_data = vsag::Dataset::Make();
            new_data->NumElements(1)
                ->Dim(dataset->base_->GetDim())
                ->Ids(&i)
                ->SparseVectors(dataset->base_->GetSparseVectors())
                ->Paths(dataset->base_->GetPaths())
                ->Float32Vectors(dataset->base_->GetFloat32Vectors())
                ->Owner(false);
            auto add_result = index->Add(new_data);
            REQUIRE(add_result.has_value() == expect_success);
        }
        auto result = index->Add(dataset->base_);
        REQUIRE(result.has_value() == expect_success);
    } else if (duplicate_pos == "middle") {
        auto add_result = index->Add(dataset->base_);
        REQUIRE(add_result.has_value() == expect_success);
    } else {
        throw std::invalid_argument("Invalid duplicate position: " + duplicate_pos);
    }
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

void
TestIndex::TestExportIDs(const IndexPtr& index, const TestDatasetPtr& dataset) {
    if (not index->CheckFeature(vsag::SUPPORT_EXPORT_IDS)) {
        return;
    }
    auto result = index->ExportIDs();
    REQUIRE(result.has_value());
    const auto* ids = result.value()->GetIds();
    auto num_element = result.value()->GetNumElements();
    REQUIRE(num_element == dataset->base_->GetNumElements());
    auto* origin_ids = dataset->base_->GetIds();
    // check ids, no order
    std::unordered_set<int64_t> id_set(origin_ids, origin_ids + num_element);
    for (int64_t i = 0; i < num_element; ++i) {
        REQUIRE(id_set.find(ids[i]) != id_set.end());
    }
    std::unordered_set<int64_t> id_set2(ids, ids + num_element);
    REQUIRE(id_set2.size() == num_element);
}

template <typename T>
static void
compare_attr_value(const vsag::Attribute* attr1, const vsag::Attribute* attr2) {
    auto count = attr1->GetValueCount();
    auto* ptr1 = dynamic_cast<const vsag::AttributeValue<T>*>(attr1);
    auto* ptr2 = dynamic_cast<const vsag::AttributeValue<T>*>(attr2);
    const auto& temp_vec1 = ptr1->GetValue();
    const auto& temp_vec2 = ptr2->GetValue();
    std::unordered_set<T> temp_set1(temp_vec1.begin(), temp_vec1.end());
    std::unordered_set<T> temp_set2(temp_vec2.begin(), temp_vec2.end());
    REQUIRE(temp_set1 == temp_set2);
}

static void
compare_attr_set(const vsag::AttributeSet& attr1, const vsag::AttributeSet& attr2) {
    REQUIRE(attr1.attrs_.size() == attr2.attrs_.size());
    auto size = attr1.attrs_.size();
    auto temp_vec1 = attr1.attrs_;
    auto temp_vec2 = attr2.attrs_;
    std::sort(temp_vec1.begin(), temp_vec1.end(), [](const auto& a, const auto& b) {
        return a->name_ < b->name_;
    });
    std::sort(temp_vec2.begin(), temp_vec2.end(), [](const auto& a, const auto& b) {
        return a->name_ < b->name_;
    });
    for (int i = 0; i < size; ++i) {
        auto& attr = temp_vec1[i];
        auto& gt_attr = temp_vec2[i];
        REQUIRE(attr->name_ == gt_attr->name_);
        REQUIRE(attr->GetValueType() == gt_attr->GetValueType());
        if (attr->GetValueType() == vsag::AttrValueType::UINT64) {
            compare_attr_value<uint64_t>(attr, gt_attr);
        } else if (attr->GetValueType() == vsag::AttrValueType::INT64) {
            compare_attr_value<int64_t>(attr, gt_attr);
        } else if (attr->GetValueType() == vsag::AttrValueType::UINT32) {
            compare_attr_value<uint32_t>(attr, gt_attr);
        } else if (attr->GetValueType() == vsag::AttrValueType::INT32) {
            compare_attr_value<int32_t>(attr, gt_attr);
        } else if (attr->GetValueType() == vsag::AttrValueType::UINT16) {
            compare_attr_value<uint16_t>(attr, gt_attr);
        } else if (attr->GetValueType() == vsag::AttrValueType::INT16) {
            compare_attr_value<int16_t>(attr, gt_attr);
        } else if (attr->GetValueType() == vsag::AttrValueType::UINT8) {
            compare_attr_value<uint8_t>(attr, gt_attr);
        } else if (attr->GetValueType() == vsag::AttrValueType::INT8) {
            compare_attr_value<int8_t>(attr, gt_attr);
        } else if (attr->GetValueType() == vsag::AttrValueType::STRING) {
            compare_attr_value<std::string>(attr, gt_attr);
        }
    }
}

void
TestIndex::TestGetDataById(const IndexPtr& index, const TestDatasetPtr& dataset) {
    if (not index->CheckFeature(vsag::SUPPORT_GET_DATA_BY_IDS)) {
        return;
    }
    auto result = index->GetDataByIds(dataset->base_->GetIds(), dataset->base_->GetNumElements());
    REQUIRE(result.has_value());
    auto data = result.value();
    REQUIRE(data->GetNumElements() == dataset->base_->GetNumElements());
    REQUIRE(data->GetDim() == dataset->base_->GetDim());
    // vectors
    auto float_vectors = data->GetFloat32Vectors();
    for (int i = 0; i < data->GetNumElements(); ++i) {
        REQUIRE(memcmp(float_vectors + i * data->GetDim(),
                       dataset->base_->GetFloat32Vectors() + i * data->GetDim(),
                       data->GetDim() * sizeof(float)) == 0);
    }
    // attributes
    auto attrs = data->GetAttributeSets();
    auto gt_attrs = dataset->base_->GetAttributeSets();
    for (int i = 0; i < data->GetNumElements(); ++i) {
        auto& attr = attrs[i];
        auto& gt_attr = gt_attrs[i];
        compare_attr_set(attr, gt_attr);
    }
}

void
TestIndex::TestIndexStatus(const IndexPtr& index) {
    auto set_result = index->SetImmutable();
    if (not set_result.has_value()) {
        return;
    }
    REQUIRE_FALSE(index->Train(nullptr));
    REQUIRE_FALSE(index->Build(nullptr));
    REQUIRE_FALSE(index->Add(nullptr));
    std::ifstream inf;
    REQUIRE_FALSE(index->Deserialize(inf));
    REQUIRE_FALSE(index->Remove(0));
    std::vector<vsag::MergeUnit> merge_units;
    REQUIRE_FALSE(index->Merge(merge_units));
    vsag::AttributeSet new_attrs;
    REQUIRE_FALSE(index->UpdateAttribute(0, new_attrs));
    REQUIRE_FALSE(index->UpdateAttribute(0, new_attrs, new_attrs));
    REQUIRE_FALSE(index->UpdateId(0, 0));
    REQUIRE_FALSE(index->UpdateVector(0, nullptr, false));
}

void
TestIndex::TestGetDataByIdWithFlag(const IndexPtr& index, const TestDatasetPtr& dataset) {
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
        return add_index.has_value() & search_index.has_value() & remove_index.has_value();
    };

    for (uint64_t j = temp_count; j < base_count; ++j) {
        futures.emplace_back(pool.enqueue(func, j));
    }

    for (auto& res : futures) {
        auto val = res.get();
        REQUIRE(val);
    }
}

void
TestIndex::TestIndexDetailData(const IndexPtr& index) {
    auto infos = index->GetIndexDetailInfos();
    REQUIRE(infos.has_value());
    const auto& detail_infos = infos.value();
    REQUIRE(detail_infos.size() > 0);
    for (const auto& info : detail_infos) {
        vsag::IndexDetailInfo r_info;
        auto detail_data_value = index->GetDetailDataByName(info.name, r_info);
        REQUIRE(info.name == r_info.name);
        REQUIRE(info.type == r_info.type);
        REQUIRE(info.description == r_info.description);

        REQUIRE(detail_data_value.has_value());
        auto detail_data = detail_data_value.value();
        if (info.type == vsag::IndexDetailDataType::TYPE_SCALAR_INT64) {
            REQUIRE_NOTHROW(detail_data->GetDataScalarInt64());
        } else if (info.type == vsag::IndexDetailDataType::TYPE_SCALAR_DOUBLE) {
            REQUIRE_NOTHROW(detail_data->GetDataScalarDouble());
        } else if (info.type == vsag::IndexDetailDataType::TYPE_SCALAR_STRING) {
            REQUIRE_NOTHROW(detail_data->GetDataScalarString());
        } else if (info.type == vsag::IndexDetailDataType::TYPE_SCALAR_BOOL) {
            REQUIRE_NOTHROW(detail_data->GetDataScalarBool());
        } else if (info.type == vsag::IndexDetailDataType::TYPE_1DArray_INT64) {
            REQUIRE_NOTHROW(detail_data->GetData1DArrayInt64());
        } else if (info.type == vsag::IndexDetailDataType::TYPE_2DArray_INT64) {
            REQUIRE_NOTHROW(detail_data->GetData2DArrayInt64());
        } else {
            REQUIRE(false);
        }
    }
}
}  // namespace fixtures
