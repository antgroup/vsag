
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

#include "fixtures/test_logger.h"
#include "fixtures/thread_pool.h"
#include "simd/fp32_simd.h"

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

void
TestIndex::TestBuildIndex(const IndexPtr& index,
                          const TestDatasetPtr& dataset,
                          bool expected_success) {
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
            query->NumElements(1)->Dim(dim)->Float32Vectors(base + i * dim)->Owner(false);

            auto result = index->KnnSearch(query, gt_topK, search_param);
            REQUIRE(result.has_value());

            if (round == 0) {
                if (result.value()->GetIds()[0] == ids[i]) {
                    correct_num[round] += 1;
                }

                auto succ_update_res = index->UpdateId(ids[i], update_id_map[ids[i]]);
                REQUIRE(succ_update_res.has_value());
                if (expected_success) {
                    REQUIRE(succ_update_res.value());
                }

                // old id don't exist
                auto failed_old_res = index->UpdateId(ids[i], update_id_map[ids[i]]);
                REQUIRE(failed_old_res.has_value());
                REQUIRE(not failed_old_res.value());

                // new id is used
                auto failed_new_res = index->UpdateId(update_id_map[ids[i]], update_id_map[ids[i]]);
                REQUIRE(failed_new_res.has_value());
                REQUIRE(not failed_new_res.value());
            } else {
                if (result.value()->GetIds()[0] == update_id_map[ids[i]]) {
                    correct_num[round] += 1;
                }
            }
        }
    }

    REQUIRE(correct_num[0] == correct_num[1]);
}

void
TestIndex::TestUpdateVector(const IndexPtr& index,
                            const TestDatasetPtr& dataset,
                            const std::string& search_param,
                            bool expected_success) {
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

    std::vector<int> correct_num = {0, 0};
    for (int round = 0; round < 2; round++) {
        // round 0 for update, round 1 for validate update results
        for (int i = 0; i < num_vectors; i++) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)->Dim(dim)->Float32Vectors(base + i * dim)->Owner(false);

            auto result = index->KnnSearch(query, gt_topK, search_param);
            REQUIRE(result.has_value());

            if (round == 0) {
                if (result.value()->GetIds()[0] == ids[i]) {
                    correct_num[round] += 1;
                }

                std::vector<float> update_vecs(dim);
                for (int d = 0; d < dim; d++) {
                    update_vecs[d] = base[i * dim + d] + 0.001f;
                }
                auto new_base = vsag::Dataset::Make();
                new_base->NumElements(1)
                    ->Dim(dim)
                    ->Float32Vectors(update_vecs.data())
                    ->Owner(false);

                auto before_update_dist = *index->CalcDistanceById(base + i * dim, ids[i]);
                auto succ_vec_res = index->UpdateVector(ids[i], new_base);
                REQUIRE(succ_vec_res.has_value());
                if (expected_success) {
                    REQUIRE(succ_vec_res.value());
                }
                auto after_update_dist = *index->CalcDistanceById(base + i * dim, ids[i]);
                REQUIRE(before_update_dist < after_update_dist);

                // old id don't exist
                auto failed_old_res = index->UpdateVector(ids[i] + 2 * max_id, new_base);
                REQUIRE(failed_old_res.has_value());
                REQUIRE(not failed_old_res.value());
            } else {
                if (result.value()->GetIds()[0] == ids[i]) {
                    correct_num[round] += 1;
                }
            }
        }
    }

    REQUIRE(correct_num[0] == correct_num[1]);
}

void
TestIndex::TestContinueAdd(const IndexPtr& index,
                           const TestDatasetPtr& dataset,
                           bool expected_success) {
    auto base_count = dataset->base_->GetNumElements();
    int64_t temp_count = base_count / 2;
    auto dim = dataset->base_->GetDim();
    auto temp_dataset = vsag::Dataset::Make();
    temp_dataset->Dim(dim)
        ->Ids(dataset->base_->GetIds())
        ->NumElements(temp_count)
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);
    index->Build(temp_dataset);
    auto rest_count = base_count - temp_count;
    for (uint64_t j = rest_count; j < base_count; ++j) {
        auto data_one = vsag::Dataset::Make();
        data_one->Dim(dim)
            ->Ids(dataset->base_->GetIds() + j)
            ->NumElements(1)
            ->Float32Vectors(dataset->base_->GetFloat32Vectors() + j * dim)
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
TestIndex::TestKnnSearch(const IndexPtr& index,
                         const TestDatasetPtr& dataset,
                         const std::string& search_param,
                         float expected_recall,
                         bool expected_success) {
    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    float cur_recall = 0.0f;
    auto topk = gt_topK;
    for (auto i = 0; i < query_count; ++i) {
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
        auto result = res.value()->GetIds();
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK, result, topk);
        cur_recall += static_cast<float>(val) / static_cast<float>(gt_topK);
    }
    REQUIRE(cur_recall > expected_recall * query_count);
}

void
TestIndex::TestRangeSearch(const IndexPtr& index,
                           const TestDatasetPtr& dataset,
                           const std::string& search_param,
                           float expected_recall,
                           int64_t limited_size,
                           bool expected_success) {
    auto queries = dataset->range_query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->range_ground_truth_;
    auto gt_topK = gts->GetDim();
    const auto& radius = dataset->range_radius_;
    float cur_recall = 0.0f;
    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        auto res = index->RangeSearch(query, radius[i], search_param, limited_size);
        REQUIRE(res.has_value() == expected_success);
        if (!expected_success) {
            return;
        }
        if (limited_size > 0) {
            REQUIRE(res.value()->GetDim() <= limited_size);
        }
        auto result = res.value()->GetIds();
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK, result, res.value()->GetDim());
        cur_recall += static_cast<float>(val) / static_cast<float>(gt_topK);
    }
    REQUIRE(cur_recall > expected_recall * query_count);
}
void
TestIndex::TestFilterSearch(const TestIndex::IndexPtr& index,
                            const TestDatasetPtr& dataset,
                            const std::string& search_param,
                            float expected_recall,
                            bool expected_success) {
    auto queries = dataset->filter_query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->filter_ground_truth_;
    auto gt_topK = dataset->top_k;
    float cur_recall = 0.0f;
    auto topk = gt_topK;
    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        auto res = index->KnnSearch(query, topk, search_param, dataset->filter_function_);
        REQUIRE(res.has_value() == expected_success);
        if (!expected_success) {
            return;
        }
        REQUIRE(res.value()->GetDim() == topk);
        auto result = res.value()->GetIds();
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK, result, topk);
        cur_recall += static_cast<float>(val) / static_cast<float>(gt_topK);
    }
    REQUIRE(cur_recall > expected_recall * query_count);
}

void
TestIndex::TestCalcDistanceById(const IndexPtr& index, const TestDatasetPtr& dataset, float error) {
    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        for (auto j = 0; j < gt_topK; ++j) {
            auto id = gts->GetIds()[i * gt_topK + j];
            auto dist = gts->GetDistances()[i * gt_topK + j];
            auto result = index->CalcDistanceById(query->GetFloat32Vectors(), id);
            REQUIRE(result.has_value());
            REQUIRE(std::abs(dist - result.value()) < error);
        }
    }
}
void
TestIndex::TestSerializeFile(const IndexPtr& index_from,
                             const IndexPtr& index_to,
                             const TestDatasetPtr& dataset,
                             const std::string& search_param,
                             bool expected_success) {
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

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto topk = 10;
    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        auto res_from = index_from->KnnSearch(query, topk, search_param);
        auto res_to = index_to->KnnSearch(query, topk, search_param);
        REQUIRE(res_from.has_value());
        REQUIRE(res_to.has_value());
        REQUIRE(res_from.value()->GetDim() == res_to.value()->GetDim());
        for (auto j = 0; j < topk; ++j) {
            REQUIRE(res_to.value()->GetIds()[j] == res_from.value()->GetIds()[j]);
        }
    }
}

void
TestIndex::TestConcurrentAdd(const TestIndex::IndexPtr& index,
                             const TestDatasetPtr& dataset,
                             bool expected_success) {
    fixtures::logger::LoggerReplacer _;

    auto base_count = dataset->base_->GetNumElements();
    int64_t temp_count = base_count / 2;
    auto dim = dataset->base_->GetDim();
    auto temp_dataset = vsag::Dataset::Make();
    temp_dataset->Dim(dim)
        ->Ids(dataset->base_->GetIds())
        ->NumElements(temp_count)
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);
    index->Build(temp_dataset);
    auto rest_count = base_count - temp_count;
    fixtures::ThreadPool pool(5);
    using RetType = tl::expected<std::vector<int64_t>, vsag::Error>;
    std::vector<std::future<RetType>> futures;

    auto func = [&](uint64_t i) -> RetType {
        auto data_one = vsag::Dataset::Make();
        data_one->Dim(dim)
            ->Ids(dataset->base_->GetIds() + i)
            ->NumElements(1)
            ->Float32Vectors(dataset->base_->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        auto add_index = index->Add(data_one);
        return add_index;
    };

    for (uint64_t j = rest_count; j < base_count; ++j) {
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
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        auto res = index->KnnSearch(query, topk, search_param);
        return {res, i};
    };

    for (auto i = 0; i < query_count; ++i) {
        futures.emplace_back(pool.enqueue(func, i));
    }

    for (auto& res1 : futures) {
        auto [res, id] = res1.get();
        REQUIRE(res.has_value() == expected_success);
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
    REQUIRE(cur_recall > expected_recall * query_count);
}

void
TestIndex::TestContinueAddIgnoreRequire(const TestIndex::IndexPtr& index,
                                        const TestDatasetPtr& dataset) {
    auto base_count = dataset->base_->GetNumElements();
    int64_t temp_count = base_count / 2;
    auto dim = dataset->base_->GetDim();
    auto temp_dataset = vsag::Dataset::Make();
    temp_dataset->Dim(dim)
        ->Ids(dataset->base_->GetIds())
        ->NumElements(temp_count)
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);
    index->Build(temp_dataset);
    auto rest_count = base_count - temp_count;
    for (uint64_t j = rest_count; j < base_count; ++j) {
        auto data_one = vsag::Dataset::Make();
        data_one->Dim(dim)
            ->Ids(dataset->base_->GetIds() + j)
            ->NumElements(1)
            ->Float32Vectors(dataset->base_->GetFloat32Vectors() + j * dim)
            ->Owner(false);
        auto add_index = index->Add(data_one);
    }
}
void
TestIndex::TestDuplicateAdd(const TestIndex::IndexPtr& index, const TestDatasetPtr& dataset) {
    auto double_dataset = vsag::Dataset::Make();
    uint64_t base_count = dataset->base_->GetNumElements();
    uint64_t double_count = base_count * 2;
    auto dim = dataset->base_->GetDim();
    auto new_data = std::shared_ptr<float[]>(new float[double_count * dim]);
    auto new_ids = std::shared_ptr<int64_t[]>(new int64_t[double_count]);
    memcpy(new_data.get(), dataset->base_->GetFloat32Vectors(), base_count * dim * sizeof(float));
    memcpy(new_data.get() + base_count * dim,
           dataset->base_->GetFloat32Vectors(),
           base_count * dim * sizeof(float));
    memcpy(new_ids.get(), dataset->base_->GetIds(), base_count * sizeof(int64_t));
    memcpy(new_ids.get() + base_count, dataset->base_->GetIds(), base_count * sizeof(int64_t));
    double_dataset->Dim(dim)
        ->NumElements(double_count)
        ->Ids(new_ids.get())
        ->Float32Vectors(new_data.get())
        ->Owner(false);

    auto check_func = [&](std::vector<int64_t>& failed_ids) -> void {
        REQUIRE(failed_ids.size() == base_count);
        std::sort(failed_ids.begin(), failed_ids.end());
        for (uint64_t i = 0; i < base_count; ++i) {
            REQUIRE(failed_ids[i] == dataset->base_->GetIds()[i]);
        }
    };

    // add once with duplicate;
    auto add_index = index->Add(double_dataset);
    REQUIRE(add_index.has_value());
    check_func(add_index.value());

    // add twice with duplicate;
    auto add_index_2 = index->Add(dataset->base_);
    REQUIRE(add_index_2.has_value());
    check_func(add_index_2.value());
}

}  // namespace fixtures
