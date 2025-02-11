
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
                    if (index->CheckFeature(vsag::IndexFeature::SUPPORT_CHECK_ID_EXIST)) {
                        REQUIRE(index->CheckIdExist(ids[i]) == false);
                        REQUIRE(index->CheckIdExist(update_id_map[ids[i]]) == true);
                    }
                    REQUIRE(succ_update_res.value());
                }

                // old id don't exist
                auto failed_old_res = index->UpdateId(ids[i], update_id_map[ids[i]]);
                REQUIRE(failed_old_res.has_value());
                REQUIRE(not failed_old_res.value());

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
                REQUIRE(failed_new_res.has_value());
                REQUIRE(not failed_new_res.value());
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
    uint32_t success_force_updated = 0, failed_force_updated = 0;
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
                std::vector<float> far_vecs(dim);
                for (int d = 0; d < dim; d++) {
                    update_vecs[d] = base[i * dim + d] + 0.001f;
                    far_vecs[d] = base[i * dim + d] + 1.0f;
                }
                auto new_base = vsag::Dataset::Make();
                new_base->NumElements(1)
                    ->Dim(dim)
                    ->Float32Vectors(update_vecs.data())
                    ->Owner(false);

                // success case
                auto before_update_dist = *index->CalcDistanceById(base + i * dim, ids[i]);
                auto succ_vec_res = index->UpdateVector(ids[i], new_base);
                REQUIRE(succ_vec_res.has_value());
                if (expected_success) {
                    REQUIRE(succ_vec_res.value());
                }
                auto after_update_dist = *index->CalcDistanceById(base + i * dim, ids[i]);
                REQUIRE(before_update_dist < after_update_dist);

                // update with far vector
                new_base->Float32Vectors(far_vecs.data());
                auto fail_vec_res = index->UpdateVector(ids[i], new_base);
                REQUIRE(fail_vec_res.has_value());
                if (fail_vec_res.value()) {
                    // note that the update should be failed, but for some cases, it success
                    auto force_update_dist = *index->CalcDistanceById(base + i * dim, ids[i]);
                    REQUIRE(after_update_dist < force_update_dist);
                    success_force_updated++;
                } else {
                    failed_force_updated++;
                }

                // force update with far vector
                new_base->Float32Vectors(far_vecs.data());
                auto force_update_res1 = index->UpdateVector(ids[i], new_base, true);
                REQUIRE(force_update_res1.has_value());
                if (expected_success) {
                    REQUIRE(force_update_res1.value());
                    auto force_update_dist = *index->CalcDistanceById(base + i * dim, ids[i]);
                    REQUIRE(after_update_dist < force_update_dist);
                }

                new_base->Float32Vectors(update_vecs.data());
                auto force_update_res2 = index->UpdateVector(ids[i], new_base, true);
                REQUIRE(force_update_res2.has_value());
                if (expected_success) {
                    REQUIRE(force_update_res2.value());
                    auto force_update_dist = *index->CalcDistanceById(base + i * dim, ids[i]);
                    REQUIRE(std::abs(after_update_dist - force_update_dist) < 1e-5);
                }

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
    REQUIRE(success_force_updated < failed_force_updated);
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
        ->Paths(dataset->base_->GetPaths())
        ->Owner(false);
    index->Build(temp_dataset);
    auto rest_count = base_count - temp_count;
    for (uint64_t j = rest_count; j < base_count; ++j) {
        auto data_one = vsag::Dataset::Make();
        data_one->Dim(dim)
            ->Ids(dataset->base_->GetIds() + j)
            ->NumElements(1)
            ->Float32Vectors(dataset->base_->GetFloat32Vectors() + j * dim)
            ->Paths(dataset->base_->GetPaths() + j)
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
            ->Paths(queries->GetPaths() + i)
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
            ->Paths(queries->GetPaths() + i)
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
        auto val = Intersection(gt, gt_topK - 1, result, res.value()->GetDim());
        cur_recall += static_cast<float>(val) / static_cast<float>(gt_topK - 1);
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
    FilterObj(std::function<bool(int64_t)> filter_func, float valid_ratio)
        : filter_func_(std::move(filter_func)), valid_ratio_(valid_ratio) {
    }

    bool
    CheckValid(int64_t id) const override {
        return not filter_func_(id);
    }

    float
    ValidRatio() const override {
        return valid_ratio_;
    }

private:
    std::function<bool(int64_t)> filter_func_{nullptr};
    float valid_ratio_{1.0F};
};

void
TestIndex::TestFilterSearch(const TestIndex::IndexPtr& index,
                            const TestDatasetPtr& dataset,
                            const std::string& search_param,
                            float expected_recall,
                            bool expected_success,
                            bool support_filter_obj) {
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
            ->Paths(queries->GetPaths() + i)
            ->Owner(false);
        tl::expected<DatasetPtr, vsag::Error> res;
        if (support_filter_obj) {
            auto filter =
                std::make_shared<FilterObj>(dataset->filter_function_, dataset->valid_ratio_);
            res = index->KnnSearch(query, topk, search_param, filter);
            if (res.value()->GetDim() != topk) {
                res = index->KnnSearch(query, topk, search_param, filter);
            }
        } else {
            res = index->KnnSearch(query, topk, search_param, dataset->filter_function_);
        }
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
    if (cur_recall <= expected_recall * query_count) {
        WARN(fmt::format("cur_result({}) <= expected_recall * query_count({})",
                         cur_recall,
                         expected_recall * query_count));
    }
    REQUIRE(cur_recall > expected_recall * query_count * RECALL_THRESHOLD);
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
TestIndex::TestBatchCalcDistanceById(const IndexPtr& index,
                                     const TestDatasetPtr& dataset,
                                     float error) {
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
        auto result = index->CalDistanceById(
            query->GetFloat32Vectors(), gts->GetIds() + (i * gt_topK), gt_topK);
        for (auto j = 0; j < gt_topK; ++j) {
            REQUIRE(std::abs(gts->GetDistances()[i * gt_topK + j] -
                             result.value()->GetDistances()[j]) < error);
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
            ->Paths(queries->GetPaths() + i)
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
    auto serialize_binary = index_from->Serialize();
    REQUIRE(serialize_binary.has_value() == expected_success);

    auto deserialize_index = index_to->Deserialize(serialize_binary.value());
    REQUIRE(deserialize_index.has_value() == expected_success);

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto topk = 10;
    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Paths(queries->GetPaths() + i)
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
TestIndex::TestSerializeReaderSet(const IndexPtr& index_from,
                                  const IndexPtr& index_to,
                                  const TestDatasetPtr& dataset,
                                  const std::string& search_param,
                                  const std::string& index_name,
                                  bool expected_success) {
    vsag::ReaderSet rs;
    auto serialize_binary = index_from->Serialize();
    REQUIRE(serialize_binary.has_value() == expected_success);
    auto binary_set = serialize_binary.value();
    for (const auto& key : binary_set.GetKeys()) {
        rs.Set(key, std::make_shared<TestReader>(binary_set.Get(key)));
    }
    auto deserialize_index = index_to->Deserialize(rs);
    REQUIRE(deserialize_index.has_value() == expected_success);

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto topk = 10;
    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Paths(queries->GetPaths() + i)
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
    if (cur_recall <= expected_recall * query_count) {
        WARN(fmt::format("cur_result({}) <= expected_recall * query_count({})",
                         cur_recall,
                         expected_recall * query_count));
    }
    REQUIRE(cur_recall > expected_recall * query_count * RECALL_THRESHOLD);
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
void
TestIndex::TestEstimateMemory(const std::string& index_name,
                              const std::string& build_param,
                              const TestDatasetPtr& dataset) {
    auto allocator = std::make_shared<fixtures::MemoryRecordAllocator>();
    {
        auto index = vsag::Factory::CreateIndex(index_name, build_param, allocator.get()).value();
        REQUIRE(index->GetNumElements() == 0);
        if (index->CheckFeature(vsag::SUPPORT_ESTIMATE_MEMORY)) {
            auto data_size = dataset->base_->GetNumElements();
            auto estimate_memory = index->EstimateMemory(data_size);
            auto build_index = index->Build(dataset->base_);
            REQUIRE(build_index.has_value());
            auto real_memory = allocator->GetCurrentMemory();
            if (estimate_memory <= static_cast<uint64_t>(real_memory * 0.8) or
                estimate_memory >= static_cast<uint64_t>(real_memory * 1.2)) {
                WARN("estimate_memory failed");
            }

            REQUIRE(estimate_memory >= static_cast<uint64_t>(real_memory * 0.5));
            REQUIRE(estimate_memory <= static_cast<uint64_t>(real_memory * 1.5));
        }
    }
}

void
TestIndex::TestCheckIdExist(const TestIndex::IndexPtr& index, const TestDatasetPtr& dataset) {
    auto data_count = dataset->base_->GetNumElements();
    auto* ids = dataset->base_->GetIds();
    int N = 10;
    for (int i = 0; i < N; ++i) {
        auto good_id = ids[random() % data_count];
        REQUIRE(index->CheckIdExist(good_id) == true);
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

    auto create_index_result = vsag::Factory::CreateIndex(name, build_param);
    REQUIRE(create_index_result.has_value() == expect_success);
    auto index = create_index_result.value();
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

}  // namespace fixtures
