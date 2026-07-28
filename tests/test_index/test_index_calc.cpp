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
    for (int64_t i = 0; i < query_count; ++i) {
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
                                     bool is_sparse) {
    if (not index->CheckFeature(vsag::SUPPORT_CAL_DISTANCE_BY_ID)) {
        return;
    }
    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    for (int64_t i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        tl::expected<DatasetPtr, vsag::Error> result;
        if (is_sparse) {
            result = index->CalDistanceById(query, gts->GetIds() + (i * gt_topK), gt_topK);
        } else {
            result = index->CalDistanceById(
                query->GetFloat32Vectors(), gts->GetIds() + (i * gt_topK), gt_topK);
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

    SECTION("test topk single-query batch") {
        if (not index->CheckFeature(vsag::SUPPORT_BATCH_CALC_DISTANCE_BY_ID)) {
            return;
        }
        int64_t topk = std::min(gt_topK, (int64_t)3);
        for (int64_t i = 0; i < query_count; ++i) {
            auto query = get_one_query(queries, i);
            tl::expected<DatasetPtr, vsag::Error> result;
            if (is_sparse) {
                result = index->CalDistanceById(
                    query, gts->GetIds() + (i * gt_topK), gt_topK, true, topk);
            } else {
                result = index->CalDistanceById(
                    query->GetFloat32Vectors(), gts->GetIds() + (i * gt_topK), gt_topK, true, topk);
            }
            REQUIRE(result.has_value());
            REQUIRE(result.value()->GetDim() == topk);
            REQUIRE(result.value()->GetIds() != nullptr);
            auto* topk_dists = result.value()->GetDistances();
            auto* topk_ids = result.value()->GetIds();
            for (int64_t j = 0; j < topk - 1; ++j) {
                REQUIRE(topk_dists[j] <= topk_dists[j + 1]);
            }
            for (int64_t j = 0; j < topk; ++j) {
                bool found = false;
                for (int64_t k = 0; k < gt_topK; ++k) {
                    if (topk_ids[j] == gts->GetIds()[i * gt_topK + k]) {
                        REQUIRE(std::abs(topk_dists[j] - gts->GetDistances()[i * gt_topK + k]) <
                                error);
                        found = true;
                        break;
                    }
                }
                REQUIRE(found);
            }
        }
    }
}

void
TestIndex::TestMultiQueryBatchCalcDistanceById(const IndexPtr& index,
                                               const TestDatasetPtr& dataset,
                                               float error,
                                               bool expected_success,
                                               bool is_sparse,
                                               bool expect_all_missing_on_failure) {
    if (not index->CheckFeature(vsag::SUPPORT_BATCH_CALC_DISTANCE_BY_ID)) {
        return;
    }
    auto queries = dataset->query_;
    auto num_queries = queries->GetNumElements();
    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    (void)is_sparse;
    if (num_queries < 2) {
        return;
    }

    // Use each query's own ground-truth IDs as the row-major batch ID matrix.
    const int64_t* batch_ids = gts->GetIds();
    auto multi_result = index->CalDistanceById(queries, batch_ids, gt_topK);
    if (not expected_success) {
        if (not expect_all_missing_on_failure) {
            REQUIRE_FALSE(multi_result.has_value());
            return;
        }
        REQUIRE(multi_result.has_value());
        auto* distances = multi_result.value()->GetDistances();
        for (int64_t q = 0; q < num_queries; ++q) {
            for (int64_t j = 0; j < gt_topK; ++j) {
                REQUIRE(distances[q * gt_topK + j] == -1);
            }
        }
        return;
    }
    REQUIRE(multi_result.has_value());
    auto* multi_distances = multi_result.value()->GetDistances();

    for (int64_t q = 0; q < num_queries; ++q) {
        auto single_query = get_one_query(queries, q);
        const int64_t* row_ids = batch_ids + q * gt_topK;
        auto single_result = index->CalDistanceById(single_query, row_ids, gt_topK);
        REQUIRE(single_result.has_value());
        auto* single_distances = single_result.value()->GetDistances();
        for (int64_t j = 0; j < gt_topK; ++j) {
            float expected_dist = single_distances[j];
            float actual_dist = multi_distances[q * gt_topK + j];
            REQUIRE(std::abs(expected_dist - actual_dist) < error);
        }
    }

    SECTION("test non-existing id with multi-query") {
        int64_t test_num = 5;
        std::vector<int64_t> mixed_ids(num_queries * test_num);
        for (int64_t q = 0; q < num_queries; ++q) {
            const int64_t* row_ids = batch_ids + q * gt_topK;
            for (int64_t i = 0; i < test_num; ++i) {
                mixed_ids[q * test_num + i] =
                    (i % 2 == 0) ? row_ids[i % gt_topK] : -(q * test_num + i + 1);
            }
        }
        auto r2 = index->CalDistanceById(queries, mixed_ids.data(), test_num);
        REQUIRE(r2.has_value());
        auto* d2 = r2.value()->GetDistances();
        for (int64_t q = 0; q < num_queries; ++q) {
            for (int64_t i = 0; i < test_num; ++i) {
                if (i % 2 != 0) {
                    REQUIRE(d2[q * test_num + i] == -1);
                }
            }
        }

        auto topk_result = index->CalDistanceById(queries, mixed_ids.data(), test_num, true, 3);
        REQUIRE(topk_result.has_value());
        auto* topk_dists = topk_result.value()->GetDistances();
        auto* topk_ids = topk_result.value()->GetIds();
        for (int64_t q = 0; q < num_queries; ++q) {
            for (int64_t i = 0; i < 3; ++i) {
                REQUIRE(topk_ids[q * 3 + i] >= 0);
                REQUIRE(topk_dists[q * 3 + i] != -1.0F);
            }
        }
    }

    SECTION("test topk multi-query batch") {
        int64_t topk = std::min(gt_topK, (int64_t)3);
        auto topk_result = index->CalDistanceById(queries, batch_ids, gt_topK, true, topk);
        REQUIRE(topk_result.has_value());
        auto* topk_dists = topk_result.value()->GetDistances();
        auto* topk_ids_out = topk_result.value()->GetIds();
        REQUIRE(topk_dists != nullptr);
        REQUIRE(topk_ids_out != nullptr);
        for (int64_t q = 0; q < num_queries; ++q) {
            const float* row_dists = topk_dists + q * topk;
            const int64_t* row_ids_out = topk_ids_out + q * topk;
            for (int64_t i = 0; i < topk - 1; ++i) {
                REQUIRE(row_dists[i] <= row_dists[i + 1]);
            }
            for (int64_t i = 0; i < topk; ++i) {
                bool found = false;
                for (int64_t j = 0; j < gt_topK; ++j) {
                    if (row_ids_out[i] == batch_ids[q * gt_topK + j]) {
                        float expected = multi_distances[q * gt_topK + j];
                        REQUIRE(std::abs(row_dists[i] - expected) < error);
                        found = true;
                        break;
                    }
                }
                REQUIRE(found);
            }
        }
    }

    SECTION("test topk > count returns all") {
        int64_t big_topk = gt_topK + 100;
        auto big_result = index->CalDistanceById(queries, batch_ids, gt_topK, true, big_topk);
        REQUIRE(big_result.has_value());
        REQUIRE(big_result.value()->GetDim() == gt_topK);
        auto* big_dists = big_result.value()->GetDistances();
        for (int64_t q = 0; q < num_queries; ++q) {
            for (int64_t j = 0; j < gt_topK; ++j) {
                REQUIRE(std::abs(big_dists[q * gt_topK + j] - multi_distances[q * gt_topK + j]) <
                        error);
            }
        }
    }

    SECTION("test topk -1 no ids in result") {
        auto default_result = index->CalDistanceById(queries, batch_ids, gt_topK);
        REQUIRE(default_result.has_value());
        REQUIRE(default_result.value()->GetIds() == nullptr);
        REQUIRE(default_result.value()->GetDim() == gt_topK);
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
        const auto base_id = dataset->base_->GetIds()[j];
        res_max_id = std::max(res_max_id, base_id);
        res_min_id = std::min(res_min_id, base_id);
    }
    get_min_max_res = index->GetMinAndMaxId();
    REQUIRE(get_min_max_res.has_value() == true);
    int64_t min_id = get_min_max_res.value().first;
    int64_t max_id = get_min_max_res.value().second;

    REQUIRE(min_id == res_min_id);
    REQUIRE(max_id == res_max_id);
}

}  // namespace fixtures
