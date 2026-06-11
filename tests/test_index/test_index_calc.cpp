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
            REQUIRE_FALSE(
                index->CalDistanceById(query, gts->GetIds() + (i * gt_topK), gt_topK).has_value());
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
