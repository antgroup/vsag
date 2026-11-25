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

namespace {

/// Batch distance results must describe their shape so callers can index the flat
/// Distances()/Valids() arrays: NumElements() * Dim() == count.
void
RequireBatchDistanceShape(const vsag::DatasetPtr& result, int64_t count) {
    REQUIRE(result->GetNumElements() > 0);
    REQUIRE(result->GetDim() > 0);
    REQUIRE(result->GetNumElements() * result->GetDim() == count);
}

}  // namespace

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
    if (not expected_success) {
        return;
    }
    {
        INFO("single-ID distance of a missing label must fail");
        auto query = get_one_query(queries, 0);
        for (int64_t missing_id : {int64_t{-1}, int64_t{-12345}, int64_t{1} << 40}) {
            tl::expected<float, vsag::Error> result;
            if (is_sparse) {
                result = index->CalcDistanceById(query, missing_id);
            } else {
                result = index->CalcDistanceById(query->GetFloat32Vectors(), missing_id);
            }
            REQUIRE_FALSE(result.has_value());
            REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
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
            // Dense indexes do not support DatasetPtr queries: existing ids fail with an error;
            // ids that are all missing (e.g. an empty index) are only reported as invalid.
            auto dataset_result =
                index->CalDistanceById(query, gts->GetIds() + (i * gt_topK), gt_topK);
            if (dataset_result.has_value()) {
                const auto* dataset_valids = dataset_result.value()->GetValids();
                REQUIRE(dataset_valids != nullptr);
                for (auto j = 0; j < gt_topK; ++j) {
                    REQUIRE_FALSE(dataset_valids[j]);
                }
            }
        }
        if (not expected_success) {
            return;
        }
        REQUIRE(result.has_value());
        RequireBatchDistanceShape(result.value(), gt_topK);
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
        REQUIRE(result.has_value());
        RequireBatchDistanceShape(result.value(), test_num);
        const auto* valids = result.value()->GetValids();
        REQUIRE(valids != nullptr);
        for (int i = 0; i < test_num; ++i) {
            REQUIRE(std::isnan(result.value()->GetDistances()[i]));
            REQUIRE_FALSE(valids[i]);
        }
        queries->NumElements(query_count);
    }

    SECTION("test mixed valid and missing ids") {
        auto query = get_one_query(queries, 0);
        const auto* gt_ids = gts->GetIds();
        const int64_t valid_a = gt_ids[0];
        const int64_t valid_b = gt_ids[gt_topK > 1 ? 1 : 0];
        // Layout: missing, valid, missing, valid, missing.
        std::vector<int64_t> mixed_ids = {-101, valid_a, -102, valid_b, -103};
        const auto count = static_cast<int64_t>(mixed_ids.size());
        const std::vector<bool> expected_valid = {false, true, false, true, false};
        tl::expected<DatasetPtr, vsag::Error> result;
        if (is_sparse) {
            result = index->CalDistanceById(query, mixed_ids.data(), count);
        } else {
            result = index->CalDistanceById(query->GetFloat32Vectors(), mixed_ids.data(), count);
        }
        REQUIRE(result.has_value());
        RequireBatchDistanceShape(result.value(), count);
        const auto* valids = result.value()->GetValids();
        const auto* dists = result.value()->GetDistances();
        REQUIRE(valids != nullptr);
        for (int64_t i = 0; i < count; ++i) {
            REQUIRE(valids[i] == expected_valid[i]);
            REQUIRE(std::isnan(dists[i]) == not expected_valid[i]);
        }
        REQUIRE(std::abs(dists[1] - gts->GetDistances()[0]) < error);
    }

    SECTION("existing ids are all valid") {
        auto query = get_one_query(queries, 0);
        tl::expected<DatasetPtr, vsag::Error> result;
        if (is_sparse) {
            result = index->CalDistanceById(query, gts->GetIds(), gt_topK);
        } else {
            result = index->CalDistanceById(query->GetFloat32Vectors(), gts->GetIds(), gt_topK);
        }
        REQUIRE(result.has_value());
        RequireBatchDistanceShape(result.value(), gt_topK);
        const auto* valids = result.value()->GetValids();
        REQUIRE(valids != nullptr);
        for (int64_t i = 0; i < gt_topK; ++i) {
            REQUIRE(valids[i]);
            REQUIRE(std::isfinite(result.value()->GetDistances()[i]));
        }
    }
}

void
TestIndex::TestCalcDistanceByIdRemovedLabel(const IndexPtr& index,
                                            const TestDatasetPtr& dataset,
                                            bool is_sparse) {
    if (not index->CheckFeature(vsag::SUPPORT_CAL_DISTANCE_BY_ID)) {
        return;
    }
    const auto base_count = dataset->base_->GetNumElements();
    REQUIRE(base_count > 1);
    const auto* base_ids = dataset->base_->GetIds();
    const int64_t removed_id = base_ids[0];
    const int64_t existing_id = base_ids[base_count - 1];
    REQUIRE(removed_id != existing_id);

    auto remove_result = index->Remove(removed_id, vsag::RemoveMode::MARK_REMOVE);
    REQUIRE(remove_result.has_value());
    REQUIRE(remove_result.value() == 1);

    auto query = get_one_query(dataset->query_, 0);
    {
        INFO("single-ID distance of a removed label must fail as a missing label");
        tl::expected<float, vsag::Error> result;
        if (is_sparse) {
            result = index->CalcDistanceById(query, removed_id);
        } else {
            result = index->CalcDistanceById(query->GetFloat32Vectors(), removed_id);
        }
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }
    {
        INFO("batch distance marks the removed label invalid instead of failing");
        std::vector<int64_t> batch_ids = {removed_id, existing_id};
        const auto count = static_cast<int64_t>(batch_ids.size());
        tl::expected<DatasetPtr, vsag::Error> result;
        if (is_sparse) {
            result = index->CalDistanceById(query, batch_ids.data(), count);
        } else {
            result = index->CalDistanceById(query->GetFloat32Vectors(), batch_ids.data(), count);
        }
        REQUIRE(result.has_value());
        RequireBatchDistanceShape(result.value(), count);
        const auto* dists = result.value()->GetDistances();
        const auto* valids = result.value()->GetValids();
        REQUIRE(dists != nullptr);
        REQUIRE(valids != nullptr);
        REQUIRE_FALSE(valids[0]);
        REQUIRE(std::isnan(dists[0]));
        REQUIRE(valids[1]);
        REQUIRE(std::isfinite(dists[1]));
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
