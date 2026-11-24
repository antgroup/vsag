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

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

namespace {

bool
use_dataset_query(const vsag::DatasetPtr& queries, bool is_sparse) {
    return is_sparse || queries->GetSparseVectors() != nullptr ||
           queries->GetInt8Vectors() != nullptr || queries->GetFloat16Vectors() != nullptr;
}

}  // namespace

// General index suites include lossy storage without retained FP32 vectors. Compare the
// formal single/batch APIs at both precision settings; exact FP32-oracle checks remain in
// dedicated distance tests rather than assuming every quantizer reconstructs originals.
void
TestIndex::TestStoredDistanceConsistency(const IndexPtr& index, const TestDatasetPtr& dataset) {
    REQUIRE(index->CheckFeature(vsag::SUPPORT_CAL_DISTANCE_BY_ID));
    REQUIRE(index->CheckFeature(vsag::SUPPORT_BATCH_CALC_DISTANCE_BY_ID));
    const auto queries = dataset->query_;
    const auto count = dataset->top_k;
    for (bool precise : {false, true}) {
        for (int64_t q = 0; q < queries->GetNumElements(); ++q) {
            auto query = get_one_query(queries, q);
            const auto* ids = dataset->ground_truth_->GetIds() + q * count;
            auto batch = index->CalcDistancesById(query, ids, count, precise);
            REQUIRE(batch.has_value());
            for (int64_t j = 0; j < count; ++j) {
                auto single = index->CalcDistanceById(query, ids[j], precise);
                REQUIRE(single.has_value());
                REQUIRE(std::isfinite(single.value()));
                const float tolerance = 1e-5F * std::max(1.0F, std::abs(single.value()));
                REQUIRE(std::abs(single.value() - batch.value()->GetDistances()[j]) <= tolerance);
            }
        }
    }
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
    for (int64_t i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        for (auto j = 0; j < gt_topK; ++j) {
            auto id = gts->GetIds()[i * gt_topK + j];
            auto dist = gts->GetDistances()[i * gt_topK + j];
            tl::expected<float, vsag::Error> result;
            if (is_sparse || queries->GetSparseVectors() != nullptr ||
                queries->GetInt8Vectors() != nullptr || queries->GetFloat16Vectors() != nullptr) {
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
            if (use_dataset_query(queries, is_sparse)) {
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
        if (is_sparse || queries->GetSparseVectors() != nullptr ||
            queries->GetInt8Vectors() != nullptr || queries->GetFloat16Vectors() != nullptr) {
            result = index->CalcDistancesById(query, gts->GetIds() + (i * gt_topK), gt_topK);
        } else {
            result = index->CalcDistancesById(
                query->GetFloat32Vectors(), gts->GetIds() + (i * gt_topK), gt_topK);
        }
        if (not expected_success) {
            return;
        }
        REQUIRE(result.has_value());
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
        if (is_sparse || queries->GetSparseVectors() != nullptr ||
            queries->GetInt8Vectors() != nullptr || queries->GetFloat16Vectors() != nullptr) {
            result = index->CalcDistancesById(queries, no_exist_ids.data(), test_num);
        } else {
            result = index->CalcDistancesById(
                queries->GetFloat32Vectors(), no_exist_ids.data(), test_num);
        }
        REQUIRE(result.has_value());
        const auto* valids = result.value()->GetValids();
        REQUIRE(valids != nullptr);
        for (int i = 0; i < test_num; ++i) {
            REQUIRE(std::isnan(result.value()->GetDistances()[i]));
            REQUIRE_FALSE(valids[i]);
        }
    }

    {
        INFO("test mixed valid and missing ids");
        auto query = get_one_query(queries, 0);
        const auto* gt_ids = gts->GetIds();
        const int64_t valid_a = gt_ids[0];
        const int64_t valid_b = gt_ids[gt_topK > 1 ? 1 : 0];
        // Layout: missing, valid, missing, valid, missing.
        std::vector<int64_t> mixed_ids = {-101, valid_a, -102, valid_b, -103};
        const auto count = static_cast<int64_t>(mixed_ids.size());
        const std::vector<bool> expected_valid = {false, true, false, true, false};
        auto calc = [&](int64_t topk) {
            if (use_dataset_query(queries, is_sparse)) {
                return index->CalcDistancesById(query, mixed_ids.data(), count, true, topk);
            }
            return index->CalcDistancesById(
                query->GetFloat32Vectors(), mixed_ids.data(), count, true, topk);
        };
        {
            auto all = calc(-1);
            REQUIRE(all.has_value());
            REQUIRE(all.value()->GetDim() == count);
            const auto* valids = all.value()->GetValids();
            const auto* dists = all.value()->GetDistances();
            REQUIRE(valids != nullptr);
            for (int64_t i = 0; i < count; ++i) {
                REQUIRE(valids[i] == expected_valid[i]);
                REQUIRE(std::isnan(dists[i]) == not expected_valid[i]);
            }
            REQUIRE(std::abs(dists[1] - gts->GetDistances()[0]) < error);
        }
        if (not index->CheckFeature(vsag::SUPPORT_BATCH_CALC_DISTANCE_BY_ID)) {
            return;
        }
        {
            // Fewer valid entries (2) than topk (4): valid entries first, invalid ranked last
            // with NaN distance; Valids is aligned with the output order.
            auto top = calc(4);
            REQUIRE(top.has_value());
            REQUIRE(top.value()->GetDim() == 4);
            const auto* valids = top.value()->GetValids();
            const auto* dists = top.value()->GetDistances();
            const auto* out_ids = top.value()->GetIds();
            REQUIRE(valids != nullptr);
            REQUIRE(out_ids != nullptr);
            REQUIRE(valids[0]);
            REQUIRE(valids[1]);
            REQUIRE_FALSE(valids[2]);
            REQUIRE_FALSE(valids[3]);
            REQUIRE(std::isfinite(dists[0]));
            REQUIRE(std::isfinite(dists[1]));
            REQUIRE(dists[0] <= dists[1]);
            REQUIRE(std::isnan(dists[2]));
            REQUIRE(std::isnan(dists[3]));
            REQUIRE(((out_ids[0] == valid_a and out_ids[1] == valid_b) or
                     (out_ids[0] == valid_b and out_ids[1] == valid_a)));
            REQUIRE(out_ids[2] < 0);
            REQUIRE(out_ids[3] < 0);
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
            if (is_sparse || queries->GetSparseVectors() != nullptr ||
                queries->GetInt8Vectors() != nullptr || queries->GetFloat16Vectors() != nullptr) {
                result = index->CalcDistancesById(
                    query, gts->GetIds() + (i * gt_topK), gt_topK, true, topk);
            } else {
                result = index->CalcDistancesById(
                    query->GetFloat32Vectors(), gts->GetIds() + (i * gt_topK), gt_topK, true, topk);
            }
            REQUIRE(result.has_value());
            REQUIRE(result.value()->GetDim() == topk);
            REQUIRE(result.value()->GetIds() != nullptr);
            REQUIRE(result.value()->GetValids() != nullptr);
            auto* topk_dists = result.value()->GetDistances();
            auto* topk_ids = result.value()->GetIds();
            for (int64_t j = 0; j < topk; ++j) {
                REQUIRE(result.value()->GetValids()[j]);
            }
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
TestIndex::TestCalcDistanceByIdRemovedLabel(const IndexPtr& index,
                                            const TestDatasetPtr& dataset,
                                            bool is_sparse) {
    REQUIRE(index->CheckFeature(vsag::SUPPORT_CAL_DISTANCE_BY_ID));
    REQUIRE(index->CheckFeature(vsag::SUPPORT_BATCH_CALC_DISTANCE_BY_ID));
    const auto base_count = dataset->base_->GetNumElements();
    REQUIRE(base_count > 1);
    const auto* base_ids = dataset->base_->GetIds();
    const int64_t removed_id = base_ids[0];
    const int64_t existing_id = base_ids[base_count - 1];
    REQUIRE(removed_id != existing_id);

    auto remove_result = index->Remove(removed_id, vsag::RemoveMode::MARK_REMOVE);
    REQUIRE(remove_result.has_value());
    REQUIRE(remove_result.value() == 1);

    const auto queries = dataset->query_;
    auto query = get_one_query(queries, 0);
    const bool use_dataset = use_dataset_query(queries, is_sparse);

    {
        INFO("single-ID distance of a removed label must fail as a missing label");
        tl::expected<float, vsag::Error> result;
        if (use_dataset) {
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
        if (use_dataset) {
            result = index->CalcDistancesById(query, batch_ids.data(), count);
        } else {
            result = index->CalcDistancesById(query->GetFloat32Vectors(), batch_ids.data(), count);
        }
        REQUIRE(result.has_value());
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
    auto multi_result = index->CalcDistancesById(queries, batch_ids, gt_topK);
    if (not expected_success) {
        if (not expect_all_missing_on_failure) {
            REQUIRE_FALSE(multi_result.has_value());
            return;
        }
        REQUIRE(multi_result.has_value());
        auto* distances = multi_result.value()->GetDistances();
        for (int64_t q = 0; q < num_queries; ++q) {
            for (int64_t j = 0; j < gt_topK; ++j) {
                REQUIRE(std::isnan(distances[q * gt_topK + j]));
                REQUIRE_FALSE(multi_result.value()->GetValids()[q * gt_topK + j]);
            }
        }
        return;
    }
    REQUIRE(multi_result.has_value());
    auto* multi_distances = multi_result.value()->GetDistances();

    for (int64_t q = 0; q < num_queries; ++q) {
        auto single_query = get_one_query(queries, q);
        const int64_t* row_ids = batch_ids + q * gt_topK;
        auto single_result = index->CalcDistancesById(single_query, row_ids, gt_topK);
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
        auto r2 = index->CalcDistancesById(queries, mixed_ids.data(), test_num);
        REQUIRE(r2.has_value());
        auto* d2 = r2.value()->GetDistances();
        const auto* v2 = r2.value()->GetValids();
        REQUIRE(v2 != nullptr);
        for (int64_t q = 0; q < num_queries; ++q) {
            for (int64_t i = 0; i < test_num; ++i) {
                const bool expect_valid = i % 2 == 0;
                REQUIRE(v2[q * test_num + i] == expect_valid);
                REQUIRE(std::isnan(d2[q * test_num + i]) == not expect_valid);
            }
        }

        auto topk_result = index->CalcDistancesById(queries, mixed_ids.data(), test_num, true, 3);
        REQUIRE(topk_result.has_value());
        auto* topk_dists = topk_result.value()->GetDistances();
        auto* topk_ids = topk_result.value()->GetIds();
        const auto* topk_valids = topk_result.value()->GetValids();
        REQUIRE(topk_valids != nullptr);
        for (int64_t q = 0; q < num_queries; ++q) {
            for (int64_t i = 0; i < 3; ++i) {
                REQUIRE(topk_ids[q * 3 + i] >= 0);
                REQUIRE(topk_valids[q * 3 + i]);
                REQUIRE(std::isfinite(topk_dists[q * 3 + i]));
            }
        }

        // topk larger than the number of valid entries: invalid entries are ranked last per
        // row and the output-aligned Valids propagate through the multi-query adapter.
        const int64_t wide_topk = test_num;
        auto wide_result =
            index->CalcDistancesById(queries, mixed_ids.data(), test_num, true, wide_topk);
        REQUIRE(wide_result.has_value());
        const auto* wide_valids = wide_result.value()->GetValids();
        const auto* wide_dists = wide_result.value()->GetDistances();
        REQUIRE(wide_valids != nullptr);
        const int64_t valid_per_row = (test_num + 1) / 2;
        for (int64_t q = 0; q < num_queries; ++q) {
            for (int64_t i = 0; i < test_num; ++i) {
                const bool expect_valid = i < valid_per_row;
                REQUIRE(wide_valids[q * test_num + i] == expect_valid);
                REQUIRE(std::isnan(wide_dists[q * test_num + i]) == not expect_valid);
            }
        }
    }

    SECTION("test topk multi-query batch") {
        int64_t topk = std::min(gt_topK, (int64_t)3);
        auto topk_result = index->CalcDistancesById(queries, batch_ids, gt_topK, true, topk);
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
        auto big_result = index->CalcDistancesById(queries, batch_ids, gt_topK, true, big_topk);
        REQUIRE(big_result.has_value());
        REQUIRE(big_result.value()->GetDim() == gt_topK);
        auto* big_dists = big_result.value()->GetDistances();
        // Compare against single-query topk reference (self-consistent with stored distances)
        // rather than raw FP32 ground-truth, which lossy representations cannot replicate.
        for (int64_t q = 0; q < num_queries; ++q) {
            auto single_query = get_one_query(queries, q);
            const int64_t* row_ids = batch_ids + q * gt_topK;
            auto single_topk =
                index->CalcDistancesById(single_query, row_ids, gt_topK, true, big_topk);
            REQUIRE(single_topk.has_value());
            REQUIRE(single_topk.value()->GetDim() == gt_topK);
            auto* ref_dists = single_topk.value()->GetDistances();
            for (int64_t j = 0; j < gt_topK; ++j) {
                REQUIRE(std::abs(big_dists[q * gt_topK + j] - ref_dists[j]) < error);
            }
        }
    }

    SECTION("test topk -1 no ids in result") {
        auto default_result = index->CalcDistancesById(queries, batch_ids, gt_topK);
        REQUIRE(default_result.has_value());
        REQUIRE(default_result.value()->GetIds() == nullptr);
        REQUIRE(default_result.value()->GetDim() == gt_topK);
        const auto* default_valids = default_result.value()->GetValids();
        REQUIRE(default_valids != nullptr);
        for (int64_t i = 0; i < num_queries * gt_topK; ++i) {
            REQUIRE(default_valids[i]);
        }
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

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

}  // namespace fixtures
