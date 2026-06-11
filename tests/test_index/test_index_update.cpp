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

}  // namespace fixtures
