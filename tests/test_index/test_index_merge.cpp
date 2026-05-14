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

}  // namespace fixtures
