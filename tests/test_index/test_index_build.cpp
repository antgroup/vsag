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

}  // namespace fixtures
