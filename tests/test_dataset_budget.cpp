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

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

#include "framework/test_dataset_pool.h"

TEST_CASE("Test dataset budget query count is part of the cache key", "[ft][test_budget]") {
    fixtures::TestDatasetPool pool;
    auto small = pool.GetDatasetAndCreate(8, 32, "l2", false, 0.8F, 0, 16, "dense", 4);
    auto large = pool.GetDatasetAndCreate(8, 32, "l2", false, 0.8F, 0, 16, "dense", 12);
    CHECK(small != large);
    CHECK(small->query_->GetNumElements() == 4);
    CHECK(small->ground_truth_->GetNumElements() == 4);
    CHECK(large->query_->GetNumElements() == 12);
    CHECK(large->ground_truth_->GetNumElements() == 12);
    CHECK(pool.GetDatasetAndCreate(8, 32, "l2", false, 0.8F, 0, 16, "dense", 4) == small);
    CHECK(pool.GetDatasetAndCreate(8, 32)->query_->GetNumElements() == 100);
}

TEST_CASE("Duplicate dataset budget query counts remain independent", "[ft][test_budget]") {
    fixtures::TestDatasetPool pool;
    auto small = pool.GetDuplicateDataset(8, 32, "l2", false, 0.8F, 0, 4);
    auto large = pool.GetDuplicateDataset(8, 32, "l2", false, 0.8F, 0, 12);
    CHECK(small != large);
    CHECK(small->query_->GetNumElements() == 4);
    CHECK(large->query_->GetNumElements() == 12);
    CHECK(pool.GetDuplicateDataset(8, 32, "l2", false, 0.8F, 0, 4) == small);
    REQUIRE_THROWS_AS(fixtures::TestDataset::CreateTestDataset(
                          8, 32, "l2", false, 0.8F, "dense", 0, false, 16, true, 0),
                      std::invalid_argument);
}
