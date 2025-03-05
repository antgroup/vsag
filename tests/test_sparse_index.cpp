
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
#include <catch2/generators/catch_generators.hpp>

#include "test_index.h"
#include "fixtures/test_dataset_pool.h"

namespace fixtures {

class SparseTestIndex : public fixtures::TestIndex {
public:
    static TestDatasetPool pool;
    constexpr static uint64_t base_count = 1000;

    static const std::string build_param;
    static const std::string search_param;
};
TestDatasetPool SparseTestIndex::pool{};

} // namespace fixtures

TEST_CASE_PERSISTENT_FIXTURE(fixtures::SparseTestIndex, "SparseIndex Build and Search", "[ft][sparse_index]") {
    auto index = TestFactory("sparse_index", build_param, true);
    auto dataset = pool.GetSparseDatasetAndCreate(base_count, 0.8);
    TestContinueAdd(index, dataset, true);
    TestKnnSearch(index, dataset, search_param, 0.99, true);
    TestRangeSearch(index, dataset, search_param, 0.99, 10, true);
    TestRangeSearch(index, dataset, search_param, 0.49, 5, true);
    TestFilterSearch(index, dataset, search_param, 0.99, true);
}
