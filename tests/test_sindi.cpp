
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

#include "fixtures/test_dataset_pool.h"
#include "test_index.h"

namespace fixtures {

struct SINDIParam {
    bool use_reorder = true;
    float doc_prune_ratio = 0.0;
    int window_size = 10000;
    bool deserialize_without_footer = false;
    int term_id_limit = 2000;
};

class SINDITestIndex : public fixtures::TestIndex {
public:
    static TestDatasetPool pool;
    constexpr static uint64_t base_count = 1000;
    constexpr static const char* search_param = R"(
        {
            "sindi":
            {
                "n_candidate": 20,
                "query_prune_ratio": 0.0,
                "term_prune_ratio": 0.0
            }
        })";

    static std::string
    GenerateBuildParameter(const SINDIParam& param) {
        constexpr static const char* build_param_template = R"(
        {{
            "dim": 16,
            "dtype": "sparse",
            "metric_type": "ip",
            "index_param": {{
                "use_reorder": {},
                "doc_prune_ratio": {},
                "window_size": {},
                "term_id_limit": {},
                "deserialize_without_footer": {}
            }}
        }})";
        return fmt::format(build_param_template,
                           param.use_reorder,
                           param.doc_prune_ratio,
                           param.window_size,
                           param.term_id_limit,
                           param.deserialize_without_footer);
    }
};
TestDatasetPool SINDITestIndex::pool{};

}  // namespace fixtures

TEST_CASE_PERSISTENT_FIXTURE(fixtures::SINDITestIndex,
                             "Invalid Build and Search Parameter",
                             "[ft][sindi]") {
    SECTION("invalid doc_prune_ratio") {
        fixtures::SINDIParam param;
        param.doc_prune_ratio = 0.99;
        REQUIRE_THROWS(
            TestFactory("sindi", fixtures::SINDITestIndex::GenerateBuildParameter(param), false));
        param.doc_prune_ratio = -0.1;
        REQUIRE_THROWS(
            TestFactory("sindi", fixtures::SINDITestIndex::GenerateBuildParameter(param), false));
    }

    SECTION("invalid window_size") {
        fixtures::SINDIParam param;
        param.window_size = 5000;
        REQUIRE_THROWS(
            TestFactory("sindi", fixtures::SINDITestIndex::GenerateBuildParameter(param), false));
        param.window_size = 1100000;
        REQUIRE_THROWS(
            TestFactory("sindi", fixtures::SINDITestIndex::GenerateBuildParameter(param), false));
    }
    fixtures::SINDIParam param;
    param.window_size = 10000;
    auto build_param = fixtures::SINDITestIndex::GenerateBuildParameter(param);
    auto index = TestFactory("sindi", build_param, true);
    auto dataset = pool.GetSparseDatasetAndCreate(base_count, 128, 0.8);
    REQUIRE(index->GetIndexType() == vsag::IndexType::SINDI);
    TestBuildIndex(index, dataset, true);
    {
        auto invalid_search_param = R"({
            "sindi": {
                "n_candidate": -1,
                "query_prune_ratio": 0.0,
                "term_prune_ratio": 0.0
            }
        })";
        TestKnnSearch(index, dataset, invalid_search_param, 0.99, false);
        invalid_search_param = R"({
            "sindi":{
                "n_candidate": 10,
                "query_prune_ratio": 1.2,
                "term_prune_ratio": 0.0
            }
        })";
        TestKnnSearch(index, dataset, invalid_search_param, 0.99, false);
        invalid_search_param = R"({
            "sindi":{
                "n_candidate": 10,
                "query_prune_ratio": 0.0,
                "term_prune_ratio": -0.1
            }
        })";
        TestKnnSearch(index, dataset, invalid_search_param, 0.99, false);
    }
    // invalid multi data
    int64_t ids[2] = {114, 514};
    vsag::SparseVector invalid_sv_array[2];
    std::vector<uint32_t> sv_ids = {100};
    std::vector<float> sv_vals = {0.5};
    invalid_sv_array[0].len_ = 1;
    invalid_sv_array[0].ids_ = sv_ids.data();
    invalid_sv_array[0].vals_ = sv_vals.data();
    invalid_sv_array[1].len_ = 0;

    auto invalid_data = vsag::Dataset::Make();
    invalid_data->NumElements(2)->SparseVectors(invalid_sv_array)->Ids(ids)->Owner(false);
    auto insert_result = index->Add(invalid_data);
    REQUIRE(insert_result.has_value());
    auto failed_ids = insert_result.value();
    REQUIRE(failed_ids.size() == 1);
    REQUIRE(failed_ids[0] == ids[1]);

    // invalid single data
    vsag::SparseVector sparse_vector;
    int64_t id = 7777;
    sparse_vector.len_ = 0;
    auto data = vsag::Dataset::Make();
    data->NumElements(1)->Owner(false)->SparseVectors(&sparse_vector)->Ids(&id);
    insert_result = index->Add(data);
    REQUIRE(insert_result.has_value());
    failed_ids = insert_result.value();
    REQUIRE(failed_ids[0] == id);
    auto search_result = index->KnnSearch(data, 1, search_param);
    REQUIRE_FALSE(search_result.has_value());
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::SINDITestIndex, "SINDI Build and Search", "[ft][sindi]") {
    fixtures::SINDIParam param;
    param.use_reorder = GENERATE(true, false);
    auto build_param = fixtures::SINDITestIndex::GenerateBuildParameter(param);
    auto index = TestFactory("sindi", build_param, true);
    auto dataset = pool.GetSparseDatasetAndCreate(base_count, 128, 0.8);
    REQUIRE(index->GetIndexType() == vsag::IndexType::SINDI);
    TestContinueAdd(index, dataset, true);
    TestGetRawVectorByIds(index, dataset, true);
    TestKnnSearch(index, dataset, search_param, 0.99, true);
    TestSearchAllocator(index, dataset, search_param, 0.99, true);
    TestRangeSearch(index, dataset, search_param, 0.99, 10, true);
    TestRangeSearch(index, dataset, search_param, 0.49, 5, true);
    TestFilterSearch(index, dataset, search_param, 0.99, true);
    TestConcurrentKnnSearch(index, dataset, search_param, 0.99, true);
    TestGetMinAndMaxId(index, dataset, true);
    TestCalcDistanceById(index, dataset, 1e-4, true, true);
    TestBatchCalcDistanceById(index, dataset, 1e-4, true, true);
    TestUpdateId(index, dataset, search_param, true);
    TestEstimateMemory("sindi", build_param, dataset);
    TestIndexStatus(index);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::SINDITestIndex, "SINDI Concurrent", "[ft][sindi]") {
    fixtures::SINDIParam param;
    param.use_reorder = GENERATE(true, false);
    auto build_param = fixtures::SINDITestIndex::GenerateBuildParameter(param);
    auto index = TestFactory("sindi", build_param, true);
    auto dataset = pool.GetSparseDatasetAndCreate(base_count, 128, 0.8);
    REQUIRE(index->GetIndexType() == vsag::IndexType::SINDI);
    TestConcurrentAddSearch(index, dataset, search_param, 0.99, true);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::SINDITestIndex, "SINDI Serialize File", "[ft][sindi]") {
    fixtures::SINDIParam param;
    param.deserialize_without_footer = GENERATE(true, false);
    param.use_reorder = GENERATE(true, false);
    auto build_param = fixtures::SINDITestIndex::GenerateBuildParameter(param);
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("ip");
    const std::string name = "sindi";
    vsag::Options::Instance().set_block_size_limit(size);
    auto index = TestFactory(name, build_param, true);
    SECTION("serialize empty index") {
        auto index2 = TestFactory(name, build_param, true);
        auto serialize_binary = index->Serialize();
        REQUIRE(serialize_binary.has_value());
        auto deserialize_index = index2->Deserialize(serialize_binary.value());
        REQUIRE(deserialize_index.has_value());
    }
    auto dataset = pool.GetSparseDatasetAndCreate(base_count, 128, 0.8);
    TestBuildIndex(index, dataset, true);
    SECTION("serialize/deserialize by binary") {
        auto index2 = TestFactory(name, build_param, true);
        TestSerializeBinarySet(index, index2, dataset, search_param, true);
    }
    SECTION("serialize/deserialize by readerset") {
        auto index2 = TestFactory(name, build_param, true);
        TestSerializeReaderSet(index, index2, dataset, search_param, name, true);
    }
    SECTION("serialize/deserialize by file") {
        auto index2 = TestFactory(name, build_param, true);
        TestSerializeFile(index, index2, dataset, search_param, true);
    }
    vsag::Options::Instance().set_block_size_limit(origin_size);
}
