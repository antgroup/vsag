
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
#include <limits>

#include "functest.h"
#include "test_index.h"
#include "vsag/vsag.h"

namespace fixtures {
class HNSWTestIndex : public fixtures::TestIndex {
public:
    static std::string
    GenerateHNSWBuildParametersString(const std::string& metric_type,
                                      int64_t dim,
                                      std::string dtype = "float32");

    static void
    TestGeneral(const IndexPtr& index,
                const TestDatasetPtr& dataset,
                const std::string& search_param,
                float recall,
                bool expect_success = true);

    static TestDatasetPool pool;

    static std::vector<int> dims;

    static std::vector<float> valid_ratios;

    constexpr static uint64_t base_count = 1000;

    constexpr static const char* search_param_tmp = R"(
        {{
            "hnsw": {{
                "ef_search": {},
                "skip_ratio": 0.3
            }}
        }})";
};

TestDatasetPool HNSWTestIndex::pool{};
std::vector<int> HNSWTestIndex::dims = fixtures::get_common_used_dims(2, RandomValue(0, 999));
std::vector<float> HNSWTestIndex::valid_ratios{0.01, 0.05, 0.99};

std::string
HNSWTestIndex::GenerateHNSWBuildParametersString(const std::string& metric_type,
                                                 int64_t dim,
                                                 std::string dtype) {
    // should be aligned with HGraphTestIndex::GenerateHGraphBuildParametersString
    constexpr auto parameter_temp = R"(
    {{
        "dtype": "{}",
        "metric_type": "{}",
        "dim": {},
        "hnsw": {{
            "max_degree": 96,
            "ef_construction": 500
        }}
    }}
    )";
    auto build_parameters_str = fmt::format(parameter_temp, dtype, metric_type, dim);
    return build_parameters_str;
}

void
HNSWTestIndex::TestGeneral(const TestIndex::IndexPtr& index,
                           const TestDatasetPtr& dataset,
                           const std::string& search_param,
                           float recall,
                           bool expect_success) {
    REQUIRE(index->GetIndexType() == vsag::IndexType::HNSW);
    TestKnnSearch(index, dataset, search_param, recall, true);
    TestConcurrentKnnSearch(index, dataset, search_param, recall, true);
    TestRangeSearch(index, dataset, search_param, recall, 10, true);
    TestRangeSearch(index, dataset, search_param, recall / 2.0, 5, true);
    TestFilterSearch(index, dataset, search_param, recall, true, true);
    TestCheckIdExist(index, dataset);
    TestBatchCalcDistanceById(index, dataset, 1e-5, true, false);
    TestSearchAllocator(index, dataset, search_param, recall, true);
    TestUpdateVector(index, dataset, search_param, false);
    TestUpdateId(index, dataset, search_param, true);
}

}  // namespace fixtures

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex,
                             "HNSW Factory Test With Exceptions",
                             "[ft][factory][hnsw]") {
    auto name = "hnsw";
    SECTION("Empty parameters") {
        auto param = "{}";
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("No dim param") {
        auto param = R"(
        {{
            "dtype": "float32",
            "metric_type": "l2",
            "hnsw": {{
                "max_degree": 32,
                "ef_construction": 200
            }}
        }})";
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid param") {
        auto metric = GENERATE("", "l4", "inner_product", "cosin", "hamming");
        constexpr const char* param_tmp = R"(
        {{
            "dtype": "float32",
            "metric_type": "{}",
            "dim": 23,
            "hnsw": {{
                "max_degree": 32,
                "ef_construction": 300
            }}
        }})";
        auto param = fmt::format(param_tmp, metric);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid datatype param") {
        auto datatype = GENERATE("fp32", "uint8_t", "binary", "", "float");
        constexpr const char* param_tmp = R"(
        {{
            "dtype": "{}",
            "metric_type": "l2",
            "dim": 23,
            "hnsw": {{
                "max_degree": 32,
                "ef_construction": 300
            }}
        }})";
        auto param = fmt::format(param_tmp, datatype);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }
    // TODO(lht)dim check
    /*
    SECTION("Invalid dim param") {
        auto dim = GENERATE(-1, std::numeric_limits<uint64_t>::max(), 0, 8.6);
        constexpr const char* param_tmp = R"(
        {{
            "dtype": "float32",
            "metric_type": "l2",
            "dim": {},
            "hnsw": {{
                "max_degree": 64,
                "ef_construction": 500
            }}
        }})";
        auto param = fmt::format(param_tmp, dim);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }
    */

    SECTION("Miss hnsw param") {
        auto param = GENERATE(
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35,
                "hnsw": {{
                    "ef_construction": 300
                }}
            }})",
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35,
                "hnsw": {{
                    "max_degree": 32,
                }}
            }})");
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid hnsw param max_degree") {
        auto max_degree = GENERATE(-1, 0, 256, 3);
        // TODO(LHT): test for float param
        constexpr const char* param_temp =
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35,
                "hnsw": {{
                    "max_degree": {},
                    "ef_construction": 300
                }}
            }})";
        auto param = fmt::format(param_temp, max_degree);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid hnsw param ef_construction") {
        auto ef_construction = GENERATE(-1, 0, 100000, 31);
        // TODO(LHT): test for float param
        constexpr const char* param_temp =
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35,
                "hnsw": {{
                    "max_degree": 32,
                    "ef_construction": {}
                }}
            }})";
        auto param = fmt::format(param_temp, ef_construction);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex,
                             "HNSW Estimate Memory",
                             "[ft][memory][hnsw]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "cosine");

    const std::string name = "hnsw";
    auto search_param = fmt::format(search_param_tmp, 200, false);
    uint64_t estimate_count = 1000;
    for (auto dim : dims) {
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GenerateHNSWBuildParametersString(metric_type, dim);
        auto dataset = pool.GetDatasetAndCreate(dim,
                                                estimate_count,
                                                metric_type,
                                                false /*with_path*/,
                                                0.8 /*valid_ratio*/,
                                                0 /*extro_info_size*/);
        TestEstimateMemory(name, param, dataset);
        vsag::Options::Instance().set_block_size_limit(origin_size);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex,
                             "HNSW Build & ContinueAdd Test",
                             "[ft][build][hnsw]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "hnsw";
    auto search_param = fmt::format(search_param_tmp, 100);
    for (auto& dim : dims) {
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GenerateHNSWBuildParametersString(metric_type, dim);
        auto index = TestFactory(name, param, true);
        REQUIRE(index->GetIndexType() == vsag::IndexType::HNSW);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestContinueAdd(index, dataset, true);
        TestKnnSearch(index, dataset, search_param, 0.99, true);
        TestKnnSearchIter(index, dataset, search_param, 0.99, true);
        TestConcurrentKnnSearch(index, dataset, search_param, 0.99, true);
        TestRangeSearch(index, dataset, search_param, 0.99, 10, true);
        TestRangeSearch(index, dataset, search_param, 0.49, 5, true);
        TestFilterSearch(index, dataset, search_param, 0.99, true);
        TestSearchAllocator(index, dataset, search_param, 0.99, true);
    }
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex, "HNSW Test non-standard IDs", "[ft][hnsw]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2");

    const std::string name = "hnsw";
    auto search_param = fmt::format(search_param_tmp, 200);
    auto id_shift = 52;
    constexpr auto parameter_temp = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "hnsw": {{
            "max_degree": 64,
            "ef_construction": 200
        }}
    }}
    )";
    for (auto& dim : dims) {
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = fmt::format(parameter_temp, metric_type, dim);
        auto index = TestFactory(name, param, true);
        REQUIRE(index->GetIndexType() == vsag::IndexType::HNSW);
        auto dataset = pool.GetDatasetAndCreate(dim, 1200, metric_type, false, 0.8, 0, id_shift);
        TestContinueAdd(index, dataset, true);
        TestKnnSearch(index, dataset, search_param, 0.99, true);
        TestKnnSearchIter(index, dataset, search_param, 0.99, true);
        TestConcurrentKnnSearch(index, dataset, search_param, 0.99, true);
        TestRangeSearch(index, dataset, search_param, 0.99, 10, true);
        TestRangeSearch(index, dataset, search_param, 0.49, 5, true);
        TestFilterSearch(index, dataset, search_param, 0.99, true);
        TestSearchAllocator(index, dataset, search_param, 0.99, true);
    }
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex,
                             "HNSW Continue Destruct V.S. All Test",
                             "[ft][build][hnsw]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2");
    const std::string name = "hnsw";
    auto search_param = fmt::format(search_param_tmp, 100);

    vsag::Options::Instance().set_block_size_limit(size);
    auto dims_ = fixtures::get_common_used_dims(20);
    for (auto& dim : dims_) {
        auto param = GenerateHNSWBuildParametersString(metric_type, dim);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestBuildIndex(index, dataset, true);
        TestConcurrentDestruct(index, dataset, search_param);
    }

    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex,
                             "HNSW Search with Dirty Vector",
                             "[ft][search][hnsw]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    auto dataset = pool.GetNanDataset(metric_type);
    auto dim = dataset->dim_;
    const std::string name = "hnsw";
    auto search_param = fmt::format(search_param_tmp, 100);

    vsag::Options::Instance().set_block_size_limit(size);
    auto param = GenerateHNSWBuildParametersString(metric_type, dim);
    auto index = TestFactory(name, param, true);
    TestBuildIndex(index, dataset, true);
    TestSearchWithDirtyVector(index, dataset, search_param, true);
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex, "HNSW Build", "[ft][build][hnsw]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "hnsw";
    auto search_param = fmt::format(search_param_tmp, 100);
    for (auto& dim : dims) {
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GenerateHNSWBuildParametersString(metric_type, dim);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);

        TestBuildIndex(index, dataset, true);
        TestKnnSearch(index, dataset, search_param, 0.99, true);
        TestConcurrentKnnSearch(index, dataset, search_param, 0.99, true);
        TestRangeSearch(index, dataset, search_param, 0.99, 10, true);
        TestRangeSearch(index, dataset, search_param, 0.49, 5, true);
        TestFilterSearch(index, dataset, search_param, 0.99, true);
        if (index->CheckFeature(vsag::IndexFeature::SUPPORT_CHECK_ID_EXIST)) {
            TestCheckIdExist(index, dataset);
        }
    }
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex, "HNSW Merge", "[ft][merge][hnsw]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2");
    const std::string name = "hnsw";
    auto search_param = fmt::format(search_param_tmp, 100);
    for (auto& dim : dims) {
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GenerateHNSWBuildParametersString(metric_type, dim);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        auto index = TestMergeIndex(name, param, dataset, 5, true);
        TestKnnSearch(index, dataset, search_param, 0.99, true);
        TestRangeSearch(index, dataset, search_param, 0.49, 5, true);
        TestFilterSearch(index, dataset, search_param, 0.99, true);
        if (index->CheckFeature(vsag::IndexFeature::SUPPORT_CHECK_ID_EXIST)) {
            TestCheckIdExist(index, dataset);
        }
    }
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex, "HNSW Filter", "[ft][filter_search][hnsw]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2");
    const std::string name = "hnsw";
    auto search_param = fmt::format(search_param_tmp, 100);
    auto dim = 32;
    for (auto& valid_ratio : valid_ratios) {
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GenerateHNSWBuildParametersString(metric_type, dim);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type, false, valid_ratio);

        TestBuildIndex(index, dataset, true);
        TestFilterSearch(index, dataset, search_param, 0.99, true, true);
        if (index->CheckFeature(vsag::IndexFeature::SUPPORT_CHECK_ID_EXIST)) {
            TestCheckIdExist(index, dataset);
        }
    }
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex, "HNSW Add", "[ft][build][hnsw]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "hnsw";
    auto search_param = fmt::format(search_param_tmp, 100);
    for (auto& dim : dims) {
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GenerateHNSWBuildParametersString(metric_type, dim);
        auto index = TestFactory(name, param, true);

        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestAddIndex(index, dataset, true);
        TestKnnSearch(index, dataset, search_param, 0.99, true);
        TestConcurrentKnnSearch(index, dataset, search_param, 0.99, true);
        TestRangeSearch(index, dataset, search_param, 0.99, 10, true);
        TestRangeSearch(index, dataset, search_param, 0.49, 5, true);
        TestFilterSearch(index, dataset, search_param, 0.99, true);
        if (index->CheckFeature(vsag::IndexFeature::SUPPORT_CHECK_ID_EXIST)) {
            TestCheckIdExist(index, dataset);
        }

        vsag::Options::Instance().set_block_size_limit(origin_size);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex,
                             "HNSW Concurrent Add",
                             "[ft][build][hnsw][concurrent][build]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_types = fixtures::RandomSelect<std::string>({"l2", "ip", "cosine"}, 1);
    auto selected_dims = fixtures::RandomSelect<int>(dims, 1);
    const std::string name = "hnsw";
    auto search_param = fmt::format(search_param_tmp, 100);
    vsag::Options::Instance().set_block_size_limit(size);
    for (auto& metric_type : metric_types) {
        for (auto& dim : selected_dims) {
            INFO(fmt::format("metric_type: {}, dim: {}", metric_type, dim));
            auto param = GenerateHNSWBuildParametersString(metric_type, dim);
            auto index = TestFactory(name, param, true);

            auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
            TestConcurrentAdd(index, dataset, true);
            TestKnnSearch(index, dataset, search_param, 0.95, true);
            TestConcurrentKnnSearch(index, dataset, search_param, 0.95, true);
            TestRangeSearch(index, dataset, search_param, 0.95, 10, true);
            TestRangeSearch(index, dataset, search_param, 0.45, 5, true);
            TestFilterSearch(index, dataset, search_param, 0.95, true);
            if (index->CheckFeature(vsag::IndexFeature::SUPPORT_CHECK_ID_EXIST)) {
                TestCheckIdExist(index, dataset);
            }
        }
    }
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex, "HNSW Remove", "[ft][remove][hnsw]") {
    auto test_recovery = GENERATE(true, false);
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    std::vector<std::string> metric_types = {"l2", "ip", "cosine"};
    const std::string name = "hnsw";
    auto search_param = fmt::format(search_param_tmp, 100);
    for (auto& dim : dims) {
        auto metric_type = metric_types[random() % metric_types.size()];
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GenerateHNSWBuildParametersString(metric_type, dim);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);

        if (test_recovery) {
            TestIndex::TestRecoverRemoveIndex(index, dataset, search_param);
            HNSWTestIndex::TestGeneral(index, dataset, search_param, 0.85, false);
        } else {
            TestIndex::TestRemoveIndex(index, dataset, true);
            HNSWTestIndex::TestGeneral(index, dataset, search_param, 0.85);
        }

        vsag::Options::Instance().set_block_size_limit(origin_size);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex, "HNSW Update Id", "[ft][update][hnsw]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "hnsw";
    auto search_param = fmt::format(search_param_tmp, 100);
    for (auto& dim : dims) {
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GenerateHNSWBuildParametersString(metric_type, dim);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestBuildIndex(index, dataset, true);
        TestUpdateId(index, dataset, search_param, true);
        vsag::Options::Instance().set_block_size_limit(origin_size);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex,
                             "HNSW Batch Calc Dis Id",
                             "[ft][distance][hnsw]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "hnsw";
    auto search_param = fmt::format(search_param_tmp, 100);
    for (auto& dim : dims) {
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GenerateHNSWBuildParametersString(metric_type, dim);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestBuildIndex(index, dataset, true);
        TestBatchCalcDistanceById(index, dataset, 1e-5, true, false);
        vsag::Options::Instance().set_block_size_limit(origin_size);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex, "HNSW Get Min Max ID", "[ft][search][hnsw]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2");
    const std::string name = "hnsw";
    auto search_param = fmt::format(search_param_tmp, 100);
    auto dim = 128;
    vsag::Options::Instance().set_block_size_limit(size);
    auto param = GenerateHNSWBuildParametersString(metric_type, dim);
    auto index = TestFactory(name, param, true);
    auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
    TestBuildIndex(index, dataset, true);
    TestGetMinAndMaxId(index, dataset);
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex, "HNSW Get Raw Vector", "[ft][update][hnsw]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto dtype = GENERATE("float32", "int8");
    auto metric_type = GENERATE("l2");
    const std::string name = "hnsw";
    for (auto& dim : dims) {
        if (dtype == std::string("int8")) {
            metric_type = "ip";
        }
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GenerateHNSWBuildParametersString(metric_type, dim, dtype);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestBuildIndex(index, dataset, true);
        TestGetRawVectorByIds(index, dataset, true);
        vsag::Options::Instance().set_block_size_limit(origin_size);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex, "HNSW Update Vector", "[ft][update][hnsw]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2");
    const std::string name = "hnsw";
    auto search_param = fmt::format(search_param_tmp, 100);
    for (auto& dim : dims) {
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GenerateHNSWBuildParametersString(metric_type, dim);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestBuildIndex(index, dataset, true);
        TestUpdateVector(index, dataset, search_param, true);
        vsag::Options::Instance().set_block_size_limit(origin_size);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex,
                             "HNSW Serialize File",
                             "[ft][serialize][hnsw][serialization]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "hnsw";
    auto search_param = fmt::format(search_param_tmp, 100);

    for (auto& dim : dims) {
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GenerateHNSWBuildParametersString(metric_type, dim);
        auto index = TestFactory(name, param, true);

        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestBuildIndex(index, dataset, true);

        auto index2 = TestFactory(name, param, true);
        TestSerializeFile(index, index2, dataset, search_param, true);
    }
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex,
                             "HNSW Build & ContinueAdd Test With Random Allocator",
                             "[ft][build][hnsw]") {
    auto allocator = std::make_shared<fixtures::RandomAllocator>();
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "hnsw";
    for (auto& dim : dims) {
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GenerateHNSWBuildParametersString(metric_type, dim);
        auto index = vsag::Factory::CreateIndex(name, param, allocator.get());
        if (not index.has_value()) {
            continue;
        }
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestContinueAddIgnoreRequire(index.value(), dataset);
    }
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex,
                             "HNSW Duplicate Add",
                             "[ft][build][hnsw][concurrent]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_types = fixtures::RandomSelect<std::string>({"l2", "ip", "cosine"}, 1);
    auto selected_dims = fixtures::RandomSelect<int>(dims, 1);
    const std::string name = "hnsw";
    auto search_param = fmt::format(search_param_tmp, 100);
    vsag::Options::Instance().set_block_size_limit(size);
    for (auto& metric_type : metric_types) {
        for (auto& dim : selected_dims) {
            INFO(fmt::format("metric_type: {}, dim: {}", metric_type, dim));
            auto param = GenerateHNSWBuildParametersString(metric_type, dim);
            auto index = TestFactory(name, param, true);

            auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
            TestDuplicateAdd(index, dataset);
            TestKnnSearch(index, dataset, search_param, 0.99, true);
            TestConcurrentKnnSearch(index, dataset, search_param, 0.99, true);
            TestRangeSearch(index, dataset, search_param, 0.99, 10, true);
            TestRangeSearch(index, dataset, search_param, 0.49, 5, true);
            TestFilterSearch(index, dataset, search_param, 0.99, true);
        }
    }
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex,
                             "HNSW Set Immutable",
                             "[ft][immutable][hnsw]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = "l2";
    const std::string name = "hnsw";
    auto dim = 128;
    auto search_param = fmt::format(search_param_tmp, 100);
    vsag::Options::Instance().set_block_size_limit(size);
    auto param = GenerateHNSWBuildParametersString(metric_type, dim);
    auto index = TestFactory(name, param, true);
    auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
    auto result_immutable = index->SetImmutable();
    REQUIRE(result_immutable.has_value());
    // test SetImmutable Again
    auto result_immutable_again = index->SetImmutable();
    REQUIRE(result_immutable_again.has_value());
    TestDuplicateAdd(index, dataset);
    TestKnnSearch(index, dataset, search_param, 0.99, true);
    TestConcurrentKnnSearch(index, dataset, search_param, 0.99, true);
    TestRangeSearch(index, dataset, search_param, 0.99, 10, true);
    TestRangeSearch(index, dataset, search_param, 0.49, 5, true);
    TestFilterSearch(index, dataset, search_param, 0.99, true);
}

TEST_CASE("hnsw int8 recall", "[ft][search][hnsw]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);

    int64_t num_vectors = 1000;
    int64_t dim = 104;

    std::string index_parameters = R"({
        "dim": 104,
        "dtype": "int8",
        "hnsw": {
            "ef_construction": 50,
            "ef_search": 50,
            "max_degree": 12
        },
        "metric_type": "ip"
    })";

    auto index_result = vsag::Factory::CreateIndex("hnsw", index_parameters);
    REQUIRE(index_result.has_value());
    std::shared_ptr<vsag::Index> hnsw = index_result.value();

    vsag::IndexDetailInfo info;
    auto data_type = hnsw->GetDetailDataByName(vsag::INDEX_DETAIL_DATA_TYPE, info)
                         .value()
                         ->GetDataScalarString();
    REQUIRE(data_type == vsag::DATATYPE_INT8);

    std::vector<int64_t> ids(num_vectors);
    std::vector<int8_t> data(dim * num_vectors);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_int_distribution<int> distrib_real(-128, 127);
    for (int i = 0; i < num_vectors; i++) ids[i] = i;
    for (int i = 0; i < dim * num_vectors; i++) data[i] = static_cast<int8_t>(distrib_real(rng));

    // build index
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Int8Vectors(data.data())
        ->Owner(false);
    auto buildindex = hnsw->Build(base);
    REQUIRE(buildindex.has_value());

    for (int i = 0; i < num_vectors; i++) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dim)->Int8Vectors(data.data() + i * dim)->Owner(false);
        auto search_parameters = R"(
                {
                    "hnsw": {
                        "ef_search": 100
                    }
                }
                )";
        int64_t k = 10;
        auto result = hnsw->KnnSearch(query, k, search_parameters);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetIds()[0] == ids[i]);
    }
}

TEST_CASE("int8 + freshhnsw + feedback + update", "[ft][feedback][update][hnsw]") {
    auto logger = vsag::Options::Instance().logger();
    logger->SetLevel(vsag::Logger::Level::kDEBUG);

    // parameters
    int dim = 256;
    int num_base = 1000;
    int num_query = 1000;
    int64_t k = 3;
    auto metric_type = GENERATE("ip");
    constexpr auto build_parameter_json = R"(
    {{
        "dtype": "int8",
        "metric_type": "{}",
        "dim": {},
        "fresh_hnsw": {{
            "max_degree": 16,
            "ef_construction": 20,
            "use_conjugate_graph": true
        }}
    }}
    )";
    auto build_parameter = fmt::format(build_parameter_json, metric_type, dim);

    // create index
    auto createindex = vsag::Factory::CreateIndex("fresh_hnsw", build_parameter);
    REQUIRE(createindex.has_value());
    auto index = createindex.value();

    // generate dataset
    std::vector<int64_t> base_ids(num_base);
    std::vector<int64_t> update_ids(num_base);
    for (int64_t i = 0; i < num_base; ++i) {
        base_ids[i] = i;
        update_ids[i] = i + 2 * num_base;
    }
    auto base_vectors = fixtures::generate_int8_codes(num_base, dim);
    auto base = vsag::Dataset::Make();
    auto queries = vsag::Dataset::Make();
    base->NumElements(num_base)
        ->Dim(dim)
        ->Ids(base_ids.data())
        ->Int8Vectors(base_vectors.data())
        ->Owner(false);

    auto query_vectors = fixtures::generate_int8_codes(num_query, dim);
    queries->NumElements(num_query)->Dim(dim)->Int8Vectors(query_vectors.data())->Owner(false);

    // build index
    auto buildindex = index->Build(base);
    REQUIRE(buildindex.has_value());

    // train and search
    float recall[2];
    int correct;
    uint32_t error_fix = 0;
    bool use_conjugate_graph_search = false;
    for (int round = 0; round < 2; round++) {
        correct = 0;

        if (round == 0) {
            logger->Debug("====train stage====");
        } else {
            logger->Debug("====test stage====");
        }

        logger->Debug(fmt::format(R"(Memory Usage: {:.3f} KB)", index->GetMemoryUsage() / 1024.0));

        use_conjugate_graph_search = (round != 0);
        constexpr auto search_parameters_json = R"(
        {{
            "fresh_hnsw": {{
                "ef_search": 10,
                "use_conjugate_graph_search": {}
            }}
        }}
        )";
        auto search_parameters = fmt::format(search_parameters_json, use_conjugate_graph_search);

        for (int i = 0; i < num_query; i++) {
            auto query = vsag::Dataset::Make();
            query->Dim(dim)
                ->Int8Vectors(queries->GetInt8Vectors() + i * dim)
                ->NumElements(1)
                ->Owner(false);

            auto result = index->KnnSearch(query, k, search_parameters);
            REQUIRE(result.has_value());
            auto bf_result = fixtures::brute_force(query, base, 1, metric_type, "int8");
            int64_t global_optimum = bf_result->GetIds()[0];
            int64_t local_optimum = result.value()->GetIds()[0];

            if (local_optimum != global_optimum and round == 0) {
                auto feedback_result = index->Feedback(query, k, search_parameters, global_optimum);
                REQUIRE(feedback_result.has_value());
                error_fix += feedback_result.value();
                auto feedback_default = index->Feedback(query, k, search_parameters);
                REQUIRE(feedback_default.has_value());
                REQUIRE(feedback_default.value() == 0);
            }

            if (local_optimum == global_optimum or local_optimum == update_ids[global_optimum]) {
                correct++;
            }
        }

        if (round == 0) {
            for (int i = 0; i < num_base; i++) {
                auto update_res = index->UpdateId(base_ids[i], update_ids[i]);
                REQUIRE(update_res.has_value());
                REQUIRE(update_res.value() == true);
            }
        }
        recall[round] = correct / (1.0 * num_query);
        logger->Debug(fmt::format(R"(Recall: {:.4f})", recall[round]));
    }

    logger->Debug("====summary====");
    logger->Debug(fmt::format(R"(Error fix: {})", error_fix));

    const bool initial_recall_is_perfect =
        fixtures::recall_t(recall[0]) == fixtures::recall_t(1.0f);
    if (not initial_recall_is_perfect) {
        REQUIRE(error_fix > 0);
        REQUIRE(recall[0] < recall[1]);
    }
    REQUIRE(recall[1] >= recall[0]);
    REQUIRE(fixtures::recall_t(recall[1]) == fixtures::recall_t(1.0f));
}

TEST_CASE("hnsw + feedback with global optimum id + remove", "[ft][feedback][remove][hnsw]") {
    auto logger = vsag::Options::Instance().logger();
    logger->SetLevel(vsag::Logger::Level::kDEBUG);

    // parameters
    auto is_remove = GENERATE(true, false);
    int dim = 128;
    int num_base = 1000;
    int num_query = 1000;
    int64_t k = 10;
    auto metric_type = GENERATE("l2");
    constexpr auto build_parameter_json = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "hnsw": {{
            "max_degree": 16,
            "ef_construction": 20,
            "use_conjugate_graph": true
        }}
    }}
    )";
    auto build_parameter = fmt::format(build_parameter_json, metric_type, dim);

    // create index
    auto createindex = vsag::Factory::CreateIndex("hnsw", build_parameter);
    REQUIRE(createindex.has_value());
    auto index = createindex.value();

    vsag::IndexDetailInfo info;
    auto data_type = index->GetDetailDataByName(vsag::INDEX_DETAIL_DATA_TYPE, info)
                         .value()
                         ->GetDataScalarString();
    REQUIRE(data_type == vsag::DATATYPE_FLOAT32);

    // generate dataset
    auto [base_ids, base_vectors] = fixtures::generate_ids_and_vectors(num_base, dim);
    auto base = vsag::Dataset::Make();
    auto queries = vsag::Dataset::Make();
    base->NumElements(num_base)
        ->Dim(dim)
        ->Ids(base_ids.data())
        ->Float32Vectors(base_vectors.data())
        ->Owner(false);

    auto [query_ids, query_vectors] = fixtures::generate_ids_and_vectors(num_query, dim);
    queries->NumElements(num_query)
        ->Dim(dim)
        ->Ids(query_ids.data())
        ->Float32Vectors(query_vectors.data())
        ->Owner(false);

    // build index
    auto buildindex = index->Build(base);
    REQUIRE(buildindex.has_value());

    // train and search
    float recall[2];
    int correct;
    uint32_t error_fix = 0;
    bool use_conjugate_graph_search = false;
    std::vector<int64_t> removed_id;
    for (int round = 0; round < 2; round++) {
        correct = 0;

        if (round == 0) {
            logger->Debug("====train stage====");
        } else {
            logger->Debug("====test stage====");
        }

        logger->Debug(fmt::format(R"(Memory Usage: {:.3f} KB)", index->GetMemoryUsage() / 1024.0));

        use_conjugate_graph_search = (round != 0);
        constexpr auto search_parameters_json = R"(
        {{
            "hnsw": {{
                "ef_search": 10,
                "use_conjugate_graph_search": {}
            }}
        }}
        )";
        auto search_parameters = fmt::format(search_parameters_json, use_conjugate_graph_search);

        for (int i = 0; i < num_query; i++) {
            auto query = vsag::Dataset::Make();
            query->Dim(dim)
                ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
                ->NumElements(1)
                ->Owner(false);

            auto result = index->KnnSearch(query, k, search_parameters);
            REQUIRE(result.has_value());
            auto bf_result = fixtures::brute_force(query, base, 1, metric_type);
            int64_t global_optimum = bf_result->GetIds()[0];
            int64_t local_optimum = result.value()->GetIds()[0];

            if (local_optimum != global_optimum and round == 0) {
                auto feedback_result = index->Feedback(query, k, search_parameters, global_optimum);
                REQUIRE(feedback_result.has_value());
                error_fix += feedback_result.value();
                auto feedback_default = index->Feedback(query, k, search_parameters);
                REQUIRE(feedback_default.has_value());
                REQUIRE(feedback_default.value() == 0);
                if (is_remove) {
                    index->Remove(global_optimum);
                    removed_id.push_back(global_optimum);
                }
            }

            if (local_optimum == global_optimum) {
                correct++;
            }
        }
        recall[round] = correct / (1.0 * num_query);
        logger->Debug(fmt::format(R"(Recall: {:.4f})", recall[round]));
    }

    logger->Debug("====summary====");
    logger->Debug(fmt::format(R"(Error fix: {})", error_fix));

    const bool initial_recall_is_perfect =
        fixtures::recall_t(recall[0]) == fixtures::recall_t(1.0f);
    if (not initial_recall_is_perfect) {
        REQUIRE(error_fix > 0);
    }
    if (not is_remove) {
        if (not initial_recall_is_perfect) {
            REQUIRE(recall[0] < recall[1]);
        }
        REQUIRE(recall[1] >= recall[0]);
        REQUIRE(fixtures::recall_t(recall[1]) == fixtures::recall_t(1.0f));
    }
}

TEST_CASE("hnsw with pretrained by conjugate graph", "[ft][feedback][hnsw]") {
    auto logger = vsag::Options::Instance().logger();
    logger->SetLevel(vsag::Logger::Level::kDEBUG);

    // parameters
    int dim = 128;
    int base_elements = 10000;
    int query_elements = 1000;
    int ef_search = 10;
    int64_t k = 10;
    auto metric_type = GENERATE("l2");
    std::set<int64_t> failed_base_set;
    constexpr auto search_parameters_json = R"(
        {{
            "hnsw": {{
                "ef_search": {},
                "use_conjugate_graph_search": true
            }}
        }}
        )";
    auto search_parameters = fmt::format(search_parameters_json, ef_search);
    constexpr auto build_parameter_json = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "hnsw": {{
            "max_degree": 16,
            "ef_construction": 200,
            "use_conjugate_graph": true
        }}
    }}
    )";
    auto build_parameter = fmt::format(build_parameter_json, metric_type, dim);

    // generate data (use base[0: query_num] as query)
    auto base = vsag::Dataset::Make();
    auto query = vsag::Dataset::Make();
    std::vector<int64_t> base_ids(base_elements);
    std::vector<float> base_data(dim * base_elements);
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<float> distribution_real(-1, 1);
    for (int i = 0; i < base_elements; i++) {
        base_ids[i] = i;

        for (int d = 0; d < dim; d++) {
            base_data[d + i * dim] = distribution_real(rng);
        }
    }
    base->Dim(dim)
        ->NumElements(base_elements)
        ->Ids(base_ids.data())
        ->Float32Vectors(base_data.data())
        ->Owner(false);
    query->Dim(dim)->NumElements(1)->Owner(false);

    // Create index
    std::shared_ptr<vsag::Index> hnsw;
    auto index = vsag::Factory::CreateIndex("hnsw", build_parameter);
    REQUIRE(index.has_value());
    hnsw = index.value();

    // Build index
    {
        auto build_result = hnsw->Build(base);
        REQUIRE(build_result.has_value());
    }

    // Search without empty conjugate graph
    {
        int correct = 0;
        logger->Debug("====Search Stage====");
        logger->Debug(fmt::format("Memory Usage: {:.3f} KB", hnsw->GetMemoryUsage() / 1024.0));

        for (int i = 0; i < query_elements; i++) {
            query->Float32Vectors(base_data.data() + i * dim);

            auto result = hnsw->KnnSearch(query, k, search_parameters);
            REQUIRE(result.has_value());
            int64_t global_optimum = i;  // global optimum is itself
            int64_t local_optimum = result.value()->GetIds()[0];

            if (local_optimum != global_optimum) {
                failed_base_set.emplace(global_optimum);
            }

            if (local_optimum == global_optimum) {
                correct++;
            }
        }
        logger->Debug(fmt::format("Recall: {:.4f}", correct / (1.0 * query_elements)));
    }

    // Pretrain
    {
        logger->Debug("====Pretrain Stage====");
        logger->Debug(fmt::format("Before Pretrain, Memory Usage: {:.3f} KB",
                                  hnsw->GetMemoryUsage() / 1024.0));
        std::vector<int64_t> failed_base_vec(failed_base_set.begin(), failed_base_set.end());
        REQUIRE(hnsw->Pretrain(failed_base_vec, k, search_parameters).has_value());
        logger->Debug(fmt::format("After Pretrain, Memory Usage: {:.3f} KB",
                                  hnsw->GetMemoryUsage() / 1024.0));
    }

    // Search with pretrained conjugate graph
    {
        int correct = 0;
        logger->Debug("====Enhanced Search Stage====");
        logger->Debug(fmt::format("Memory Usage: {:.3f} KB", hnsw->GetMemoryUsage() / 1024.0));

        for (int i = 0; i < query_elements; i++) {
            query->Float32Vectors(base_data.data() + i * dim);

            auto result = hnsw->KnnSearch(query, k, search_parameters);
            REQUIRE(result.has_value());
            int64_t global_optimum = i;  // global optimum is itself
            int64_t local_optimum = result.value()->GetIds()[0];

            if (local_optimum == global_optimum) {
                correct++;
            }
        }
        logger->Debug(fmt::format("Enhanced Recall: {:.4f}", correct / (1.0 * query_elements)));

        fixtures::recall_t recall = 1.0f * correct / query_elements;
        REQUIRE(recall == 1.0);
    }
}
