
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

#include <spdlog/spdlog.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <limits>

#include "fixtures/test_dataset_pool.h"
#include "test_index.h"
#include "vsag/options.h"

namespace fixtures {
class BruteForceTestIndex : public fixtures::TestIndex {
public:
    static std::string
    GenerateBruteForceBuildParametersString(const std::string& metric_type,
                                            int64_t dim,
                                            const std::string& quantization_str = "sq8",
                                            int thread_count = 5);
    static TestDatasetPool pool;

    static std::vector<int> dims;

    constexpr static uint64_t base_count = 3000;

    const std::vector<std::pair<std::string, float>> test_cases = {
        {"sq8", 0.94}, {"fp32", 0.999999}, {"sq8_uniform", 0.94}};
};

TestDatasetPool BruteForceTestIndex::pool{};
std::vector<int> BruteForceTestIndex::dims = fixtures::get_common_used_dims(2, RandomValue(0, 999));

std::string
BruteForceTestIndex::GenerateBruteForceBuildParametersString(const std::string& metric_type,
                                                             int64_t dim,
                                                             const std::string& quantization_str,
                                                             int thread_count) {
    std::string build_parameters_str;

    constexpr auto parameter_temp = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "quantization_type": "{}"
        }}
    }}
    )";

    build_parameters_str = fmt::format(parameter_temp, metric_type, dim, quantization_str);

    return build_parameters_str;
}
}  // namespace fixtures

TEST_CASE_PERSISTENT_FIXTURE(fixtures::BruteForceTestIndex,
                             "BruteForce Factory Test With Exceptions",
                             "[ft][bruteforce]") {
    auto name = "brute_force";
    SECTION("Empty parameters") {
        auto param = "{}";
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("No dim param") {
        auto param = R"(
        {{
            "dtype": "float32",
            "metric_type": "l2",
            "index_param": {{
                "base_quantization_type": "sq8"
            }}
        }})";
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid metric param") {
        auto metric = GENERATE("", "l4", "inner_product", "cosin", "hamming");
        constexpr const char* param_tmp = R"(
        {{
            "dtype": "float32",
            "metric_type": "{}",
            "dim": 23,
            "index_param": {{
                "base_quantization_type": "sq8"
            }}
        }})";
        auto param = fmt::format(param_tmp, metric);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid datatype param") {
        auto datatype = GENERATE("fp32", "uint8_t", "binary", "", "float", "int8");
        constexpr const char* param_tmp = R"(
        {{
            "dtype": "{}",
            "metric_type": "l2",
            "dim": 23,
            "index_param": {{
                "base_quantization_type": "sq8"
            }}
        }})";
        auto param = fmt::format(param_tmp, datatype);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid dim param") {
        int dim = GENERATE(-12, -1, 0);
        constexpr const char* param_tmp = R"(
        {{
            "dtype": "float32",
            "metric_type": "l2",
            "dim": {},
            "index_param": {{
                "base_quantization_type": "sq8"
            }}
        }})";
        auto param = fmt::format(param_tmp, dim);
        REQUIRE_THROWS(TestFactory(name, param, false));
        auto float_param = R"(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 3.51,
            "index_param": {
                "base_quantization_type": "sq8"
            }
        })";
        REQUIRE_THROWS(TestFactory(name, float_param, false));
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::BruteForceTestIndex,
                             "BruteForce Build & ContinueAdd Test",
                             "[ft][bruteforce]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");

    const std::string name = "brute_force";
    auto search_param = "";
    for (auto& dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            vsag::Options::Instance().set_block_size_limit(size);
            auto param =
                GenerateBruteForceBuildParametersString(metric_type, dim, base_quantization_str);
            auto index = TestFactory(name, param, true);
            if (index->CheckFeature(vsag::SUPPORT_ADD_AFTER_BUILD)) {
                auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
                TestContinueAdd(index, dataset, true);
                if (index->CheckFeature(vsag::SUPPORT_KNN_SEARCH)) {
                    TestKnnSearch(index, dataset, search_param, recall, true);
                    if (index->CheckFeature(vsag::SUPPORT_SEARCH_CONCURRENT)) {
                        TestConcurrentKnnSearch(index, dataset, search_param, recall, true);
                    }
                }
                if (index->CheckFeature(vsag::SUPPORT_RANGE_SEARCH)) {
                    TestRangeSearch(index, dataset, search_param, recall, 10, true);
                    TestRangeSearch(index, dataset, search_param, recall / 2.0, 5, true);
                }
                if (index->CheckFeature(vsag::SUPPORT_KNN_SEARCH_WITH_ID_FILTER)) {
                    TestFilterSearch(index, dataset, search_param, recall, true);
                }
                if (index->CheckFeature(vsag::IndexFeature::SUPPORT_CHECK_ID_EXIST)) {
                    TestCheckIdExist(index, dataset);
                }
            }
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::BruteForceTestIndex,
                             "BruteForce Build",
                             "[ft][bruteforce]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("ip");

    const std::string name = "brute_force";
    auto search_param = "";
    for (auto& dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            vsag::Options::Instance().set_block_size_limit(size);
            auto param =
                GenerateBruteForceBuildParametersString(metric_type, dim, base_quantization_str);
            auto index = TestFactory(name, param, true);
            auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
            if (index->CheckFeature(vsag::SUPPORT_BUILD)) {
                TestBuildIndex(index, dataset, true);
                if (index->CheckFeature(vsag::SUPPORT_KNN_SEARCH)) {
                    TestKnnSearch(index, dataset, search_param, recall, true);
                    if (index->CheckFeature(vsag::SUPPORT_SEARCH_CONCURRENT)) {
                        TestConcurrentKnnSearch(index, dataset, search_param, recall, true);
                    }
                }
                if (index->CheckFeature(vsag::SUPPORT_RANGE_SEARCH)) {
                    TestRangeSearch(index, dataset, search_param, recall, 10, true);
                    TestRangeSearch(index, dataset, search_param, recall / 2.0, 5, true);
                }
                if (index->CheckFeature(vsag::SUPPORT_KNN_SEARCH_WITH_ID_FILTER)) {
                    TestFilterSearch(index, dataset, search_param, recall, true);
                }
                if (index->CheckFeature(vsag::IndexFeature::SUPPORT_CHECK_ID_EXIST)) {
                    TestCheckIdExist(index, dataset);
                }
            }
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::BruteForceTestIndex, "BruteForce Add", "[ft][bruteforce]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");

    const std::string name = "brute_force";
    auto search_param = "";
    for (auto& dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            vsag::Options::Instance().set_block_size_limit(size);
            auto param =
                GenerateBruteForceBuildParametersString(metric_type, dim, base_quantization_str);
            auto index = TestFactory(name, param, true);
            if (index->CheckFeature(vsag::SUPPORT_ADD_FROM_EMPTY)) {
                auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
                TestAddIndex(index, dataset, true);
                if (index->CheckFeature(vsag::SUPPORT_KNN_SEARCH)) {
                    TestKnnSearch(index, dataset, search_param, recall, true);
                    if (index->CheckFeature(vsag::SUPPORT_SEARCH_CONCURRENT)) {
                        TestConcurrentKnnSearch(index, dataset, search_param, recall, true);
                    }
                }
                if (index->CheckFeature(vsag::SUPPORT_RANGE_SEARCH)) {
                    TestRangeSearch(index, dataset, search_param, recall, 10, true);
                    TestRangeSearch(index, dataset, search_param, recall / 2.0, 5, true);
                }
                if (index->CheckFeature(vsag::SUPPORT_KNN_SEARCH_WITH_ID_FILTER)) {
                    TestFilterSearch(index, dataset, search_param, recall, true);
                }
                if (index->CheckFeature(vsag::IndexFeature::SUPPORT_CHECK_ID_EXIST)) {
                    TestCheckIdExist(index, dataset);
                }
            }
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::BruteForceTestIndex,
                             "BruteForce Concurrent Add",
                             "[ft][bruteforce]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");

    const std::string name = "brute_force";
    auto search_param = "";
    for (auto& dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            vsag::Options::Instance().set_block_size_limit(size);
            auto param =
                GenerateBruteForceBuildParametersString(metric_type, dim, base_quantization_str);
            auto index = TestFactory(name, param, true);
            if (index->CheckFeature(vsag::SUPPORT_ADD_CONCURRENT)) {
                auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
                TestConcurrentAdd(index, dataset, true);
                if (index->CheckFeature(vsag::SUPPORT_KNN_SEARCH)) {
                    TestKnnSearch(index, dataset, search_param, recall, true);
                    if (index->CheckFeature(vsag::SUPPORT_SEARCH_CONCURRENT)) {
                        TestConcurrentKnnSearch(index, dataset, search_param, recall, true);
                    }
                }
                if (index->CheckFeature(vsag::SUPPORT_RANGE_SEARCH)) {
                    TestRangeSearch(index, dataset, search_param, recall, 10, true);
                    TestRangeSearch(index, dataset, search_param, recall / 2.0, 5, true);
                }
                if (index->CheckFeature(vsag::SUPPORT_KNN_SEARCH_WITH_ID_FILTER)) {
                    TestFilterSearch(index, dataset, search_param, recall, true);
                }
                if (index->CheckFeature(vsag::IndexFeature::SUPPORT_CHECK_ID_EXIST)) {
                    TestCheckIdExist(index, dataset);
                }
            }
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::BruteForceTestIndex,
                             "BruteForce Serialize File",
                             "[ft][bruteforce]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "brute_force";
    auto search_param = "";

    for (auto& dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            vsag::Options::Instance().set_block_size_limit(size);
            auto param =
                GenerateBruteForceBuildParametersString(metric_type, dim, base_quantization_str);
            auto index = TestFactory(name, param, true);

            if (index->CheckFeature(vsag::SUPPORT_BUILD)) {
                auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
                TestBuildIndex(index, dataset, true);
                if (index->CheckFeature(vsag::SUPPORT_SERIALIZE_FILE) and
                    index->CheckFeature(vsag::SUPPORT_DESERIALIZE_FILE)) {
                    auto index2 = TestFactory(name, param, true);
                    TestSerializeFile(index, index2, dataset, search_param, true);
                }
                if (index->CheckFeature(vsag::SUPPORT_SERIALIZE_BINARY_SET) and
                    index->CheckFeature(vsag::SUPPORT_DESERIALIZE_BINARY_SET)) {
                    auto index2 = TestFactory(name, param, true);
                    TestSerializeBinarySet(index, index2, dataset, search_param, true);
                }
                if (index->CheckFeature(vsag::SUPPORT_SERIALIZE_FILE) and
                    index->CheckFeature(vsag::SUPPORT_DESERIALIZE_READER_SET)) {
                    auto index2 = TestFactory(name, param, true);
                    TestSerializeReaderSet(index, index2, dataset, search_param, name, true);
                }
            }
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::BruteForceTestIndex,
                             "BruteForce Build & ContinueAdd Test With Random Allocator",
                             "[ft][hnsw]") {
    auto allocator = std::make_shared<fixtures::RandomAllocator>();
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "brute_force";
    for (auto& dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            vsag::Options::Instance().set_block_size_limit(size);
            auto param =
                GenerateBruteForceBuildParametersString(metric_type, dim, base_quantization_str, 1);
            auto index = vsag::Factory::CreateIndex(name, param, allocator.get());
            if (not index.has_value()) {
                continue;
            }
            auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
            TestContinueAddIgnoreRequire(index.value(), dataset);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::BruteForceTestIndex,
                             "BruteForce Duplicate Build",
                             "[ft][bruteforce]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");

    const std::string name = "brute_force";
    auto search_param = "";
    for (auto& dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            vsag::Options::Instance().set_block_size_limit(size);
            auto param =
                GenerateBruteForceBuildParametersString(metric_type, dim, base_quantization_str);
            auto index = TestFactory(name, param, true);
            if (index->CheckFeature(vsag::SUPPORT_BUILD)) {
                auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
                TestDuplicateAdd(index, dataset);
                if (index->CheckFeature(vsag::SUPPORT_KNN_SEARCH)) {
                    TestKnnSearch(index, dataset, search_param, recall, true);
                    if (index->CheckFeature(vsag::SUPPORT_SEARCH_CONCURRENT)) {
                        TestConcurrentKnnSearch(index, dataset, search_param, recall, true);
                    }
                }
                if (index->CheckFeature(vsag::SUPPORT_RANGE_SEARCH)) {
                    TestRangeSearch(index, dataset, search_param, recall, 10, true);
                    TestRangeSearch(index, dataset, search_param, recall / 2.0, 5, true);
                }
                if (index->CheckFeature(vsag::SUPPORT_KNN_SEARCH_WITH_ID_FILTER)) {
                    TestFilterSearch(index, dataset, search_param, recall, true);
                }
                if (index->CheckFeature(vsag::IndexFeature::SUPPORT_CHECK_ID_EXIST)) {
                    TestCheckIdExist(index, dataset);
                }
            }
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}
