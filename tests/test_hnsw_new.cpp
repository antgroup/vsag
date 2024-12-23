
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
#include "simd/simd.h"
#include "test_index.h"
#include "vsag/vsag.h"

namespace fixtures {
class HNSWTestIndex : public fixtures::TestIndex {
public:
    static std::string
    GenerateHNSWBuildParametersString(const std::string& metric_type, int64_t dim);

    static TestDatasetPool pool;

    static std::vector<int> dims;

    constexpr static uint64_t base_count = 1000;

    constexpr static const char* search_param_tmp = R"(
        {{
            "hnsw": {{
                "ef_search": {}
            }}
        }})";
};

TestDatasetPool HNSWTestIndex::pool{};
std::vector<int> HNSWTestIndex::dims = fixtures::get_common_used_dims(2, RandomValue(0, 999));

std::string
HNSWTestIndex::GenerateHNSWBuildParametersString(const std::string& metric_type, int64_t dim) {
    constexpr auto parameter_temp = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "hnsw": {{
            "max_degree": 64,
            "ef_construction": 500
        }}
    }}
    )";
    auto build_parameters_str = fmt::format(parameter_temp, metric_type, dim);
    return build_parameters_str;
}
}  // namespace fixtures

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex,
                             "HNSW Factory Test With Exceptions",
                             "[ft][hnsw]") {
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
                "max_degree": 64,
                "ef_construction": 500
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
                "max_degree": 64,
                "ef_construction": 500
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
                "max_degree": 64,
                "ef_construction": 500
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
                    "ef_construction": 500
                }}
            }})",
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35,
                "hnsw": {{
                    "max_degree": 64,
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
                    "ef_construction": 500
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
                             "HNSW Build & ContinueAdd Test",
                             "[ft][hnsw]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    std::string base_quantization_str = GENERATE("sq8", "fp32");
    const std::string name = "hnsw";
    auto search_param = fmt::format(search_param_tmp, 100);
    for (auto& dim : dims) {
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GenerateHNSWBuildParametersString(metric_type, dim);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestContinueAdd(index, dataset, true);
        TestKnnSearch(index, dataset, search_param, 0.99, true);
        TestConcurrentKnnSearch(index, dataset, search_param, 0.99, true);
        TestRangeSearch(index, dataset, search_param, 0.99, 10, true);
        TestRangeSearch(index, dataset, search_param, 0.49, 5, true);
        TestFilterSearch(index, dataset, search_param, 0.99, true);
    }
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex, "HNSW Build", "[ft][hnsw]") {
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
    }
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex, "HNSW Add", "[ft][hnsw]") {
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

        vsag::Options::Instance().set_block_size_limit(origin_size);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HNSWTestIndex, "HNSW Serialize File", "[ft][hnsw]") {
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
                             "[ft][hnsw]") {
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