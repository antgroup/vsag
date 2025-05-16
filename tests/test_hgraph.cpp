
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
#include "fixtures/test_logger.h"
#include "inner_string_params.h"
#include "test_index.h"
#include "vsag/options.h"

namespace fixtures {
class HgraphTestIndex : public fixtures::TestIndex {
public:
    static std::string
    GenerateHGraphBuildParametersString(const std::string& metric_type,
                                        int64_t dim,
                                        const std::string& quantization_str = "sq8",
                                        int thread_count = 5,
                                        int extra_info_size = 0,
                                        const std::string& data_type = "float32",
                                        std::string graph_type = "nsw");

    static bool
    IsRaBitQ(const std::string& quantization_str);

    static void
    TestGeneral(const IndexPtr& index,
                const TestDatasetPtr& dataset,
                const std::string& search_param,
                float recall);

    static TestDatasetPool pool;

    static std::vector<int> dims;

    static fixtures::TempDir dir;

    constexpr static uint64_t base_count = 1200;

    constexpr static const char* search_param_tmp = R"(
        {{
            "hgraph": {{
                "ef_search": {},
                "use_extra_info_filter": {}
            }}
        }})";
    const std::vector<std::pair<std::string, float>> test_cases = {
        {"fp32", 0.99},
        {"bf16", 0.98},
        {"fp16", 0.98},
        {"sq8", 0.95},
        {"sq8_uniform", 0.95},
        {"rabitq,fp32", 0.3},
        {"pq,fp32", 0.95},
        {"sq8_uniform,fp32", 0.98},
        {"sq8_uniform,fp16", 0.98},
        {"sq8_uniform,bf16", 0.98},
    };
};

TestDatasetPool HgraphTestIndex::pool{};
std::vector<int> HgraphTestIndex::dims = fixtures::get_common_used_dims(1, RandomValue(0, 999));
fixtures::TempDir HgraphTestIndex::dir{"hgraph_test"};

std::string
HgraphTestIndex::GenerateHGraphBuildParametersString(const std::string& metric_type,
                                                     int64_t dim,
                                                     const std::string& quantization_str,
                                                     int thread_count,
                                                     int extra_info_size,
                                                     const std::string& data_type,
                                                     std::string graph_type) {
    std::string build_parameters_str;

    constexpr auto parameter_temp_reorder = R"(
    {{
        "dtype": "{}",
        "metric_type": "{}",
        "dim": {},
        "extra_info_size": {},
        "index_param": {{
            "use_reorder": {},
            "base_quantization_type": "{}",
            "max_degree": 96,
            "ef_construction": 500,
            "build_thread_count": {},
            "base_pq_dim": {},
            "precise_quantization_type": "{}",
            "precise_io_type": "{}",
            "precise_file_path": "{}",
            "graph_type": "{}",
            "graph_iter_turn": 10,
            "neighbor_sample_rate": 0.3,
            "alpha": 1.2
        }}
    }}
    )";

    constexpr auto parameter_temp_origin = R"(
    {{
        "dtype": "{}",
        "metric_type": "{}",
        "dim": {},
        "extra_info_size": {},
        "index_param": {{
            "base_quantization_type": "{}",
            "max_degree": 96,
            "base_pq_dim": {},
            "ef_construction": 500,
            "build_thread_count": {},
            "graph_type": "{}",
            "graph_iter_turn": 10,
            "neighbor_sample_rate": 0.3,
            "alpha": 1.2
        }}
    }}
    )";

    int pq_dim = dim;
    if (pq_dim % 2 == 0) {
        pq_dim /= 2;
    }

    auto strs = fixtures::SplitString(quantization_str, ',');
    std::string high_quantizer_str, precise_io_type = "block_memory_io";
    auto& base_quantizer_str = strs[0];
    if (strs.size() > 1) {
        high_quantizer_str = strs[1];
        if (strs.size() > 2) {
            precise_io_type = strs[2];
        }
        build_parameters_str = fmt::format(parameter_temp_reorder,
                                           data_type,
                                           metric_type,
                                           dim,
                                           extra_info_size,
                                           true, /* reorder */
                                           base_quantizer_str,
                                           thread_count,
                                           pq_dim,
                                           high_quantizer_str,
                                           precise_io_type,
                                           dir.GenerateRandomFile(),
                                           graph_type);
    } else {
        build_parameters_str = fmt::format(parameter_temp_origin,
                                           data_type,
                                           metric_type,
                                           dim,
                                           extra_info_size,
                                           base_quantizer_str,
                                           pq_dim,
                                           thread_count,
                                           graph_type);
    }
    return build_parameters_str;
}

bool
HgraphTestIndex::IsRaBitQ(const std::string& quantization_str) {
    return (quantization_str.find(vsag::QUANTIZATION_TYPE_VALUE_RABITQ) != std::string::npos);
}

void
HgraphTestIndex::TestGeneral(const TestIndex::IndexPtr& index,
                             const TestDatasetPtr& dataset,
                             const std::string& search_param,
                             float recall) {
    TestGetMinAndMaxId(index, dataset);
    TestKnnSearch(index, dataset, search_param, recall, true);
    TestKnnSearchIter(index, dataset, search_param, recall, true);
    TestConcurrentKnnSearch(index, dataset, search_param, recall, true);
    TestRangeSearch(index, dataset, search_param, recall, 10, true);
    TestRangeSearch(index, dataset, search_param, recall / 2.0, 5, true);
    TestFilterSearch(index, dataset, search_param, recall, true, true);
    TestCheckIdExist(index, dataset);
    TestCalcDistanceById(index, dataset);
    TestBatchCalcDistanceById(index, dataset);
}
}  // namespace fixtures

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex,
                             "HGraph Factory Test With Exceptions",
                             "[ft][hgraph]") {
    auto name = "hgraph";
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

    SECTION("Invalid param") {
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

    SECTION("Miss hgraph param") {
        auto param = GENERATE(
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35,
                "index_param": {{
                }}
            }})",
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35
            }})");
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid hgraph param base_quantization_type") {
        auto base_quantization_types = GENERATE("fsa");
        constexpr const char* param_temp =
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35,
                "index_param": {{
                    "base_quantization_type": "{}"
                }}
            }})";
        auto param = fmt::format(param_temp, base_quantization_types);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid hgraph param key") {
        auto param_keys = GENERATE("base_quantization_types", "base_quantization");
        constexpr const char* param_temp =
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35,
                "index_param": {{
                    "{}": "sq8"
                }}
            }})";
        auto param = fmt::format(param_temp, param_keys);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex,
                             "HGraph Build & ContinueAdd Test",
                             "[ft][hgraph]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "hgraph";
    auto search_param = fmt::format(search_param_tmp, 200, false);
    for (auto dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            if (IsRaBitQ(base_quantization_str)) {
                if (std::string(metric_type) != "l2") {
                    continue;
                }
                if (dim <= fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim += fixtures::RABITQ_MIN_RACALL_DIM;
                }
            }
            vsag::Options::Instance().set_block_size_limit(size);
            auto param =
                GenerateHGraphBuildParametersString(metric_type, dim, base_quantization_str);
            auto index = TestFactory(name, param, true);
            auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
            TestContinueAdd(index, dataset, true);
            TestGeneral(index, dataset, search_param, recall);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex, "HGraph Train & Add Test", "[ft][hgraph]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "hgraph";
    auto search_param = fmt::format(search_param_tmp, 200, false);
    for (auto dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            if (IsRaBitQ(base_quantization_str)) {
                if (std::string(metric_type) != "l2") {
                    continue;
                }
                if (dim <= fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim += fixtures::RABITQ_MIN_RACALL_DIM;
                }
            }
            vsag::Options::Instance().set_block_size_limit(size);
            auto param =
                GenerateHGraphBuildParametersString(metric_type, dim, base_quantization_str);
            auto index = TestFactory(name, param, true);
            auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
            TestTrainAndAdd(index, dataset, true);
            TestGeneral(index, dataset, search_param, recall);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex,
                             "HGraph Search Empty Index",
                             "[ft][hgraph]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2");

    const std::string name = "hgraph";
    auto search_param = fmt::format(search_param_tmp, 200, false);
    auto ex_search_param = fmt::format(search_param_tmp, 200, true);
    auto dim = dims[0];
    auto& [base_quantization_str, recall] = test_cases[0];
    vsag::Options::Instance().set_block_size_limit(size);
    auto param = GenerateHGraphBuildParametersString(metric_type, dim, base_quantization_str);
    auto index = TestFactory(name, param, true);
    auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
    TestGetMinAndMaxId(index, dataset, false);
    TestKnnSearch(index, dataset, search_param, recall, false);
    TestKnnSearchIter(index, dataset, search_param, recall, false);
    TestConcurrentKnnSearch(index, dataset, search_param, recall, false);
    TestRangeSearch(index, dataset, search_param, recall, 10, false);
    TestRangeSearch(index, dataset, search_param, recall / 2.0, 5, false);
    TestFilterSearch(index, dataset, search_param, recall, false, true);
    TestCheckIdExist(index, dataset, false);
    TestCalcDistanceById(index, dataset, 2e-6, false);
    TestBatchCalcDistanceById(index, dataset, 2e-6, false);
    TestKnnSearchExFilter(index, dataset, ex_search_param, recall, false);
    TestKnnSearchIter(index, dataset, ex_search_param, recall, false, true);
    // with ex info empty index
    auto extra_info_size = 256;
    auto ex_param = GenerateHGraphBuildParametersString(
        metric_type, dim, base_quantization_str, 5, extra_info_size);
    auto ex_index = TestFactory(name, param, true);
    auto ex_dataset =
        pool.GetDatasetAndCreate(dim, base_count, metric_type, false, 0.8, extra_info_size);
    TestKnnSearchExFilter(ex_index, ex_dataset, ex_search_param, recall, false);
    TestKnnSearchIter(ex_index, ex_dataset, ex_search_param, recall, false, true);
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex, "HGraph Build", "[ft][hgraph]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");

    const std::string name = "hgraph";
    auto search_param = fmt::format(search_param_tmp, 200, false);
    for (auto dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            if (IsRaBitQ(base_quantization_str)) {
                if (std::string(metric_type) != "l2") {
                    continue;
                }
                if (dim <= fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim += fixtures::RABITQ_MIN_RACALL_DIM;
                }
            }
            vsag::Options::Instance().set_block_size_limit(size);
            auto param =
                GenerateHGraphBuildParametersString(metric_type, dim, base_quantization_str);
            auto index = TestFactory(name, param, true);
            auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
            TestBuildIndex(index, dataset, true);
            TestGeneral(index, dataset, search_param, recall);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex, "HGraph ODescent Build", "[ft][hgraph]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");

    const std::string name = "hgraph";
    auto search_param = fmt::format(search_param_tmp, 200, false);
    for (auto dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            if (IsRaBitQ(base_quantization_str)) {
                if (std::string(metric_type) != "l2") {
                    continue;
                }
                if (dim <= fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim += fixtures::RABITQ_MIN_RACALL_DIM;
                }
            }
            auto param = GenerateHGraphBuildParametersString(
                metric_type, dim, base_quantization_str, 0, 0, "float32", "odescent");
            vsag::Options::Instance().set_block_size_limit(size);
            auto index = TestFactory(name, param, true);
            auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
            TestBuildIndex(index, dataset, true);
            TestGeneral(index, dataset, search_param, recall);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex, "HGraph Add", "[ft][hgraph]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "hgraph";
    auto search_param = fmt::format(search_param_tmp, 200, false);
    for (auto dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            if (IsRaBitQ(base_quantization_str)) {
                if (std::string(metric_type) != "l2") {
                    continue;
                }
                if (dim <= fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim += fixtures::RABITQ_MIN_RACALL_DIM;
                }
            }
            vsag::Options::Instance().set_block_size_limit(size);
            auto param =
                GenerateHGraphBuildParametersString(metric_type, dim, base_quantization_str);
            auto index = TestFactory(name, param, true);
            auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
            TestAddIndex(index, dataset, true);
            if (index->CheckFeature(vsag::SUPPORT_ADD_FROM_EMPTY)) {
                TestGeneral(index, dataset, search_param, recall);
            }
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex,
                             "HGraph Search with Dirty Vector",
                             "[ft][hgraph]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "cosine");
    auto dataset = pool.GetNanDataset(metric_type);
    auto dim = dataset->dim_;
    const std::string name = "hgraph";
    auto search_param = fmt::format(search_param_tmp, 100, false);
    for (auto& [base_quantization_str, recall] : test_cases) {
        if (IsRaBitQ(base_quantization_str)) {
            if (std::string(metric_type) != "l2") {
                continue;
            }
        }
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GenerateHGraphBuildParametersString(metric_type, dim, base_quantization_str);
        auto index = TestFactory(name, param, true);
        TestBuildIndex(index, dataset, true);
        TestSearchWithDirtyVector(index, dataset, search_param, true);
    }
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex,
                             "HGraph Search with Sparse Vector",
                             "[ft][hgraph][concurrent]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("ip");
    auto dim = 128;
    auto dataset = pool.GetSparseDatasetAndCreate(base_count, dim, 0.8);
    const std::string name = "hgraph";
    auto search_param = fmt::format(search_param_tmp, 100, false);
    vsag::Options::Instance().set_block_size_limit(size);
    auto param = GenerateHGraphBuildParametersString(metric_type, dim, "sparse", 5, 0, "sparse");
    auto index = TestFactory(name, param, true);
    TestConcurrentAdd(index, dataset, true);
    TestKnnSearch(index, dataset, search_param, true);
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex,
                             "HGraph Concurrent Add",
                             "[ft][hgraph][concurrent]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "cosine");

    const std::string name = "hgraph";
    auto search_param = fmt::format(search_param_tmp, 200, false);
    for (auto dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            if (IsRaBitQ(base_quantization_str)) {
                if (std::string(metric_type) != "l2") {
                    continue;
                }
                if (dim <= fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim += fixtures::RABITQ_MIN_RACALL_DIM;
                }
            }
            vsag::Options::Instance().set_block_size_limit(size);
            auto param =
                GenerateHGraphBuildParametersString(metric_type, dim, base_quantization_str);
            auto index = TestFactory(name, param, true);
            auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
            TestConcurrentAdd(index, dataset, true);
            if (index->CheckFeature(vsag::SUPPORT_ADD_CONCURRENT)) {
                TestGeneral(index, dataset, search_param, recall);
            }
        }
        vsag::Options::Instance().set_block_size_limit(origin_size);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex, "HGraph Serialize File", "[ft][hgraph]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "cosine");
    const std::string name = "hgraph";
    auto search_param = fmt::format(search_param_tmp, 200, false);
    uint64_t extra_info_size = 64;

    for (auto dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            if (IsRaBitQ(base_quantization_str)) {
                if (std::string(metric_type) != "l2") {
                    continue;
                }
                if (dim <= fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim += fixtures::RABITQ_MIN_RACALL_DIM;
                }
            }
            vsag::Options::Instance().set_block_size_limit(size);
            auto param = GenerateHGraphBuildParametersString(
                metric_type, dim, base_quantization_str, 5 /*thread_count*/, extra_info_size);
            auto index = TestFactory(name, param, true);
            auto dataset = pool.GetDatasetAndCreate(dim,
                                                    base_count,
                                                    metric_type,
                                                    false /*with_path*/,
                                                    0.8 /*valid_ratio*/,
                                                    extra_info_size);
            TestBuildIndex(index, dataset, true);
            auto index2 = TestFactory(name, param, true);
            TestSerializeFile(index, index2, dataset, search_param, true);
            index2 = TestFactory(name, param, true);
            TestSerializeBinarySet(index, index2, dataset, search_param, true);
            index2 = TestFactory(name, param, true);
            TestSerializeReaderSet(index, index2, dataset, search_param, name, true);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex, "HGraph Clone", "[ft][hgraph]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "cosine");
    const std::string name = "hgraph";
    auto search_param = fmt::format(search_param_tmp, 200, false);
    uint64_t extra_info_size = 64;

    for (auto dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            if (IsRaBitQ(base_quantization_str)) {
                if (std::string(metric_type) != "l2") {
                    continue;
                }
                if (dim <= fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim += fixtures::RABITQ_MIN_RACALL_DIM;
                }
            }
            vsag::Options::Instance().set_block_size_limit(size);
            auto param = GenerateHGraphBuildParametersString(
                metric_type, dim, base_quantization_str, 5 /*thread_count*/, extra_info_size);
            auto index = TestFactory(name, param, true);
            auto dataset = pool.GetDatasetAndCreate(dim,
                                                    base_count,
                                                    metric_type,
                                                    false /*with_path*/,
                                                    0.8 /*valid_ratio*/,
                                                    extra_info_size);
            TestBuildIndex(index, dataset, true);
            TestClone(index, dataset, search_param);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex, "HGraph Export Model", "[ft][hgraph]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "cosine");
    const std::string name = "hgraph";
    auto search_param = fmt::format(search_param_tmp, 200, false);
    uint64_t extra_info_size = 64;
    for (auto dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            if (IsRaBitQ(base_quantization_str)) {
                if (std::string(metric_type) != "l2") {
                    continue;
                }
                if (dim <= fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim += fixtures::RABITQ_MIN_RACALL_DIM;
                }
            }
            vsag::Options::Instance().set_block_size_limit(size);
            auto param = GenerateHGraphBuildParametersString(
                metric_type, dim, base_quantization_str, 5 /*thread_count*/, extra_info_size);
            auto index = TestFactory(name, param, true);
            auto dataset = pool.GetDatasetAndCreate(dim,
                                                    base_count,
                                                    metric_type,
                                                    false /*with_path*/,
                                                    0.8 /*valid_ratio*/,
                                                    extra_info_size);
            TestBuildIndex(index, dataset, true);
            TestExportModel(index, dataset, search_param);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex,
                             "HGraph Build & ContinueAdd Test With Random Allocator",
                             "[ft][hgraph]") {
    auto allocator = std::make_shared<fixtures::RandomAllocator>();
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "cosine");
    const std::string name = "hgraph";
    for (auto dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            if (IsRaBitQ(base_quantization_str)) {
                if (std::string(metric_type) != "l2") {
                    continue;
                }
                if (dim <= fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim += fixtures::RABITQ_MIN_RACALL_DIM;
                }
            }
            vsag::Options::Instance().set_block_size_limit(size);
            auto param =
                GenerateHGraphBuildParametersString(metric_type, dim, base_quantization_str, 1);
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

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex, "HGraph Duplicate Build", "[ft][hgraph]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip");

    const std::string name = "hgraph";
    auto search_param = fmt::format(search_param_tmp, 200, false);
    for (auto dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            if (IsRaBitQ(base_quantization_str)) {
                if (std::string(metric_type) != "l2") {
                    continue;
                }
                if (dim <= fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim += fixtures::RABITQ_MIN_RACALL_DIM;
                }
            }
            vsag::Options::Instance().set_block_size_limit(size);
            auto param =
                GenerateHGraphBuildParametersString(metric_type, dim, base_quantization_str);
            auto index = TestFactory(name, param, true);

            auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
            TestDuplicateAdd(index, dataset);
            TestGeneral(index, dataset, search_param, recall);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex, "HGraph Estimate Memory", "[ft][hgraph]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "cosine");

    const std::string name = "hgraph";
    auto search_param = fmt::format(search_param_tmp, 200, false);
    uint64_t estimate_count = 1000;
    uint64_t extra_info_size = 64;
    for (auto dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            if (IsRaBitQ(base_quantization_str)) {
                if (std::string(metric_type) != "l2") {
                    continue;
                }
                if (dim <= fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim += fixtures::RABITQ_MIN_RACALL_DIM;
                }
            }
            vsag::Options::Instance().set_block_size_limit(size);
            auto param = GenerateHGraphBuildParametersString(
                metric_type, dim, base_quantization_str, 5 /*thread_count*/, extra_info_size);
            auto dataset = pool.GetDatasetAndCreate(dim,
                                                    estimate_count,
                                                    metric_type,
                                                    false /*with_path*/,
                                                    0.8 /*valid_ratio*/,
                                                    extra_info_size);
            TestEstimateMemory(name, param, dataset);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex, "HGraph ELP Optimizer", "[ft][hgraph]") {
    fixtures::logger::LoggerReplacer _;
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);

    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "cosine");

    const std::string name = "hgraph";
    auto search_param = fmt::format(search_param_tmp, 200, false);
    constexpr auto parameter_temp = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "use_reorder": true,
            "use_elp_optimizer": {},
            "base_quantization_type": "sq4_uniform",
            "max_degree": 64,
            "ef_construction": 200,
            "precise_quantization_type": "fp32",
            "ignore_reorder": true
        }}
    }}
    )";

    auto dim = 128;
    vsag::Options::Instance().set_block_size_limit(size);
    auto base = pool.GetDatasetAndCreate(dim, 100, metric_type);
    std::string param_weak = fmt::format(parameter_temp, metric_type, dim, false);
    std::string param_strong = fmt::format(parameter_temp, metric_type, dim, true);
    auto index_weak = TestFactory(name, param_weak, true);
    TestBuildIndex(index_weak, base);
    auto index_strong = TestFactory(name, param_strong, true);
    TestBuildIndex(index_strong, base);
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex, "HGraph Ignore Reorder", "[ft][hgraph]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "cosine");

    const std::string name = "hgraph";
    auto search_param = fmt::format(search_param_tmp, 200, false);
    constexpr auto parameter_temp_reorder = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "use_reorder": true,
            "base_quantization_type": "sq8",
            "max_degree": 96,
            "ef_construction": 400,
            "precise_quantization_type": "fp32",
            "ignore_reorder": true
        }}
    }}
    )";
    for (auto dim : dims) {
        vsag::Options::Instance().set_block_size_limit(size);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        std::string param = fmt::format(parameter_temp_reorder, metric_type, dim);
        auto index = TestFactory(name, param, true);
        TestBuildIndex(index, dataset);
        TestGeneral(index, dataset, search_param, 0.95);
        vsag::Options::Instance().set_block_size_limit(origin_size);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex, "HGraph With Extra Info", "[ft][hgraph]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip");
    uint64_t extra_info_size = 256;

    const std::string name = "hgraph";
    auto search_param = fmt::format(search_param_tmp, 200, false);
    auto search_ex_filter_param = fmt::format(search_param_tmp, 500, true);
    for (auto& dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            if (IsRaBitQ(base_quantization_str)) {
                if (std::string(metric_type) != "l2") {
                    continue;
                }
                if (dim <= fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim += fixtures::RABITQ_MIN_RACALL_DIM;
                }
            }
            vsag::Options::Instance().set_block_size_limit(size);
            auto param = GenerateHGraphBuildParametersString(
                metric_type, dim, base_quantization_str, 5 /*thread_count*/, extra_info_size);
            auto index = TestFactory(name, param, true);
            auto dataset = pool.GetDatasetAndCreate(dim,
                                                    base_count,
                                                    metric_type,
                                                    false /*with_path*/,
                                                    0.8 /*valid_ratio*/,
                                                    extra_info_size);
            TestBuildIndex(index, dataset, true);
            TestKnnSearch(index, dataset, search_param, recall, true);
            TestKnnSearchIter(index, dataset, search_param, recall, true);
            TestRangeSearch(index, dataset, search_param, recall, 10, true);
            TestGetExtraInfoById(index, dataset, extra_info_size);
            TestKnnSearchExFilter(index, dataset, search_ex_filter_param, recall, true);
            TestKnnSearchIter(index, dataset, search_ex_filter_param, recall, true, true);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HgraphTestIndex,
                             "HGraph Disk IO Type Index",
                             "[ft][hgraph]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "hgraph";
    auto search_param = fmt::format(search_param_tmp, 200, false);
    float recall = 0.98;
    const std::vector<std::pair<std::string, std::string>> io_cases = {
        {"sq8_uniform,bf16", "sq8_uniform,bf16,buffer_io"},
        {"sq8_uniform,fp16", "sq8_uniform,fp16,async_io"},
    };
    for (auto dim : dims) {
        for (auto& [memory_io_str, disk_io_str] : io_cases) {
            if (IsRaBitQ(memory_io_str)) {
                if (std::string(metric_type) != "l2") {
                    continue;
                }
                if (dim <= fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim += fixtures::RABITQ_MIN_RACALL_DIM;
                }
            }
            vsag::Options::Instance().set_block_size_limit(size);
            auto param = GenerateHGraphBuildParametersString(metric_type, dim, memory_io_str);
            auto index = TestFactory(name, param, true);
            auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
            TestBuildIndex(index, dataset, true);

            param = GenerateHGraphBuildParametersString(metric_type, dim, disk_io_str);
            auto disk_index = TestFactory(name, param, true);
            TestSerializeFile(index, disk_index, dataset, search_param, true);
            TestGeneral(disk_index, dataset, search_param, recall);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}
