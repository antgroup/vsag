
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
#include "typing.h"
#include "vsag/options.h"

namespace fixtures {

class HGraphTestResource {
public:
    std::vector<int> dims;
    std::vector<std::pair<std::string, float>> test_cases;
    std::vector<std::string> metric_types;
    uint64_t base_count;
};

using HGraphResourcePtr = std::shared_ptr<HGraphTestResource>;
class HGraphTestIndex : public fixtures::TestIndex {
public:
    struct HGraphBuildParam {
        std::string metric_type;
        int64_t dim;
        std::string quantization_str = "sq8";
        int thread_count = 5;
        int extra_info_size = 0;
        std::string data_type = "float32";
        std::string graph_type = "nsw";
        std::string graph_storage = "flat";
        bool support_remove = false;
        bool use_attr_filter = false;
        bool store_raw_vector = false;
        bool support_duplicate = false;
        HGraphBuildParam(const std::string& metric_type,
                         int64_t dim,
                         const std::string& quantization_str)
            : metric_type(metric_type), dim(dim), quantization_str(quantization_str) {
        }
    };

    static std::string
    GenerateHGraphBuildParametersString(const HGraphBuildParam& param);

    static HGraphResourcePtr
    GetResource(bool sample = true);

    static bool
    IsRaBitQ(const std::string& quantization_str);

    static void
    TestGeneral(const IndexPtr& index,
                const TestDatasetPtr& dataset,
                const std::string& search_param,
                float recall);

    static void
    TestMemoryUsageDetail(const IndexPtr& index);

    static TestDatasetPool pool;

    static fixtures::TempDir dir;

    static uint64_t base_count;

    static const std::string name;

    static const std::vector<std::pair<std::string, float>> all_test_cases;
};
using HGraphTestIndexPtr = std::shared_ptr<HGraphTestIndex>;

TestDatasetPool HGraphTestIndex::pool{};
fixtures::TempDir HGraphTestIndex::dir{"hgraph_test"};
uint64_t HGraphTestIndex::base_count = 1200;
const std::string HGraphTestIndex::name = "hgraph";
const std::vector<std::pair<std::string, float>> HGraphTestIndex::all_test_cases = {
    {"fp32", 0.99},
    {"bf16", 0.98},
    {"fp16", 0.98},
    {"sq8", 0.95},
    {"sq8_uniform", 0.95},
    {"rabitq,fp32", 0.3},
    {"pq,fp32", 0.95},
    {"sq4_uniform,fp32", 0.95},
    {"sq8_uniform,fp32", 0.98},
    {"sq8_uniform,fp16", 0.98},
    {"sq8_uniform,bf16", 0.98},
};

constexpr static const char* search_param_tmp = R"(
        {{
            "hgraph": {{
                "ef_search": {},
                "use_extra_info_filter": {}
            }}
        }})";

HGraphResourcePtr
HGraphTestIndex::GetResource(bool sample) {
    auto resource = std::make_shared<HGraphTestResource>();
    if (sample) {
        resource->dims = fixtures::get_common_used_dims(1, RandomValue(0, 999));
        resource->test_cases = fixtures::RandomSelect(HGraphTestIndex::all_test_cases, 3);
        resource->metric_types = fixtures::RandomSelect<std::string>({"ip", "l2", "cosine"}, 1);
        resource->base_count = HGraphTestIndex::base_count;
    } else {
        resource->dims = fixtures::get_common_used_dims();
        resource->test_cases = HGraphTestIndex::all_test_cases;
        resource->metric_types = {"ip", "l2", "cosine"};
        resource->base_count = HGraphTestIndex::base_count * 10;
    }
    return resource;
}

std::string
HGraphTestIndex::GenerateHGraphBuildParametersString(const HGraphBuildParam& param) {
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
            "graph_storage_type": "{}",
            "graph_iter_turn": 10,
            "neighbor_sample_rate": 0.3,
            "alpha": 1.2,
            "support_remove": {},
            "use_attribute_filter": {},
            "store_raw_vector": {},
            "support_duplicate": {}
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
            "graph_storage_type": "{}",
            "graph_iter_turn": 10,
            "neighbor_sample_rate": 0.3,
            "alpha": 1.2,
            "support_remove": {},
            "use_attribute_filter": {},
            "store_raw_vector": {},
            "support_duplicate": {}
        }}
    }}
    )";

    int pq_dim = param.dim;
    if (pq_dim % 2 == 0) {
        pq_dim /= 2;
    }

    auto strs = fixtures::SplitString(param.quantization_str, ',');
    std::string high_quantizer_str, precise_io_type = "block_memory_io";
    auto& base_quantizer_str = strs[0];
    if (strs.size() > 1) {
        high_quantizer_str = strs[1];
        if (strs.size() > 2) {
            precise_io_type = strs[2];
        }
        build_parameters_str = fmt::format(parameter_temp_reorder,
                                           param.data_type,
                                           param.metric_type,
                                           param.dim,
                                           param.extra_info_size,
                                           true, /* reorder */
                                           base_quantizer_str,
                                           param.thread_count,
                                           pq_dim,
                                           high_quantizer_str,
                                           precise_io_type,
                                           dir.GenerateRandomFile(),
                                           param.graph_type,
                                           param.graph_storage,
                                           param.support_remove,
                                           param.use_attr_filter,
                                           param.store_raw_vector,
                                           param.support_duplicate);
    } else {
        build_parameters_str = fmt::format(parameter_temp_origin,
                                           param.data_type,
                                           param.metric_type,
                                           param.dim,
                                           param.extra_info_size,
                                           base_quantizer_str,
                                           pq_dim,
                                           param.thread_count,
                                           param.graph_type,
                                           param.graph_storage,
                                           param.support_remove,
                                           param.use_attr_filter,
                                           param.store_raw_vector,
                                           param.support_duplicate);
    }
    return build_parameters_str;
}

bool
HGraphTestIndex::IsRaBitQ(const std::string& quantization_str) {
    return (quantization_str.find(vsag::QUANTIZATION_TYPE_VALUE_RABITQ) != std::string::npos);
}

void
HGraphTestIndex::TestGeneral(const TestIndex::IndexPtr& index,
                             const TestDatasetPtr& dataset,
                             const std::string& search_param,
                             float recall) {
    REQUIRE(index->GetIndexType() == vsag::IndexType::HGRAPH);
    TestGetMinAndMaxId(index, dataset);
    TestKnnSearch(index, dataset, search_param, recall, true);
    TestKnnSearchIter(index, dataset, search_param, recall, true);
    TestConcurrentKnnSearch(index, dataset, search_param, recall, true);
    TestRangeSearch(index, dataset, search_param, recall, 10, true);
    TestRangeSearch(index, dataset, search_param, recall / 2.0, 5, true);
    TestFilterSearch(index, dataset, search_param, recall, true, true);
    TestCheckIdExist(index, dataset);
    TestCalcDistanceById(index, dataset);
    TestGetRawVectorByIds(index, dataset);
    TestBatchCalcDistanceById(index, dataset);
    TestSearchAllocator(index, dataset, search_param, recall, true);
    TestMemoryUsageDetail(index);
}

void
HGraphTestIndex::TestMemoryUsageDetail(const IndexPtr& index) {
    auto memory_detail = vsag::JsonType::parse(index->GetMemoryUsageDetail());
    REQUIRE(memory_detail.contains("basic_flatten_codes"));
    REQUIRE(memory_detail.contains("bottom_graph"));
    REQUIRE(memory_detail.contains("route_graph"));
}
}  // namespace fixtures

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HGraphTestIndex,
                             "HGraph Factory Test With Exceptions",
                             "[ft][hgraph]") {
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

    SECTION(
        "Invalid hgraph param "
        "base_quantization_type") {
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

    SECTION(
        "Invalid hgraph param "
        "graph_storage_type") {
        auto graph_storage_type = "fsa";
        constexpr const char* param_temp =
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35,
                "index_param": {{
                    "graph_storage_type": "{}"
                }}
            }})";
        auto param = fmt::format(param_temp, graph_storage_type);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HGraphTestIndex,
                             "HGraph Factory Test With Correct Parameters",
                             "[ft][hgraph]") {
    // bug issue #883
    SECTION("Empty index_param") {
        auto param = R"(
        {
            "dtype": "float32",
            "dim": 128,
            "metric_type": "l2",
            "index_param": {
            }
        })";
        REQUIRE(TestFactory(name, param, true));
    }
    SECTION("pq index_param") {
        auto param = R"(
        {
            "dtype": "float32",
            "dim": 128,
            "metric_type": "l2",
            "index_param": {
                "base_quantization_type": "pq"
            }
        })";
        REQUIRE(TestFactory(name, param, true));
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HGraphTestIndex, "HGraph GetStatus", "[ft][hgraph]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.support_duplicate = true;
                build_param.support_remove = true;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestBuildIndex(index, dataset, true);
                INFO(index->GetStats());
            }
        }
    }
}

static void
TestHGraphBuildAndContinueAdd(const fixtures::HGraphTestIndexPtr& test_index,
                              const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                // TODO
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestContinueAdd(index, dataset, true);
                HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph Build & ContinueAdd Test", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphBuildAndContinueAdd(test_index, resource);
}

TEST_CASE("[Daily] HGraph Build & ContinueAdd Test", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphBuildAndContinueAdd(test_index, resource);
}

void
TestHGraphTrainAndAddTest(const fixtures::HGraphTestIndexPtr& test_index,
                          const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestTrainAndAdd(index, dataset, true);
                HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph Train & Add Test", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphTrainAndAddTest(test_index, resource);
}

TEST_CASE("[Daily] HGraph Train & Add Test", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphTrainAndAddTest(test_index, resource);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HGraphTestIndex,
                             "HGraph Search Empty Index",
                             "[ft][hgraph]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    auto ex_search_param = fmt::format(fixtures::search_param_tmp, 200, true);
    auto dim = fixtures::get_common_used_dims(1, fixtures::RandomValue(0, 999))[0];
    auto& [base_quantization_str, recall] = all_test_cases[0];
    vsag::Options::Instance().set_block_size_limit(size);
    HGraphTestIndex::HGraphBuildParam build_param(metric_type, dim, base_quantization_str);
    auto param = GenerateHGraphBuildParametersString(build_param);
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
    build_param.extra_info_size = 256;
    auto ex_param = GenerateHGraphBuildParametersString(build_param);
    auto ex_index = TestFactory(name, param, true);
    auto ex_dataset = pool.GetDatasetAndCreate(
        dim, base_count, metric_type, false, 0.8, build_param.extra_info_size);
    TestKnnSearchExFilter(ex_index, ex_dataset, ex_search_param, recall, false);
    TestKnnSearchIter(ex_index, ex_dataset, ex_search_param, recall, false, true);
    vsag::Options::Instance().set_block_size_limit(origin_size);
}
static void
TestHGraphBuild(const fixtures::HGraphTestIndexPtr& test_index,
                const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }

                vsag::Options::Instance().set_block_size_limit(size);

                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);

                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);

                TestIndex::TestBuildIndex(index, dataset, true);
                HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);

                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph Build Test", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphBuild(test_index, resource);
}

TEST_CASE("[Daily] HGraph Build Test", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphBuild(test_index, resource);
}
static void
TestHGraphWithAttr(const fixtures::HGraphTestIndexPtr& test_index,
                   const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);

    auto size = GENERATE(1024 * 1024 * 2);
    std::vector<std::pair<std::string, float>> tmp_test_cases = {
        {"fp32", 0.75},
    };
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : tmp_test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }

                // Set block size limit for current test iteration
                vsag::Options::Instance().set_block_size_limit(size);

                // Generate index parameters with attribute support enabled
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.use_attr_filter = true;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);

                // Create index and dataset
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);

                if (not index->CheckFeature(vsag::SUPPORT_BUILD)) {
                    continue;
                }
                auto build_result = index->Build(dataset->base_);
                REQUIRE(build_result.has_value());
                TestIndex::TestWithAttr(index, dataset, search_param, false);
                auto index2 = TestIndex::TestFactory(HGraphTestIndex::name, param, true);

                REQUIRE_NOTHROW(test_serializion_file(*index, *index2, "serialize_hgraph"));
                TestIndex::TestWithAttr(index2, dataset, search_param, true);

                // Restore original block size limit
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph With Attr", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphWithAttr(test_index, resource);
}

TEST_CASE("[Daily] HGraph With Attr", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphWithAttr(test_index, resource);
}

static void
TestHGraphGetRawVector(const fixtures::HGraphTestIndexPtr& test_index,
                       const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    const std::vector<std::pair<std::string, float>> test_cases = {{"fp32", 0.99},
                                                                   {"sq4_uniform,fp32", 0.95}};
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    for (auto metric_type : resource->metric_types) {
        metric_type = "cosine";
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }

                // Set block size limit for current test iteration
                vsag::Options::Instance().set_block_size_limit(size);

                // Generate index parameters with attribute support enabled
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.store_raw_vector = true;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);

                // Create index and dataset
                auto index = TestIndex::TestFactory(test_index->name, param, true);

                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);

                TestIndex::TestBuildIndex(index, dataset, true);

                // Execute attribute-aware build test
                HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);

                // Restore original block size limit
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph Support Get Raw Vector", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphGetRawVector(test_index, resource);
}

TEST_CASE("[Daily] HGraph Support Get Raw Vector", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphGetRawVector(test_index, resource);
}

static void
TestHGraphODescentBuild(const fixtures::HGraphTestIndexPtr& test_index,
                        const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));

                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }

                // Set block size limit for current test iteration
                vsag::Options::Instance().set_block_size_limit(size);

                // Generate index parameters with attribute support enabled
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.graph_type = "odescent";
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                // Create index and dataset
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);

                // Execute build test
                TestIndex::TestBuildIndex(index, dataset, true);
                HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);

                // Restore original block size limit
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph ODescent Build", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphODescentBuild(test_index, resource);
}

TEST_CASE("[Daily] HGraph ODescent Build", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphODescentBuild(test_index, resource);
}

static void
TestHGraphRemove(const fixtures::HGraphTestIndexPtr& test_index,
                 const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.support_remove = true;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestRemoveIndex(index, dataset, true);
                HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph Remove", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphRemove(test_index, resource);
}

TEST_CASE("[Daily] HGraph Remove", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphRemove(test_index, resource);
}

static void
TestHGraphCompressedBuild(const fixtures::HGraphTestIndexPtr& test_index,
                          const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.graph_storage = "compressed";
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestBuildIndex(index, dataset, true);
                HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph Compressed Graph Build", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphCompressedBuild(test_index, resource);
}

TEST_CASE("[Daily] HGraph Compressed Graph Build", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphCompressedBuild(test_index, resource);
}

static void
TestHGraphMerge(const fixtures::HGraphTestIndexPtr& test_index,
                const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto model = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                auto ret = model->Train(dataset->base_);
                REQUIRE(ret.has_value() == true);
                auto merge_index = TestIndex::TestMergeIndexWithSameModel(model, dataset, 5, true);
                HGraphTestIndex::TestGeneral(merge_index, dataset, search_param, recall);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph Merge", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphMerge(test_index, resource);
}

TEST_CASE("[Daily] HGraph Merge", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphMerge(test_index, resource);
}

static void
TestHGraphAdd(const fixtures::HGraphTestIndexPtr& test_index,
              const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestAddIndex(index, dataset, true);
                if (index->CheckFeature(vsag::SUPPORT_ADD_FROM_EMPTY)) {
                    HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);
                }
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph Add", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphAdd(test_index, resource);
}

TEST_CASE("[Daily] HGraph Add", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphAdd(test_index, resource);
}

static void
TestHGraphDuplicate(const fixtures::HGraphTestIndexPtr& test_index,
                    const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto duplicate_pos = GENERATE("prefix", "suffix", "middle");
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    std::unordered_map<std::string, float> ratios{
        {"prefix", 0.9}, {"suffix", 0.9}, {"middle", 1.0}};
    const std::vector<std::pair<std::string, float>> all_test_cases =
        fixtures::RandomSelect<std::pair<std::string, float>>(
            {
                {"fp32", 0.99},
                {"rabitq,fp32", 0.3},
                {"pq,fp32", 0.95},
                {"sq8_uniform,fp32", 0.98},
            },
            2);
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : all_test_cases) {
                INFO(
                    fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}, "
                                "duplicate_pos: {}",
                                metric_type,
                                dim,
                                base_quantization_str,
                                recall,
                                duplicate_pos));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.support_duplicate = true;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDuplicateDataset(
                    dim, resource->base_count, metric_type);
                TestIndex::TestBuildDuplicateIndex(index, dataset, duplicate_pos, true);
                TestIndex::TestKnnSearch(index, dataset, search_param, recall, true);
                // TODO(inabao): Fix knn search iter test
                // TestIndex::TestKnnSearchIter(index, dataset, search_param, recall, true);
                TestIndex::TestConcurrentKnnSearch(index, dataset, search_param, recall, true);
                TestIndex::TestRangeSearch(index, dataset, search_param, recall, 10, true);
                TestIndex::TestRangeSearch(index, dataset, search_param, recall / 2.0, 5, true);
                TestIndex::TestFilterSearch(index, dataset, search_param, recall, true, true);
                TestIndex::TestCheckIdExist(index, dataset);
                TestIndex::TestCalcDistanceById(index, dataset);
                TestIndex::TestGetRawVectorByIds(index, dataset);
                TestIndex::TestBatchCalcDistanceById(index, dataset);
                TestIndex::TestSearchAllocator(index, dataset, search_param, recall, true);
                auto index2 = TestIndex::TestFactory(test_index->name, param, true);
                TestIndex::TestSerializeFile(index, index2, dataset, search_param, true);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph Duplicate", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphDuplicate(test_index, resource);
}

TEST_CASE("[Daily] HGraph Duplicate", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphDuplicate(test_index, resource);
}

static void
TestHGraphSearchWithDirtyVector(const fixtures::HGraphTestIndexPtr& test_index,
                                const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);

    for (auto metric_type : resource->metric_types) {
        auto dataset = HGraphTestIndex::pool.GetNanDataset(metric_type);
        auto dim = dataset->dim_;

        for (auto& [base_quantization_str, recall] : resource->test_cases) {
            INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                             metric_type,
                             dim,
                             base_quantization_str,
                             recall));
            if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                continue;  // Skip invalid RaBitQ configurations
            }
            vsag::Options::Instance().set_block_size_limit(size);
            HGraphTestIndex::HGraphBuildParam build_param(metric_type, dim, base_quantization_str);
            auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
            auto index = TestIndex::TestFactory(test_index->name, param, true);
            TestIndex::TestBuildIndex(index, dataset, true);
            TestIndex::TestSearchWithDirtyVector(index, dataset, search_param, true);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE("[PR] HGraph Search with Dirty Vector", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphSearchWithDirtyVector(test_index, resource);
}

TEST_CASE("[Daily] HGraph Search with Dirty Vector", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphSearchWithDirtyVector(test_index, resource);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HGraphTestIndex,
                             "HGraph Search with Sparse Vector",
                             "[ft][hgraph][concurrent]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = "ip";
    INFO(fmt::format("metric_type: {}", metric_type));
    auto dim = 128;
    auto dataset = pool.GetSparseDatasetAndCreate(base_count, dim, 0.8);
    auto search_param = fmt::format(fixtures::search_param_tmp, 100, false);
    vsag::Options::Instance().set_block_size_limit(size);

    HGraphTestIndex::HGraphBuildParam build_param(metric_type, dim, "sparse");
    build_param.data_type = "sparse";
    auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
    auto index = TestFactory(name, param, true);
    TestConcurrentAdd(index, dataset, true);
    TestKnnSearch(index, dataset, search_param, true);
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

static void
TestHGraphConcurrentAdd(const fixtures::HGraphTestIndexPtr& test_index,
                        const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }

                // Set block size limit for current test iteration
                vsag::Options::Instance().set_block_size_limit(size);

                // Generate index parameters with attribute support enabled
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);

                // Execute build test
                TestIndex::TestConcurrentAdd(index, dataset, true);
                if (index->CheckFeature(vsag::SUPPORT_ADD_CONCURRENT)) {
                    HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);
                }
                // Restore original block size limit
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph Concurrent Add", "[ft][hgraph][pr][concurrent]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphConcurrentAdd(test_index, resource);
}

TEST_CASE("[Daily] HGraph Concurrent Add", "[ft][hgraph][daily][concurrent]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphConcurrentAdd(test_index, resource);
}

static void
TestHGraphSerialize(const fixtures::HGraphTestIndexPtr& test_index,
                    const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    uint64_t extra_info_size = 64;

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.extra_info_size = extra_info_size;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(dim,
                                                                         resource->base_count,
                                                                         metric_type,
                                                                         false /*with_path*/,
                                                                         0.8 /*valid_ratio*/,
                                                                         extra_info_size);
                TestIndex::TestBuildIndex(index, dataset, true);
                auto index2 = TestIndex::TestFactory(test_index->name, param, true);
                TestIndex::TestSerializeFile(index, index2, dataset, search_param, true);
                index2 = TestIndex::TestFactory(test_index->name, param, true);
                TestIndex::TestSerializeBinarySet(index, index2, dataset, search_param, true);
                index2 = TestIndex::TestFactory(test_index->name, param, true);
                TestIndex::TestSerializeReaderSet(
                    index, index2, dataset, search_param, test_index->name, true);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph Serialize File", "[ft][hgraph][serialization][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphSerialize(test_index, resource);
}

TEST_CASE("[Daily] HGraph Serialize File", "[ft][hgraph][serialization][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphSerialize(test_index, resource);
}

static void
TestHGraphReaderIO(const fixtures::HGraphTestIndexPtr& test_index,
                   const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    uint64_t extra_info_size = 64;

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    (metric_type != "l2" || dim < fixtures::RABITQ_MIN_RACALL_DIM)) {
                    continue;  // Skip invalid RaBitQ configurations
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.extra_info_size = extra_info_size;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(dim,
                                                                         resource->base_count,
                                                                         metric_type,
                                                                         false /*with_path*/,
                                                                         0.8 /*valid_ratio*/,
                                                                         extra_info_size);

                TestIndex::TestBuildIndex(index, dataset, true);
                if (base_quantization_str.find(',') != std::string::npos) {
                    build_param.quantization_str += ",reader_io";
                }
                auto reader_param =
                    HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index2 = TestIndex::TestFactory(test_index->name, reader_param, true);
                TestIndex::TestSerializeReaderSet(
                    index, index2, dataset, search_param, test_index->name, true);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph Reader IO", "[ft][hgraph][serialization][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphReaderIO(test_index, resource);
}

TEST_CASE("[Daily] HGraph Reader IO", "[ft][hgraph][serialization][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphReaderIO(test_index, resource);
}

static void
TestHGraphClone(const fixtures::HGraphTestIndexPtr& test_index,
                const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    uint64_t extra_info_size = 64;

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.extra_info_size = extra_info_size;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(dim,
                                                                         resource->base_count,
                                                                         metric_type,
                                                                         false /*with_path*/,
                                                                         0.8 /*valid_ratio*/,
                                                                         extra_info_size);
                TestIndex::TestBuildIndex(index, dataset, true);
                TestIndex::TestClone(index, dataset, search_param);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph Clone", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphClone(test_index, resource);
}

TEST_CASE("[Daily] HGraph Clone", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphClone(test_index, resource);
}

static void
TestHGraphExportModel(const fixtures::HGraphTestIndexPtr& test_index,
                      const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    uint64_t extra_info_size = 64;

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.extra_info_size = extra_info_size;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(dim,
                                                                         resource->base_count,
                                                                         metric_type,
                                                                         false /*with_path*/,
                                                                         0.8 /*valid_ratio*/,
                                                                         extra_info_size);
                TestIndex::TestBuildIndex(index, dataset, true);
                TestIndex::TestExportModel(index, dataset, search_param);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph Export Model", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphExportModel(test_index, resource);
}

TEST_CASE("[Daily] HGraph Export Model", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphExportModel(test_index, resource);
}

static void
TestHGraphRandomAllocator(const fixtures::HGraphTestIndexPtr& test_index,
                          const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto allocator = std::make_shared<fixtures::RandomAllocator>();

    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    uint64_t extra_info_size = 64;

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.thread_count = 1;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = vsag::Factory::CreateIndex(test_index->name, param, allocator.get());
                if (not index.has_value()) {
                    continue;
                }
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestContinueAddIgnoreRequire(index.value(), dataset);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph Build & ContinueAdd Test With Random Allocator", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphRandomAllocator(test_index, resource);
}

TEST_CASE("[Daily] HGraph Build & ContinueAdd Test With Random Allocator", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphRandomAllocator(test_index, resource);
}

static void
TestHGraphDuplicateBuild(const fixtures::HGraphTestIndexPtr& test_index,
                         const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    uint64_t extra_info_size = 64;

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }
                vsag::Options::Instance().set_block_size_limit(size);

                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestDuplicateAdd(index, dataset);
                HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph Duplicate Build", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphDuplicateBuild(test_index, resource);
}

TEST_CASE("[Daily] HGraph Duplicate Build", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphDuplicateBuild(test_index, resource);
}

static void
TestHGraphEstimateMemory(const fixtures::HGraphTestIndexPtr& test_index,
                         const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    uint64_t extra_info_size = 64;
    uint64_t estimate_count = 1000;

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }
                vsag::Options::Instance().set_block_size_limit(size);
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.extra_info_size = extra_info_size;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(dim,
                                                                         estimate_count,
                                                                         metric_type,
                                                                         false /*with_path*/,
                                                                         0.8 /*valid_ratio*/,
                                                                         extra_info_size);
                TestIndex::TestEstimateMemory(test_index->name, param, dataset);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph Estimate Memory", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphEstimateMemory(test_index, resource);
}

TEST_CASE("[Daily] HGraph Estimate Memory", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphEstimateMemory(test_index, resource);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HGraphTestIndex, "HGraph ELP Optimizer", "[ft][hgraph]") {
    fixtures::logger::LoggerReplacer _;
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);

    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = fixtures::RandomSelect<std::string>({"l2", "ip", "cosine"})[0];
    INFO(fmt::format("metric_type: {}", metric_type));

    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
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
static void
TestHGraphIgnoreReorder(const fixtures::HGraphTestIndexPtr& test_index,
                        const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
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
    float recall = 0.95;
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            INFO(fmt::format("metric_type: {}, dim: {}, recall: {}", metric_type, dim, recall));
            vsag::Options::Instance().set_block_size_limit(size);
            auto dataset =
                HGraphTestIndex::pool.GetDatasetAndCreate(dim, resource->base_count, metric_type);
            std::string param = fmt::format(parameter_temp_reorder, metric_type, dim);
            auto index = TestIndex::TestFactory(test_index->name, param, true);
            TestIndex::TestBuildIndex(index, dataset);
            HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE("[PR] HGraph Ignore Reorder", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphIgnoreReorder(test_index, resource);
}

TEST_CASE("[Daily] HGraph Ignore Reorder", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphIgnoreReorder(test_index, resource);
}

static void
TestHGraphWithExtraInfo(const fixtures::HGraphTestIndexPtr& test_index,
                        const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    uint64_t extra_info_size = 256;
    auto search_ex_filter_param = fmt::format(fixtures::search_param_tmp, 500, true);

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.extra_info_size = extra_info_size;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(dim,
                                                                         resource->base_count,
                                                                         metric_type,
                                                                         false /*with_path*/,
                                                                         0.8 /*valid_ratio*/,
                                                                         extra_info_size);
                TestIndex::TestBuildIndex(index, dataset, true);
                TestIndex::TestKnnSearch(index, dataset, search_param, recall, true);
                TestIndex::TestKnnSearchIter(index, dataset, search_param, recall, true);
                TestIndex::TestRangeSearch(index, dataset, search_param, recall, 10, true);
                TestIndex::TestGetExtraInfoById(index, dataset, extra_info_size);
                TestIndex::TestKnnSearchExFilter(
                    index, dataset, search_ex_filter_param, recall, true);
                TestIndex::TestUpdateExtraInfo(index, dataset, extra_info_size);
                TestIndex::TestKnnSearchExFilter(
                    index, dataset, search_ex_filter_param, recall, true);
                TestIndex::TestKnnSearchIter(
                    index, dataset, search_ex_filter_param, recall, true, true);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph With Extra Info", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphWithExtraInfo(test_index, resource);
}

TEST_CASE("[Daily] HGraph With Extra Info", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphWithExtraInfo(test_index, resource);
}

static void
TestHGraphSearchOverTime(const fixtures::HGraphTestIndexPtr& test_index,
                         const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    constexpr const char* search_param = R"({
            "hgraph": {
                "ef_search": 200,
                "timeout_ms": 5.0
            }
        })";
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestBuildIndex(index, dataset, true);
                TestIndex::TestSearchOvertime(index, dataset, search_param);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph Search Over Time", "[ft][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphSearchOverTime(test_index, resource);
}

TEST_CASE("[Daily] HGraph Search Over Time", "[ft][hgraph][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphSearchOverTime(test_index, resource);
}

static void
TestHGraphDiskIOType(const fixtures::HGraphTestIndexPtr& test_index,
                     const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    float recall = 0.98;
    const std::vector<std::pair<std::string, std::string>> io_cases = {
        {"sq8_uniform,bf16", "sq8_uniform,bf16,buffer_io"},
        {"rabitq,fp16", "rabitq,fp16,async_io"},
        {"rabitq,fp16", "rabitq,fp16,mmap_io"},
    };
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [memory_io_str, disk_io_str] : io_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, memory_io_str: {}, disk_io_str: {}",
                                 metric_type,
                                 dim,
                                 memory_io_str,
                                 disk_io_str));
                if (HGraphTestIndex::IsRaBitQ(memory_io_str) &&
                    (dim < fixtures::RABITQ_MIN_RACALL_DIM)) {
                    continue;  // Skip invalid RaBitQ configurations
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(metric_type, dim, memory_io_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestBuildIndex(index, dataset, true);
                build_param.quantization_str = disk_io_str;
                param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto disk_index = TestIndex::TestFactory(test_index->name, param, true);
                TestIndex::TestSerializeFile(index, disk_index, dataset, search_param, true);
                HGraphTestIndex::TestGeneral(disk_index, dataset, search_param, recall);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("[PR] HGraph Disk IO Type Index", "[ft][hgraph][serialization][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphDiskIOType(test_index, resource);
}

TEST_CASE("[Daily]HGraph Disk IO Type Index", "[ft][hgraph][serialization][daily]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(false);
    TestHGraphDiskIOType(test_index, resource);
}
