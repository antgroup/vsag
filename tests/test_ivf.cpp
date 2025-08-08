
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

#include "fixtures/fixtures.h"
#include "fixtures/test_dataset_pool.h"
#include "storage/serialization_template_test.h"
#include "test_index.h"
namespace fixtures {

class IVFTestResource {
public:
    std::vector<int> dims;
    std::vector<std::pair<std::string, float>> test_cases;
    std::vector<std::string> metric_types;
    std::vector<std::string> train_types;
    uint64_t base_count;
};
using IVFResourcePtr = std::shared_ptr<IVFTestResource>;

class IVFTestIndex : public fixtures::TestIndex {
public:
    static std::string
    GenerateIVFBuildParametersString(const std::string& metric_type,
                                     int64_t dim,
                                     const std::string& quantization_str = "sq8",
                                     int buckets_count = 210,
                                     const std::string& train_type = "kmeans",
                                     bool use_residual = false,
                                     int buckets_per_data = 1,
                                     bool use_attr_filter = false,
                                     int thread_count = 1);

    static IVFResourcePtr
    GetResource(bool sample = true);

    static std::string
    GenerateGNOIMIBuildParametersString(const std::string& metric_type,
                                        int64_t dim,
                                        const std::string& quantization_str = "sq8",
                                        int first_order_buckets_count = 15,
                                        int second_order_buckets_count = 15,
                                        const std::string& train_type = "kmeans",
                                        bool use_residual = false,
                                        int buckets_per_data = 1,
                                        int thread_count = 1);
    static void
    TestGeneral(const IndexPtr& index,
                const TestDatasetPtr& dataset,
                const std::string& search_param,
                float recall);

    static TestDatasetPool pool;

    static fixtures::TempDir dir;

    static const std::string name;

    constexpr static uint64_t base_count = 1200;

    static const std::vector<std::pair<std::string, float>> all_test_cases;
};

using IVFTestIndexPtr = std::shared_ptr<IVFTestIndex>;

TestDatasetPool IVFTestIndex::pool{};
fixtures::TempDir IVFTestIndex::dir{"ivf_test"};
const std::string IVFTestIndex::name = "ivf";

// DON'T WORRY! IVF just can't achieve high recall on random datasets. so we set the expected
// recall with a small number in test cases
const std::vector<std::pair<std::string, float>> IVFTestIndex::all_test_cases = {
    {"fp32", 0.90},
    {"bf16", 0.88},
    {"fp16", 0.88},
    {"sq8", 0.84},
    {"sq8_uniform,fp32", 0.89},
    {"pq,fp32", 0.82},
    {"pqfs,fp16", 0.82},
};

IVFResourcePtr
IVFTestIndex::GetResource(bool sample) {
    auto resource = std::make_shared<IVFTestResource>();
    if (sample) {
        resource->dims = fixtures::get_common_used_dims(1, RandomValue(0, 999));
        resource->test_cases = fixtures::RandomSelect(IVFTestIndex::all_test_cases, 3);
        resource->metric_types = fixtures::RandomSelect<std::string>({"ip", "l2", "cosine"}, 1);
        resource->train_types = fixtures::RandomSelect<std::string>({"kmeans", "random"}, 1);
        resource->base_count = IVFTestIndex::base_count;
    } else {
        resource->dims = fixtures::get_common_used_dims();
        resource->test_cases = IVFTestIndex::all_test_cases;
        resource->metric_types = {"ip", "l2", "cosine"};
        resource->train_types = {"kmeans", "random"};
        resource->base_count = IVFTestIndex::base_count * 10;
    }
    return resource;
}

constexpr static const char* search_param_tmp = R"(
        {{
            "ivf": {{
                "scan_buckets_count": {},
                "factor": 4.0,
                "first_order_scan_ratio": 1.0
            }}
        }})";

std::string
IVFTestIndex::GenerateIVFBuildParametersString(const std::string& metric_type,
                                               int64_t dim,
                                               const std::string& quantization_str,
                                               int buckets_count,
                                               const std::string& train_type,
                                               bool use_residual,
                                               int buckets_per_data,
                                               bool use_attr_filter,
                                               int thread_count) {
    std::string build_parameters_str;

    constexpr auto parameter_temp = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "buckets_count": {},
            "base_quantization_type": "{}",
            "ivf_train_type": "{}",
            "use_reorder": {},
            "base_pq_dim": {},
            "precise_quantization_type": "{}",
            "use_residual": {},
            "buckets_per_data": {},
            "use_attribute_filter": {},
            "thread_count": {}
        }}
    }}
    )";

    auto strs = fixtures::SplitString(quantization_str, ',');
    std::string basic_quantizer_str = strs[0];
    bool use_reorder = false;
    std::string precise_quantizer_str = "fp32";
    auto pq_dim = dim;
    if (dim % 2 == 0 && basic_quantizer_str == "pq") {
        pq_dim = dim / 2;
    }
    if (strs.size() == 2) {
        use_reorder = true;
        precise_quantizer_str = strs[1];
    }
    build_parameters_str = fmt::format(parameter_temp,
                                       metric_type,
                                       dim,
                                       buckets_count,
                                       basic_quantizer_str,
                                       train_type,
                                       use_reorder,
                                       pq_dim,
                                       precise_quantizer_str,
                                       use_residual,
                                       buckets_per_data,
                                       use_attr_filter,
                                       thread_count);
    INFO(build_parameters_str);
    return build_parameters_str;
}

std::string
IVFTestIndex::GenerateGNOIMIBuildParametersString(const std::string& metric_type,
                                                  int64_t dim,
                                                  const std::string& quantization_str,
                                                  int first_order_buckets_count,
                                                  int second_order_buckets_count,
                                                  const std::string& train_type,
                                                  bool use_residual,
                                                  int buckets_per_data,
                                                  int thread_count) {
    std::string build_parameters_str;

    constexpr auto parameter_temp = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "first_order_buckets_count": {},
            "second_order_buckets_count": {},
            "base_quantization_type": "{}",
            "ivf_train_type": "{}",
            "use_reorder": {},
            "base_pq_dim": {},
            "precise_quantization_type": "{}",
            "use_residual": {},
            "buckets_per_data": {},
            "thread_count": {},
            "partition_strategy_type": "gno_imi"
        }}
    }}
    )";

    auto strs = fixtures::SplitString(quantization_str, ',');
    std::string basic_quantizer_str = strs[0];
    bool use_reorder = false;
    std::string precise_quantizer_str = "fp32";
    auto pq_dim = dim;
    if (dim % 2 == 0 && basic_quantizer_str == "pq") {
        pq_dim = dim / 2;
    }
    if (strs.size() == 2) {
        use_reorder = true;
        precise_quantizer_str = strs[1];
    }
    build_parameters_str = fmt::format(parameter_temp,
                                       metric_type,
                                       dim,
                                       first_order_buckets_count,
                                       second_order_buckets_count,
                                       basic_quantizer_str,
                                       train_type,
                                       use_reorder,
                                       pq_dim,
                                       precise_quantizer_str,
                                       use_residual,
                                       buckets_per_data,
                                       thread_count);

    INFO(build_parameters_str);
    return build_parameters_str;
}

void
IVFTestIndex::TestGeneral(const TestIndex::IndexPtr& index,
                          const TestDatasetPtr& dataset,
                          const std::string& search_param,
                          float recall) {
    REQUIRE(index->GetIndexType() == vsag::IndexType::IVF);
    TestKnnSearch(index, dataset, search_param, recall, true);
    TestConcurrentKnnSearch(index, dataset, search_param, recall, true);
    TestRangeSearch(index, dataset, search_param, recall, 10, true);
    TestRangeSearch(index, dataset, search_param, recall / 2.0, 5, true);
    TestFilterSearch(index, dataset, search_param, recall, true);
    TestCheckIdExist(index, dataset);
}
}  // namespace fixtures

TEST_CASE_PERSISTENT_FIXTURE(fixtures::IVFTestIndex,
                             "IVF Factory Test With Exceptions",
                             "[ft][ivf]") {
    auto name = "ivf";
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

    SECTION("Miss ivf param") {
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

    SECTION("Invalid ivf param base_quantization_type") {
        auto base_quantization_types = GENERATE("fsa", "aq");
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

    SECTION("Invalid ivf param key") {
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

static void
TestIVFBuildAndContinueAdd(const fixtures::IVFTestIndexPtr& test_index,
                           const fixtures::IVFResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto train_type : resource->train_types) {
                for (auto [base_quantization_str, recall] : resource->test_cases) {
                    if (train_type == "kmeans") {
                        recall *= 0.8F;  // Kmeans may not achieve high recall in random datasets
                    }
                    auto count = std::min(300, static_cast<int32_t>(dim / 4));
                    auto search_param =
                        fmt::format(fixtures::search_param_tmp, std::max(250, count));
                    INFO(
                        fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, "
                                    "train_type: {}, recall: {}",
                                    metric_type,
                                    dim,
                                    base_quantization_str,
                                    train_type,
                                    recall));
                    vsag::Options::Instance().set_block_size_limit(size);
                    auto param = IVFTestIndex::GenerateIVFBuildParametersString(
                        metric_type, dim, base_quantization_str, 300, train_type);
                    auto index = TestIndex::TestFactory(test_index->name, param, true);
                    auto dataset = IVFTestIndex::pool.GetDatasetAndCreate(
                        dim, resource->base_count, metric_type);
                    TestIndex::TestContinueAdd(index, dataset, true);
                    if (index->CheckFeature(vsag::SUPPORT_ADD_AFTER_BUILD)) {
                        IVFTestIndex::TestGeneral(index, dataset, search_param, recall);
                    }
                    vsag::Options::Instance().set_block_size_limit(origin_size);
                }
            }
        }
    }
}

TEST_CASE("[PR] IVF Build & ContinueAdd Test", "[ft][ivf][pr][concurrent]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(true);
    TestIVFBuildAndContinueAdd(test_index, resource);
}

TEST_CASE("[Daily] IVF Build & ContinueAdd Test", "[ft][ivf][daily][concurrent]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(false);
    TestIVFBuildAndContinueAdd(test_index, resource);
}

static void
TestIVFBuildWithResidual(const fixtures::IVFTestIndexPtr& test_index,
                         const fixtures::IVFResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    std::vector<std::pair<std::string, float>> tmp_test_cases = {
        {"fp32", 0.90},
        {"bf16", 0.88},
        {"fp16", 0.88},
        {"sq8", 0.84},
        {"pq,fp32", 0.82},
        {"pqfs,fp32", 0.82},
    };
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto train_type : resource->train_types) {
                for (auto [base_quantization_str, recall] : tmp_test_cases) {
                    auto count = std::min(300, static_cast<int32_t>(dim / 4));
                    if (train_type == "kmeans") {
                        recall *= 0.8F;  // Kmeans may not achieve high recall in random datasets
                    }
                    auto search_param =
                        fmt::format(fixtures::search_param_tmp, std::max(250, count));
                    INFO(
                        fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, "
                                    "train_type: {}, recall: {}",
                                    metric_type,
                                    dim,
                                    base_quantization_str,
                                    train_type,
                                    recall));
                    vsag::Options::Instance().set_block_size_limit(size);
                    auto param = IVFTestIndex::GenerateIVFBuildParametersString(
                        metric_type, dim, base_quantization_str, 300, train_type, true);
                    auto index = IVFTestIndex::TestFactory(test_index->name, param, true);
                    auto dataset = IVFTestIndex::pool.GetDatasetAndCreate(
                        dim, resource->base_count, metric_type);
                    IVFTestIndex::TestContinueAdd(index, dataset, true);
                    if (index->CheckFeature(vsag::SUPPORT_ADD_AFTER_BUILD)) {
                        IVFTestIndex::TestGeneral(index, dataset, search_param, recall);
                    }
                    vsag::Options::Instance().set_block_size_limit(origin_size);
                }
            }
        }
    }
}

TEST_CASE("[PR] IVF Build with Residual", "[ft][ivf][pr]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(true);
    TestIVFBuildWithResidual(test_index, resource);
}

TEST_CASE("[Daily] IVF Build with Residual", "[ft][ivf][daily]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(false);
    TestIVFBuildWithResidual(test_index, resource);
}

static void
TestIVFBuild(const fixtures::IVFTestIndexPtr& test_index,
             const fixtures::IVFResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    constexpr static const char* search_param_tmp2 = R"(
        {{
            "ivf": {{
                "scan_buckets_count": {},
                "factor": 4.0,
                "first_order_scan_ratio": 1.0,
                "parallelism": {}
            }}
        }})";
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto train_type : resource->train_types) {
                for (auto [base_quantization_str, recall] : resource->test_cases) {
                    auto count = std::min(300, static_cast<int32_t>(dim / 4));
                    if (train_type == "kmeans") {
                        recall *= 0.8F;  // Kmeans may not achieve high recall in random datasets
                    }
                    auto search_thread_count = GENERATE(1, 3);
                    auto search_param =
                        fmt::format(search_param_tmp2, std::max(200, count), search_thread_count);
                    INFO(
                        fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, "
                                    "train_type: {}, recall: {}",
                                    metric_type,
                                    dim,
                                    base_quantization_str,
                                    train_type,
                                    recall));
                    vsag::Options::Instance().set_block_size_limit(size);
                    auto param =
                        IVFTestIndex::GenerateIVFBuildParametersString(metric_type,
                                                                       dim,
                                                                       base_quantization_str,
                                                                       300,
                                                                       train_type,
                                                                       false,
                                                                       1,
                                                                       false,
                                                                       3);
                    auto index = IVFTestIndex::TestFactory(IVFTestIndex::name, param, true);
                    auto dataset = IVFTestIndex::pool.GetDatasetAndCreate(
                        dim, resource->base_count, metric_type);
                    IVFTestIndex::TestBuildIndex(index, dataset, true);
                    if (index->CheckFeature(vsag::SUPPORT_BUILD)) {
                        IVFTestIndex::TestGeneral(index, dataset, search_param, recall);
                    }
                    vsag::Options::Instance().set_block_size_limit(origin_size);
                }
            }
        }
    }
}

TEST_CASE("[PR] IVF Build", "[ft][ivf][pr]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(true);
    TestIVFBuild(test_index, resource);
}

TEST_CASE("[Daily] IVF Build", "[ft][ivf][daily]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(false);
    TestIVFBuild(test_index, resource);
}

static void
TestIVFSearchOvertime(const fixtures::IVFTestIndexPtr& test_index,
                      const fixtures::IVFResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    constexpr static const char* search_param_tmp2 = R"(
        {{
            "ivf": {{
                "scan_buckets_count": {},
                "factor": 4.0,
                "first_order_scan_ratio": 1.0,
                "parallelism": {},
                "timeout_ms": 20.0
            }}
        }})";
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto train_type : resource->train_types) {
                for (auto [base_quantization_str, recall] : resource->test_cases) {
                    auto count = std::min(300, static_cast<int32_t>(dim / 4));
                    if (train_type == "kmeans") {
                        recall *= 0.8F;  // Kmeans may not achieve high recall in random datasets
                    }
                    auto search_thread_count = GENERATE(1, 3);
                    auto search_param =
                        fmt::format(search_param_tmp2, std::max(200, count), search_thread_count);
                    INFO(
                        fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, "
                                    "train_type: {}, recall: {}",
                                    metric_type,
                                    dim,
                                    base_quantization_str,
                                    train_type,
                                    recall));
                    vsag::Options::Instance().set_block_size_limit(size);
                    auto param =
                        IVFTestIndex::GenerateIVFBuildParametersString(metric_type,
                                                                       dim,
                                                                       base_quantization_str,
                                                                       300,
                                                                       train_type,
                                                                       false,
                                                                       1,
                                                                       false,
                                                                       3);
                    auto index = IVFTestIndex::TestFactory(IVFTestIndex::name, param, true);
                    auto dataset = IVFTestIndex::pool.GetDatasetAndCreate(
                        dim, resource->base_count, metric_type);
                    IVFTestIndex::TestBuildIndex(index, dataset, true);
                    IVFTestIndex::TestSearchOvertime(index, dataset, search_param);
                    vsag::Options::Instance().set_block_size_limit(origin_size);
                }
            }
        }
    }
}

TEST_CASE("[PR] IVF Search Overtime", "[ft][ivf][pr]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(true);
    TestIVFSearchOvertime(test_index, resource);
}

TEST_CASE("[Daily] IVF Search Overtime", "[ft][ivf][daily]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(false);
    TestIVFSearchOvertime(test_index, resource);
}

static void
TestIVFBuildWithLargeK(const fixtures::IVFTestIndexPtr& test_index,
                       const fixtures::IVFResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    std::vector<std::pair<std::string, float>> tmp_test_cases = {
        {"fp32", 0.75},
    };
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto train_type : resource->train_types) {
                for (auto [base_quantization_str, recall] : tmp_test_cases) {
                    if (train_type == "kmeans") {
                        recall *= 0.8F;  // Kmeans may not achieve high recall in random datasets
                    }
                    auto search_param = fmt::format(fixtures::search_param_tmp, 3000);
                    INFO(
                        fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, "
                                    "train_type: {}, recall: {}",
                                    metric_type,
                                    dim,
                                    base_quantization_str,
                                    train_type,
                                    recall));

                    vsag::Options::Instance().set_block_size_limit(size);
                    auto param = IVFTestIndex::GenerateIVFBuildParametersString(
                        metric_type, dim, base_quantization_str, 10000, train_type);
                    auto index = IVFTestIndex::TestFactory(IVFTestIndex::name, param, true);
                    auto dataset = IVFTestIndex::pool.GetDatasetAndCreate(dim, 20000, metric_type);
                    IVFTestIndex::TestBuildIndex(index, dataset, true);
                    if (index->CheckFeature(vsag::SUPPORT_BUILD)) {
                        IVFTestIndex::TestGeneral(index, dataset, search_param, recall);
                    }
                    vsag::Options::Instance().set_block_size_limit(origin_size);
                }
            }
        }
    }
}

TEST_CASE("[PR] IVF Build With Large K", "[ft][ivf][pr]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(true);
    TestIVFBuildWithLargeK(test_index, resource);
}

TEST_CASE("[Daily] IVF Build With Large K", "[ft][ivf][daily]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(false);
    TestIVFBuildWithLargeK(test_index, resource);
}

static void
TestIVFWithAttr(const fixtures::IVFTestIndexPtr& test_index,
                const fixtures::IVFResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    bool use_attribute_filter = true;
    std::vector<std::pair<std::string, float>> tmp_test_cases = {
        {"fp32", 0.75},
    };
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto train_type : resource->train_types) {
                for (auto [base_quantization_str, recall] : tmp_test_cases) {
                    if (train_type == "kmeans") {
                        recall *= 0.8F;  // Kmeans may not achieve high recall in random datasets
                    }
                    auto count = std::min(300, static_cast<int32_t>(dim / 4));
                    auto search_param =
                        fmt::format(fixtures::search_param_tmp, std::max(200, count));
                    INFO(
                        fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, "
                                    "train_type: {}, recall: {}",
                                    metric_type,
                                    dim,
                                    base_quantization_str,
                                    train_type,
                                    recall));
                    vsag::Options::Instance().set_block_size_limit(size);
                    auto param =
                        IVFTestIndex::GenerateIVFBuildParametersString(metric_type,
                                                                       dim,
                                                                       base_quantization_str,
                                                                       300,
                                                                       train_type,
                                                                       false,
                                                                       1,
                                                                       use_attribute_filter);
                    auto index1 = IVFTestIndex::TestFactory(IVFTestIndex::name, param, true);
                    auto dataset = IVFTestIndex::pool.GetDatasetAndCreate(
                        dim, resource->base_count, metric_type);
                    if (not index1->CheckFeature(vsag::SUPPORT_BUILD)) {
                        continue;
                    }
                    auto build_result = index1->Build(dataset->base_);
                    REQUIRE(build_result.has_value());
                    IVFTestIndex::TestWithAttr(index1, dataset, search_param, false);
                    auto index = TestIndex::TestFactory(IVFTestIndex::name, param, true);

                    REQUIRE_NOTHROW(test_serializion_file(*index1, *index, "serialize"));

                    IVFTestIndex::TestWithAttr(index, dataset, search_param);

                    vsag::Options::Instance().set_block_size_limit(origin_size);
                }
            }
        }
    }
}

TEST_CASE("[PR] IVF Build With Attribute", "[ft][ivf][pr]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(true);
    TestIVFWithAttr(test_index, resource);
}

TEST_CASE("[Daily] IVF Build With Attribute", "[ft][ivf][daily]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(false);
    TestIVFWithAttr(test_index, resource);
}

static void
TestIVFExportModel(const fixtures::IVFTestIndexPtr& test_index,
                   const fixtures::IVFResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto train_type : resource->train_types) {
                for (auto [base_quantization_str, recall] : resource->test_cases) {
                    if (train_type == "kmeans") {
                        recall *= 0.8F;  // Kmeans may not achieve high recall in random datasets
                    }
                    auto count = std::min(300, static_cast<int32_t>(dim / 4));
                    auto search_param =
                        fmt::format(fixtures::search_param_tmp, std::max(200, count));
                    INFO(
                        fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, "
                                    "train_type: {}, recall: {}",
                                    metric_type,
                                    dim,
                                    base_quantization_str,
                                    train_type,
                                    recall));
                    vsag::Options::Instance().set_block_size_limit(size);
                    auto param = IVFTestIndex::GenerateIVFBuildParametersString(
                        metric_type, dim, base_quantization_str, 300, train_type);
                    auto index = IVFTestIndex::TestFactory(IVFTestIndex::name, param, true);
                    auto dataset = IVFTestIndex::pool.GetDatasetAndCreate(
                        dim, resource->base_count, metric_type);

                    IVFTestIndex::TestBuildIndex(index, dataset, true);
                    IVFTestIndex::TestExportModel(index, dataset, search_param);

                    vsag::Options::Instance().set_block_size_limit(origin_size);
                }
            }
        }
    }
}

TEST_CASE("[PR] IVF Export Model", "[ft][ivf][pr]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(true);
    TestIVFExportModel(test_index, resource);
}

TEST_CASE("[Daily] IVF IVF Export Model", "[ft][ivf][daily]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(false);
    TestIVFExportModel(test_index, resource);
}

static void
TestIVFAdd(const fixtures::IVFTestIndexPtr& test_index, const fixtures::IVFResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto train_type : resource->train_types) {
                for (auto [base_quantization_str, recall] : resource->test_cases) {
                    if (train_type == "kmeans") {
                        recall *= 0.8F;  // Kmeans may not achieve high recall in random datasets
                    }
                    auto count = std::min(300, static_cast<int32_t>(dim / 4));
                    auto search_param =
                        fmt::format(fixtures::search_param_tmp, std::max(200, count));
                    INFO(
                        fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, "
                                    "train_type: {}, recall: {}",
                                    metric_type,
                                    dim,
                                    base_quantization_str,
                                    train_type,
                                    recall));
                    vsag::Options::Instance().set_block_size_limit(size);
                    auto param = IVFTestIndex::GenerateIVFBuildParametersString(
                        metric_type, dim, base_quantization_str, 300, train_type);
                    auto index = IVFTestIndex::TestFactory(IVFTestIndex::name, param, true);
                    auto dataset = IVFTestIndex::pool.GetDatasetAndCreate(
                        dim, resource->base_count, metric_type);
                    IVFTestIndex::TestAddIndex(index, dataset, true);
                    if (index->CheckFeature(vsag::SUPPORT_ADD_FROM_EMPTY)) {
                        IVFTestIndex::TestGeneral(index, dataset, search_param, recall);
                    }
                    vsag::Options::Instance().set_block_size_limit(origin_size);
                }
            }
        }
    }
}

TEST_CASE("[PR] IVF Add", "[ft][ivf][pr]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(true);
    TestIVFAdd(test_index, resource);
}

TEST_CASE("[Daily] IVF Add", "[ft][ivf][daily]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(false);
    TestIVFAdd(test_index, resource);
}

static void
TestIVFMerge(const fixtures::IVFTestIndexPtr& test_index,
             const fixtures::IVFResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto train_type : resource->train_types) {
                for (auto [base_quantization_str, recall] : resource->test_cases) {
                    if (train_type == "kmeans") {
                        recall *= 0.8F;  // Kmeans may not achieve high recall in random datasets
                    }
                    auto count = std::min(300, static_cast<int32_t>(dim / 4));
                    auto search_param =
                        fmt::format(fixtures::search_param_tmp, std::max(200, count));
                    INFO(
                        fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, "
                                    "train_type: {}, recall: {}",
                                    metric_type,
                                    dim,
                                    base_quantization_str,
                                    train_type,
                                    recall));
                    vsag::Options::Instance().set_block_size_limit(size);
                    auto param = IVFTestIndex::GenerateIVFBuildParametersString(
                        metric_type, dim, base_quantization_str, 300, train_type);
                    auto model = IVFTestIndex::TestFactory(IVFTestIndex::name, param, true);
                    auto dataset = IVFTestIndex::pool.GetDatasetAndCreate(
                        dim, resource->base_count, metric_type);
                    auto ret = model->Train(dataset->base_);
                    REQUIRE(ret.has_value() == true);
                    auto merge_index =
                        IVFTestIndex::TestMergeIndexWithSameModel(model, dataset, 5, true);
                    if (model->CheckFeature(vsag::SUPPORT_MERGE_INDEX)) {
                        IVFTestIndex::TestGeneral(merge_index, dataset, search_param, recall);
                    }
                    vsag::Options::Instance().set_block_size_limit(origin_size);
                }
            }
        }
    }
}

TEST_CASE("[PR] IVF Merge", "[ft][ivf][pr]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(true);
    TestIVFMerge(test_index, resource);
}

TEST_CASE("[Daily] IVF Merge", "[ft][ivf][daily]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(false);
    TestIVFMerge(test_index, resource);
}

static void
TestIVFConcurrentAdd(const fixtures::IVFTestIndexPtr& test_index,
                     const fixtures::IVFResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto train_type : resource->train_types) {
                for (auto [base_quantization_str, recall] : resource->test_cases) {
                    if (train_type == "kmeans") {
                        recall *= 0.8F;  // Kmeans may not achieve high recall in random datasets
                    }
                    if (base_quantization_str == "pqfs,fp16") {
                        continue;
                    }
                    auto count = std::min(300, static_cast<int32_t>(dim / 4));
                    auto search_param =
                        fmt::format(fixtures::search_param_tmp, std::max(200, count));
                    INFO(
                        fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, "
                                    "train_type: {}, recall: {}",
                                    metric_type,
                                    dim,
                                    base_quantization_str,
                                    train_type,
                                    recall));
                    vsag::Options::Instance().set_block_size_limit(size);
                    auto param = IVFTestIndex::GenerateIVFBuildParametersString(
                        metric_type, dim, base_quantization_str, 300, train_type);
                    auto index = IVFTestIndex::TestFactory(IVFTestIndex::name, param, true);
                    auto dataset = IVFTestIndex::pool.GetDatasetAndCreate(
                        dim, resource->base_count, metric_type);
                    IVFTestIndex::TestConcurrentAdd(index, dataset, true);
                    if (index->CheckFeature(vsag::SUPPORT_ADD_CONCURRENT)) {
                        IVFTestIndex::TestGeneral(index, dataset, search_param, recall);
                    }
                    vsag::Options::Instance().set_block_size_limit(origin_size);
                }
            }
        }
    }
}

TEST_CASE("[PR] IVF Concurrent Add", "[ft][ivf][pr]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(true);
    TestIVFConcurrentAdd(test_index, resource);
}

TEST_CASE("[Daily] IVF Concurrent Add", "[ft][ivf][daily]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(false);
    TestIVFConcurrentAdd(test_index, resource);
}

static void
TestIVFSerialize(const fixtures::IVFTestIndexPtr& test_index,
                 const fixtures::IVFResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto train_type : resource->train_types) {
                for (auto [base_quantization_str, recall] : resource->test_cases) {
                    if (train_type == "kmeans") {
                        recall *= 0.8F;  // Kmeans may not achieve high recall in random datasets
                    }
                    auto count = std::min(300, static_cast<int32_t>(dim / 4));
                    auto search_param =
                        fmt::format(fixtures::search_param_tmp, std::max(200, count));
                    INFO(
                        fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, "
                                    "train_type: {}, recall: {}",
                                    metric_type,
                                    dim,
                                    base_quantization_str,
                                    train_type,
                                    recall));
                    vsag::Options::Instance().set_block_size_limit(size);
                    auto param = IVFTestIndex::GenerateIVFBuildParametersString(
                        metric_type, dim, base_quantization_str, 300, train_type);
                    auto index = IVFTestIndex::TestFactory(IVFTestIndex::name, param, true);

                    if (index->CheckFeature(vsag::SUPPORT_BUILD)) {
                        auto dataset = IVFTestIndex::pool.GetDatasetAndCreate(
                            dim, resource->base_count, metric_type);
                        IVFTestIndex::TestBuildIndex(index, dataset, true);
                        if (index->CheckFeature(vsag::SUPPORT_SERIALIZE_FILE) and
                            index->CheckFeature(vsag::SUPPORT_DESERIALIZE_FILE)) {
                            auto index2 =
                                IVFTestIndex::TestFactory(IVFTestIndex::name, param, true);
                            IVFTestIndex::TestSerializeFile(
                                index, index2, dataset, search_param, true);
                        }
                        if (index->CheckFeature(vsag::SUPPORT_SERIALIZE_BINARY_SET) and
                            index->CheckFeature(vsag::SUPPORT_DESERIALIZE_BINARY_SET)) {
                            auto index2 =
                                IVFTestIndex::TestFactory(IVFTestIndex::name, param, true);
                            IVFTestIndex::TestSerializeBinarySet(
                                index, index2, dataset, search_param, true);
                        }
                        if (index->CheckFeature(vsag::SUPPORT_SERIALIZE_FILE) and
                            index->CheckFeature(vsag::SUPPORT_DESERIALIZE_READER_SET)) {
                            auto index2 =
                                IVFTestIndex::TestFactory(IVFTestIndex::name, param, true);
                            IVFTestIndex::TestSerializeReaderSet(
                                index, index2, dataset, search_param, IVFTestIndex::name, true);
                        }
                    }
                    vsag::Options::Instance().set_block_size_limit(origin_size);
                }
            }
        }
    }
}

TEST_CASE("[PR] IVF Serialize File", "[ft][ivf][serialization][pr]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(true);
    TestIVFSerialize(test_index, resource);
}

TEST_CASE("[Daily] IVF Serialize File", "[ft][ivf][serialization][daily]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(false);
    TestIVFSerialize(test_index, resource);
}

static void
TestIVFClone(const fixtures::IVFTestIndexPtr& test_index,
             const fixtures::IVFResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto train_type : resource->train_types) {
                for (auto [base_quantization_str, recall] : resource->test_cases) {
                    if (train_type == "kmeans") {
                        recall *= 0.8F;  // Kmeans may not achieve high recall in random datasets
                    }
                    auto count = std::min(300, static_cast<int32_t>(dim / 4));
                    auto search_param =
                        fmt::format(fixtures::search_param_tmp, std::max(200, count));
                    INFO(
                        fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, "
                                    "train_type: {}, recall: {}",
                                    metric_type,
                                    dim,
                                    base_quantization_str,
                                    train_type,
                                    recall));
                    vsag::Options::Instance().set_block_size_limit(size);
                    auto param = IVFTestIndex::GenerateIVFBuildParametersString(
                        metric_type, dim, base_quantization_str, 300, train_type);
                    auto index = IVFTestIndex::TestFactory(IVFTestIndex::name, param, true);

                    auto dataset = IVFTestIndex::pool.GetDatasetAndCreate(
                        dim, resource->base_count, metric_type);
                    IVFTestIndex::TestBuildIndex(index, dataset, true);
                    IVFTestIndex::TestClone(index, dataset, search_param);
                    vsag::Options::Instance().set_block_size_limit(origin_size);
                }
            }
        }
    }
}

TEST_CASE("[PR] IVF Clone", "[ft][ivf][pr]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(true);
    TestIVFClone(test_index, resource);
}

TEST_CASE("[Daily] IVF Clone", "[ft][ivf][daily]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(false);
    TestIVFClone(test_index, resource);
}

static void
TestIVFRandomAllocator(const fixtures::IVFTestIndexPtr& test_index,
                       const fixtures::IVFResourcePtr& resource) {
    auto allocator = std::make_shared<fixtures::RandomAllocator>();
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto train_type : resource->train_types) {
                for (auto [base_quantization_str, recall] : resource->test_cases) {
                    if (train_type == "kmeans") {
                        recall *= 0.8F;  // Kmeans may not achieve high recall in random datasets
                    }
                    auto count = std::min(300, static_cast<int32_t>(dim / 4));
                    auto search_param =
                        fmt::format(fixtures::search_param_tmp, std::max(200, count));
                    INFO(
                        fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, "
                                    "train_type: {}, recall: {}",
                                    metric_type,
                                    dim,
                                    base_quantization_str,
                                    train_type,
                                    recall));
                    vsag::Options::Instance().set_block_size_limit(size);
                    auto param = IVFTestIndex::GenerateIVFBuildParametersString(
                        metric_type, dim, base_quantization_str, 1);
                    auto index =
                        vsag::Factory::CreateIndex(IVFTestIndex::name, param, allocator.get());
                    if (not index.has_value()) {
                        continue;
                    }
                    auto dataset = IVFTestIndex::pool.GetDatasetAndCreate(
                        dim, resource->base_count, metric_type);
                    IVFTestIndex::TestContinueAddIgnoreRequire(index.value(), dataset);
                    vsag::Options::Instance().set_block_size_limit(origin_size);
                }
            }
        }
    }
}

TEST_CASE("[PR] IVF Build & ContinueAdd Test With Random Allocator", "[ft][ivf][pr]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(true);
    TestIVFRandomAllocator(test_index, resource);
}

TEST_CASE("[Daily] IVF Build & ContinueAdd Test With Random Allocator", "[ft][ivf][daily]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(false);
    TestIVFRandomAllocator(test_index, resource);
}

static void
TestIVFEstimateMemory(const fixtures::IVFTestIndexPtr& test_index,
                      const fixtures::IVFResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    uint64_t estimate_count = 1000;

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto train_type : resource->train_types) {
                for (auto [base_quantization_str, recall] : resource->test_cases) {
                    if (train_type == "kmeans") {
                        recall *= 0.8F;  // Kmeans may not achieve high recall in random datasets
                    }
                    auto count = std::min(300, static_cast<int32_t>(dim / 4));
                    auto search_param =
                        fmt::format(fixtures::search_param_tmp, std::max(200, count));
                    INFO(
                        fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, "
                                    "train_type: {}, recall: {}",
                                    metric_type,
                                    dim,
                                    base_quantization_str,
                                    train_type,
                                    recall));
                    vsag::Options::Instance().set_block_size_limit(size);
                    auto param = IVFTestIndex::GenerateIVFBuildParametersString(
                        metric_type, dim, base_quantization_str, 300, train_type);
                    auto dataset =
                        IVFTestIndex::pool.GetDatasetAndCreate(dim, estimate_count, metric_type);
                    IVFTestIndex::TestEstimateMemory(IVFTestIndex::name, param, dataset);
                    vsag::Options::Instance().set_block_size_limit(origin_size);
                }
            }
        }
    }
}

TEST_CASE("[PR] IVF Estimate Memory", "[ft][ivf][pr]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(true);
    TestIVFEstimateMemory(test_index, resource);
}

TEST_CASE("[Daily] IVF Estimate Memory", "[ft][ivf][daily]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(false);
    TestIVFEstimateMemory(test_index, resource);
}

static void
TestIVFBuildMultiBucketsPerData(const fixtures::IVFTestIndexPtr& test_index,
                                const fixtures::IVFResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto train_type : resource->train_types) {
                for (auto [base_quantization_str, recall] : resource->test_cases) {
                    if (train_type == "kmeans") {
                        recall *= 0.8F;  // Kmeans may not achieve high recall in random datasets
                    }
                    auto count = std::min(300, static_cast<int32_t>(dim / 4));
                    auto search_param =
                        fmt::format(fixtures::search_param_tmp, std::max(200, count));
                    INFO(
                        fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, "
                                    "train_type: {}, recall: {}",
                                    metric_type,
                                    dim,
                                    base_quantization_str,
                                    train_type,
                                    recall));
                    vsag::Options::Instance().set_block_size_limit(size);
                    auto param = IVFTestIndex::GenerateIVFBuildParametersString(
                        metric_type, dim, base_quantization_str, 300, train_type, false, 2);
                    auto index = IVFTestIndex::TestFactory(IVFTestIndex::name, param, true);
                    auto dataset = IVFTestIndex::pool.GetDatasetAndCreate(
                        dim, resource->base_count, metric_type);
                    IVFTestIndex::TestBuildIndex(index, dataset, true);
                    if (index->CheckFeature(vsag::SUPPORT_BUILD)) {
                        IVFTestIndex::TestGeneral(index, dataset, search_param, recall);
                    }
                    vsag::Options::Instance().set_block_size_limit(origin_size);
                }
            }
        }
    }
}

TEST_CASE("[PR] IVF Build Multi Buckets Per Data", "[ft][ivf][pr]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(true);
    TestIVFBuildMultiBucketsPerData(test_index, resource);
}

TEST_CASE("[Daily] IVF Build Multi Buckets Per Data", "[ft][ivf][daily]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(false);
    TestIVFBuildMultiBucketsPerData(test_index, resource);
}

static void
TestIVFGNOIMIBuild(const fixtures::IVFTestIndexPtr& test_index,
                   const fixtures::IVFResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto train_type : resource->train_types) {
                for (auto [base_quantization_str, recall] : resource->test_cases) {
                    auto count = std::min(400, static_cast<int32_t>(dim / 4));
                    auto search_param =
                        fmt::format(fixtures::search_param_tmp, std::max(350, count));
                    INFO(
                        fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, "
                                    "train_type: {}, recall: {}",
                                    metric_type,
                                    dim,
                                    base_quantization_str,
                                    train_type,
                                    recall));
                    vsag::Options::Instance().set_block_size_limit(size);
                    auto param = IVFTestIndex::GenerateGNOIMIBuildParametersString(
                        metric_type, dim, base_quantization_str, 20, 20, train_type, false, 1);
                    auto index = IVFTestIndex::TestFactory(IVFTestIndex::name, param, true);
                    auto dataset = IVFTestIndex::pool.GetDatasetAndCreate(
                        dim, resource->base_count, metric_type);
                    IVFTestIndex::TestBuildIndex(index, dataset, true);
                    if (index->CheckFeature(vsag::SUPPORT_BUILD)) {
                        IVFTestIndex::TestGeneral(index, dataset, search_param, recall);
                    }
                    vsag::Options::Instance().set_block_size_limit(origin_size);
                }
            }
        }
    }
}

TEST_CASE("[PR] IVF GNO-IMI Build", "[ft][ivf][pr]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(true);
    TestIVFGNOIMIBuild(test_index, resource);
}

TEST_CASE("[Daily] IVF GNO-IMI Build", "[ft][ivf][daily]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(false);
    TestIVFGNOIMIBuild(test_index, resource);
}

static void
TestIVFGNOIMIBuildWithResidual(const fixtures::IVFTestIndexPtr& test_index,
                               const fixtures::IVFResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto train_type : resource->train_types) {
                for (auto [base_quantization_str, recall] : resource->test_cases) {
                    if (train_type == "kmeans") {
                        recall *= 0.8F;  // Kmeans may not achieve high recall in random datasets
                    }
                    if (base_quantization_str == "sq8_uniform,fp32") {
                        continue;  // sq8_uniform reduce recall when using residual in GNO-IMI
                    }
                    auto count = std::min(400, static_cast<int32_t>(dim / 4));
                    auto search_param =
                        fmt::format(fixtures::search_param_tmp, std::max(400, count));
                    INFO(
                        fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, "
                                    "train_type: {}, recall: {}",
                                    metric_type,
                                    dim,
                                    base_quantization_str,
                                    train_type,
                                    recall));
                    vsag::Options::Instance().set_block_size_limit(size);
                    auto param = IVFTestIndex::GenerateGNOIMIBuildParametersString(
                        metric_type, dim, base_quantization_str, 20, 20, train_type, true, 1);
                    auto index = IVFTestIndex::TestFactory(IVFTestIndex::name, param, true);
                    auto dataset = IVFTestIndex::pool.GetDatasetAndCreate(
                        dim, resource->base_count, metric_type);
                    IVFTestIndex::TestBuildIndex(index, dataset, true);
                    if (index->CheckFeature(vsag::SUPPORT_BUILD)) {
                        IVFTestIndex::TestGeneral(index, dataset, search_param, recall);
                    }
                    vsag::Options::Instance().set_block_size_limit(origin_size);
                }
            }
        }
    }
}

TEST_CASE("[PR] IVF GNO-IMI Build with Residual", "[ft][ivf][pr]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(true);
    TestIVFGNOIMIBuildWithResidual(test_index, resource);
}

TEST_CASE("[Daily] IVF GNO-IMI Build with Residual", "[ft][ivf][daily]") {
    auto test_index = std::make_shared<fixtures::IVFTestIndex>();
    auto resource = test_index->GetResource(false);
    TestIVFGNOIMIBuildWithResidual(test_index, resource);
}
