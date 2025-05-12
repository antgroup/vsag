
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
#include "test_index.h"

namespace fixtures {
class IVFTestIndex : public fixtures::TestIndex {
public:
    static std::string
    GenerateIVFBuildParametersString(const std::string& metric_type,
                                     int64_t dim,
                                     const std::string& quantization_str = "sq8",
                                     int buckets_count = 210,
                                     const std::string& train_type = "kmeans");
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
            "ivf": {{
                "scan_buckets_count": {}
            }}
        }})";

    // DON'T WORRY! IVF just can't achieve high recall on random datasets. so we set the expected
    // recall with a small number in test cases
    const std::vector<std::pair<std::string, float>> test_cases = {
        {"fp32", 0.90},
        {"bf16", 0.88},
        {"fp16", 0.88},
        {"sq8", 0.84},
        {"sq8_uniform", 0.83},
        {"sq8_uniform,fp32", 0.89},
        {"pq,fp32", 0.82},
        {"pqfs,fp32", 0.82},
    };
};

TestDatasetPool IVFTestIndex::pool{};
std::vector<int> IVFTestIndex::dims = fixtures::get_common_used_dims(2, RandomValue(0, 999));
fixtures::TempDir IVFTestIndex::dir{"hgraph_test"};

std::string
IVFTestIndex::GenerateIVFBuildParametersString(const std::string& metric_type,
                                               int64_t dim,
                                               const std::string& quantization_str,
                                               int buckets_count,
                                               const std::string& train_type) {
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
            "precise_quantization_type": "{}"
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
                                       precise_quantizer_str);

    INFO(build_parameters_str);
    return build_parameters_str;
}
void
IVFTestIndex::TestGeneral(const TestIndex::IndexPtr& index,
                          const TestDatasetPtr& dataset,
                          const std::string& search_param,
                          float recall) {
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

TEST_CASE_PERSISTENT_FIXTURE(fixtures::IVFTestIndex, "IVF Build & ContinueAdd Test", "[ft][ivf]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    std::string train_type = GENERATE("random", "kmeans");

    const std::string name = "ivf";
    auto search_param = fmt::format(search_param_tmp, 200);
    for (auto& dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            vsag::Options::Instance().set_block_size_limit(size);
            auto param = GenerateIVFBuildParametersString(
                metric_type, dim, base_quantization_str, 300, train_type);
            auto index = TestFactory(name, param, true);
            auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
            TestContinueAdd(index, dataset, true);
            if (index->CheckFeature(vsag::SUPPORT_ADD_AFTER_BUILD)) {
                TestGeneral(index, dataset, search_param, recall);
            }
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::IVFTestIndex, "IVF Build", "[ft][ivf]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    std::string train_type = GENERATE("random", "kmeans");

    const std::string name = "ivf";
    auto search_param = fmt::format(search_param_tmp, 200);
    for (auto& dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            vsag::Options::Instance().set_block_size_limit(size);
            auto param = GenerateIVFBuildParametersString(
                metric_type, dim, base_quantization_str, 300, train_type);
            auto index = TestFactory(name, param, true);
            auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
            TestBuildIndex(index, dataset, true);
            if (index->CheckFeature(vsag::SUPPORT_BUILD)) {
                TestGeneral(index, dataset, search_param, recall);
            }
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::IVFTestIndex, "IVF Export Model", "[ft][ivf]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    std::string train_type = GENERATE("kmeans");

    const std::string name = "ivf";
    auto search_param = fmt::format(search_param_tmp, 200);
    for (auto& dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            vsag::Options::Instance().set_block_size_limit(size);
            auto param = GenerateIVFBuildParametersString(
                metric_type, dim, base_quantization_str, 300, train_type);
            auto index = TestFactory(name, param, true);
            auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);

            TestBuildIndex(index, dataset, true);
            TestExportModel(index, dataset, search_param);

            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::IVFTestIndex, "IVF Add", "[ft][ivf]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    std::string train_type = GENERATE("kmeans");

    const std::string name = "ivf";
    auto search_param = fmt::format(search_param_tmp, 200);
    for (auto& dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            vsag::Options::Instance().set_block_size_limit(size);
            auto param = GenerateIVFBuildParametersString(
                metric_type, dim, base_quantization_str, 300, train_type);
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

TEST_CASE_PERSISTENT_FIXTURE(fixtures::IVFTestIndex, "IVF Merge", "[ft][ivf]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "cosine");
    std::string train_type = GENERATE("kmeans");

    const std::string name = "ivf";
    auto search_param = fmt::format(search_param_tmp, 200);
    for (auto& dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            vsag::Options::Instance().set_block_size_limit(size);
            auto param = GenerateIVFBuildParametersString(
                metric_type, dim, base_quantization_str, 300, train_type);
            auto model = TestFactory(name, param, true);
            auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
            auto ret = model->Train(dataset->base_);
            REQUIRE(ret.has_value() == true);
            auto merge_index = TestMergeIndexWithSameModel(model, dataset, 5, true);
            if (model->CheckFeature(vsag::SUPPORT_MERGE_INDEX)) {
                TestGeneral(merge_index, dataset, search_param, recall);
            }
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::IVFTestIndex,
                             "IVF Concurrent Add",
                             "[ft][ivf][concurrent]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    std::string train_type = GENERATE("random", "kmeans");

    const std::string name = "ivf";
    auto search_param = fmt::format(search_param_tmp, 200);
    for (auto& dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            vsag::Options::Instance().set_block_size_limit(size);
            auto param = GenerateIVFBuildParametersString(
                metric_type, dim, base_quantization_str, 300, train_type);
            auto index = TestFactory(name, param, true);
            auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
            TestConcurrentAdd(index, dataset, true);
            if (index->CheckFeature(vsag::SUPPORT_ADD_CONCURRENT)) {
                TestGeneral(index, dataset, search_param, recall);
            }
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::IVFTestIndex, "IVF Serialize File", "[ft][ivf]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "ivf";
    auto search_param = fmt::format(search_param_tmp, 200);
    std::string train_type = GENERATE("kmeans");

    for (auto& dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            vsag::Options::Instance().set_block_size_limit(size);
            auto param = GenerateIVFBuildParametersString(
                metric_type, dim, base_quantization_str, 300, train_type);
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

TEST_CASE_PERSISTENT_FIXTURE(fixtures::IVFTestIndex, "IVF Clone", "[ft][ivf]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "ivf";
    auto search_param = fmt::format(search_param_tmp, 200);
    std::string train_type = GENERATE("kmeans");

    for (auto& dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            vsag::Options::Instance().set_block_size_limit(size);
            auto param = GenerateIVFBuildParametersString(
                metric_type, dim, base_quantization_str, 300, train_type);
            auto index = TestFactory(name, param, true);

            auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
            TestBuildIndex(index, dataset, true);
            TestClone(index, dataset, search_param);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::IVFTestIndex,
                             "IVF Build & ContinueAdd Test With Random Allocator",
                             "[ft][ivf]") {
    auto allocator = std::make_shared<fixtures::RandomAllocator>();
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    std::string train_type = GENERATE("random", "kmeans");
    const std::string name = "ivf";
    for (auto& dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            vsag::Options::Instance().set_block_size_limit(size);
            auto param =
                GenerateIVFBuildParametersString(metric_type, dim, base_quantization_str, 1);
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

TEST_CASE_PERSISTENT_FIXTURE(fixtures::IVFTestIndex, "IVF Estimate Memory", "[ft][ivf]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    std::string train_type = GENERATE("random", "kmeans");

    const std::string name = "ivf";
    auto search_param = fmt::format(search_param_tmp, 200);
    uint64_t estimate_count = 1000;
    for (auto& dim : dims) {
        for (auto& [base_quantization_str, recall] : test_cases) {
            vsag::Options::Instance().set_block_size_limit(size);
            auto param = GenerateIVFBuildParametersString(
                metric_type, dim, base_quantization_str, 300, train_type);
            auto dataset = pool.GetDatasetAndCreate(dim, estimate_count, metric_type);
            TestEstimateMemory(name, param, dataset);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}
