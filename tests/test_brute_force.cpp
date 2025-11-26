
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

#include "fixtures/fixtures.h"
#include "fixtures/test_dataset_pool.h"
#include "test_index.h"
#include "vsag/options.h"

namespace fixtures {

class BruteForceTestResource {
public:
    std::vector<int> dims;
    std::vector<std::pair<std::string, float>> test_cases;
    std::vector<std::string> metric_types;
    std::vector<std::string> train_types;
    uint64_t base_count;
};
using BruteForceResourcePtr = std::shared_ptr<BruteForceTestResource>;

class BruteForceTestIndex : public fixtures::TestIndex {
public:
    static std::string
    GenerateBruteForceBuildParametersString(const std::string& metric_type,
                                            int64_t dim,
                                            const std::string& quantization_str = "sq8",
                                            bool use_attr_filter = false);

    static BruteForceResourcePtr
    GetResource(bool sample = true);

    static void
    TestGeneral(const IndexPtr& index,
                const TestDatasetPtr& dataset,
                const std::string& search_param,
                float recall);

    static TestDatasetPool pool;

    static fixtures::TempDir dir;

    static const std::string name;

    constexpr static uint64_t base_count = 1000;

    static const std::vector<std::pair<std::string, float>> all_test_cases;
};

TestDatasetPool BruteForceTestIndex::pool{};
fixtures::TempDir BruteForceTestIndex::dir{"BruteForce_test"};
const std::string BruteForceTestIndex::name = "brute_force";
const std::vector<std::pair<std::string, float>> BruteForceTestIndex::all_test_cases = {
    {"sq8", 0.90},
    {"fp32", 0.99},
    {"sq8_uniform", 0.90},
    {"bf16", 0.92},
    {"fp16", 0.92},
};

constexpr static const char* search_param_tmp = "";

BruteForceResourcePtr
BruteForceTestIndex::GetResource(bool sample) {
    auto resource = std::make_shared<BruteForceTestResource>();
    if (sample) {
        resource->dims = fixtures::get_common_used_dims(1, RandomValue(0, 999));
        resource->test_cases = fixtures::RandomSelect(BruteForceTestIndex::all_test_cases, 3);
        resource->metric_types = fixtures::RandomSelect<std::string>({"ip", "l2", "cosine"}, 1);
        resource->base_count = BruteForceTestIndex::base_count;
    } else {
        resource->dims = fixtures::get_index_test_dims(3, RandomValue(0, 999));
        resource->test_cases = BruteForceTestIndex::all_test_cases;
        resource->metric_types = fixtures::RandomSelect<std::string>({"ip", "l2", "cosine"}, 2);
        resource->base_count = BruteForceTestIndex::base_count * 3;
    }
    return resource;
}

std::string
BruteForceTestIndex::GenerateBruteForceBuildParametersString(const std::string& metric_type,
                                                             int64_t dim,
                                                             const std::string& quantization_str,
                                                             bool use_attr_filter) {
    std::string build_parameters_str;

    constexpr auto parameter_temp = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "base_quantization_type": "{}",
            "store_raw_vector": true,
            "use_attribute_filter": {}
        }}
    }}
    )";

    build_parameters_str =
        fmt::format(parameter_temp, metric_type, dim, quantization_str, use_attr_filter);

    return build_parameters_str;
}

void
BruteForceTestIndex::TestGeneral(const IndexPtr& index,
                                 const TestDatasetPtr& dataset,
                                 const std::string& search_param,
                                 float recall) {
    REQUIRE(index->GetIndexType() == vsag::IndexType::BRUTEFORCE);
    TestKnnSearch(index, dataset, search_param, recall, true);
    TestConcurrentKnnSearch(index, dataset, search_param, recall, true);
    TestRangeSearch(index, dataset, search_param, recall, 10, true);
    TestRangeSearch(index, dataset, search_param, recall / 2.0, 5, true);
    TestFilterSearch(index, dataset, search_param, recall, true);
    TestGetRawVectorByIds(index, dataset, true);
    TestCheckIdExist(index, dataset);
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

static void
TestBruteForceBuildAndContinueAdd(const fixtures::BruteForceResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = 1024 * 1024 * 2;

    vsag::Options::Instance().set_block_size_limit(size);
    for (const auto& metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (const auto& [base_quantization_str, recall] : resource->test_cases) {
                auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString(
                    metric_type, dim, base_quantization_str);
                auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
                auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(
                    dim, BruteForceTestIndex::base_count, metric_type);
                TestIndex::TestContinueAdd(index, dataset, true);
                BruteForceTestIndex::TestGeneral(index, dataset, search_param_tmp, recall);
            }
        }
    }
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE("(PR) BruteForce Build & ContinueAdd Test", "[ft][BruteForce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceBuildAndContinueAdd(resource);
}

TEST_CASE("(Daily) BruteForce Build & ContinueAdd Test", "[ft][BruteForce][daily]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(false);
    TestBruteForceBuildAndContinueAdd(resource);
}

static void
TestBruteForceBuild(const fixtures::BruteForceResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    std::vector<int32_t> search_threads_counts{1, 3};
    constexpr static const char* search_param_tmp2 = R"(
    {{
        "parallelism": {}
    }})";
    for (const auto& metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (const auto& [base_quantization_str, recall] : resource->test_cases) {
                vsag::Options::Instance().set_block_size_limit(size);
                auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString(
                    metric_type, dim, base_quantization_str);
                auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
                auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(
                    dim, BruteForceTestIndex::base_count, metric_type);
                TestIndex::TestBuildIndex(index, dataset, true);
                for (auto search_thread_count : search_threads_counts) {
                    auto search_param = fmt::format(search_param_tmp2, search_thread_count);
                    BruteForceTestIndex::TestGeneral(index, dataset, search_param, recall);
                }
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("(PR) BruteForce Build Test", "[ft][BruteForce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceBuild(resource);
}

TEST_CASE("(Daily) BruteForce Build Test", "[ft][BruteForce][daily]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(false);
    TestBruteForceBuild(resource);
}

static void
TestBruteForceAdd(const fixtures::BruteForceResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = 1024 * 1024 * 2;
    for (const auto& metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (const auto& [base_quantization_str, recall] : resource->test_cases) {
                vsag::Options::Instance().set_block_size_limit(size);
                auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString(
                    metric_type, dim, base_quantization_str);
                auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
                auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(
                    dim, BruteForceTestIndex::base_count, metric_type);
                TestIndex::TestAddIndex(index, dataset, true);
                if (index->CheckFeature(vsag::SUPPORT_ADD_FROM_EMPTY)) {
                    BruteForceTestIndex::TestGeneral(index, dataset, search_param_tmp, recall);
                }
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("(PR) BruteForce Add Test", "[ft][BruteForce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceAdd(resource);
}

TEST_CASE("(Daily) BruteForce Add Test", "[ft][BruteForce][daily]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(false);
    TestBruteForceAdd(resource);
}

static void
TestBruteForceConcurrentAdd(const fixtures::BruteForceResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = 1024 * 1024 * 2;
    for (const auto& metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (const auto& [base_quantization_str, recall] : resource->test_cases) {
                vsag::Options::Instance().set_block_size_limit(size);
                auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString(
                    metric_type, dim, base_quantization_str);
                auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
                auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(
                    dim, BruteForceTestIndex::base_count, metric_type);
                TestIndex::TestConcurrentAdd(index, dataset, true);
                if (index->CheckFeature(vsag::SUPPORT_ADD_CONCURRENT)) {
                    BruteForceTestIndex::TestGeneral(index, dataset, search_param_tmp, recall);
                }
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("(PR) BruteForce Concurrent Add Test", "[ft][BruteForce][pr][concurrent]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceConcurrentAdd(resource);
}

TEST_CASE("(Daily) BruteForce Concurrent Add Test", "[ft][BruteForce][daily][concurrent]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(false);
    TestBruteForceConcurrentAdd(resource);
}

static void
TestBruteForceSerializeFile(const fixtures::BruteForceResourcePtr& resource) {
    using namespace fixtures;

    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = 1024 * 1024 * 2;
    for (const auto& metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (const auto& [base_quantization_str, recall] : resource->test_cases) {
                vsag::Options::Instance().set_block_size_limit(size);
                auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString(
                    metric_type, dim, base_quantization_str);
                auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
                auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(
                    dim, BruteForceTestIndex::base_count, metric_type);
                TestIndex::TestBuildIndex(index, dataset, true);
                auto index2 = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
                TestIndex::TestSerializeFile(index, index2, dataset, search_param_tmp, true);
                index2 = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
                TestIndex::TestSerializeBinarySet(index, index2, dataset, search_param_tmp, true);
                index2 = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
                TestIndex::TestSerializeReaderSet(
                    index, index2, dataset, search_param_tmp, BruteForceTestIndex::name, true);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("(PR) BruteForce Serialize File Test", "[ft][BruteForce][pr][serialization]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceSerializeFile(resource);
}

TEST_CASE("(Daily) BruteForce Serialize File Test", "[ft][BruteForce][daily][serialization]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(false);
    TestBruteForceSerializeFile(resource);
}

static void
TestBruteForceClone(const fixtures::BruteForceResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = 1024 * 1024 * 2;
    for (const auto& metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (const auto& [base_quantization_str, recall] : resource->test_cases) {
                vsag::Options::Instance().set_block_size_limit(size);
                auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString(
                    metric_type, dim, base_quantization_str);
                auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
                auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(
                    dim, BruteForceTestIndex::base_count, metric_type);
                TestIndex::TestBuildIndex(index, dataset, true);
                auto index2 = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
                TestIndex::TestClone(index, dataset, search_param_tmp);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("(PR) BruteForce Clone Test", "[ft][BruteForce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceClone(resource);
}

TEST_CASE("(Daily) BruteForce Clone Test", "[ft][BruteForce][daily]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(false);
    TestBruteForceClone(resource);
}

static void
TestBruteForceRandomAllocator(const fixtures::BruteForceResourcePtr& resource) {
    using namespace fixtures;
    auto allocator = std::make_shared<fixtures::RandomAllocator>();
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = 1024 * 1024 * 2;
    for (const auto& metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                vsag::Options::Instance().set_block_size_limit(size);
                auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString(
                    metric_type, dim, base_quantization_str);
                auto index =
                    vsag::Factory::CreateIndex(BruteForceTestIndex::name, param, allocator.get());
                if (not index.has_value()) {
                    continue;
                }
                auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(
                    dim, BruteForceTestIndex::base_count, metric_type);
                TestIndex::TestContinueAddIgnoreRequire(index.value(), dataset);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("(PR) BruteForce Build & ContinueAdd Test With Random Allocator",
          "[ft][BruteForce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceRandomAllocator(resource);
}

TEST_CASE("(Daily) BruteForce Build & ContinueAdd Test With Random Allocator",
          "[ft][BruteForce][daily]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(false);
    TestBruteForceRandomAllocator(resource);
}

static void
TestBruteForceCalcDistanceById(const fixtures::BruteForceResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = 1024 * 1024 * 2;

    for (const auto& metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            auto base_quantization_str = "fp32";
            vsag::Options::Instance().set_block_size_limit(size);
            auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString(
                metric_type, dim, base_quantization_str);
            auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(
                dim, BruteForceTestIndex::base_count, metric_type);
            auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
            TestIndex::TestBuildIndex(index, dataset, true);
            TestIndex::TestCalcDistanceById(index, dataset);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE("(PR) BruteForce GetDistance By ID Test", "[ft][BruteForce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceCalcDistanceById(resource);
}

TEST_CASE("(Daily) BruteForce GetDistance By ID Test", "[ft][BruteForce][daily]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(false);
    TestBruteForceCalcDistanceById(resource);
}

static void
TestBruteForceDuplicateBuild(const fixtures::BruteForceResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = 1024 * 1024 * 2;
    for (const auto& metric_type : resource->metric_types) {
        for (auto& dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                vsag::Options::Instance().set_block_size_limit(size);
                auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString(
                    metric_type, dim, base_quantization_str);
                auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
                auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(
                    dim, BruteForceTestIndex::base_count, metric_type);
                TestIndex::TestDuplicateAdd(index, dataset);
                BruteForceTestIndex::TestGeneral(index, dataset, search_param_tmp, recall);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("(PR) BruteForce Duplicate Build Test", "[ft][BruteForce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceDuplicateBuild(resource);
}

TEST_CASE("(Daily) BruteForce Duplicate Build Test", "[ft][BruteForce][daily]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(false);
    TestBruteForceDuplicateBuild(resource);
}

static void
TestBruteForceWithAttrFilter(const fixtures::BruteForceResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = 1024 * 1024 * 2;

    for (const auto& metric_type : resource->metric_types) {
        for (auto& dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                vsag::Options::Instance().set_block_size_limit(size);
                auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString(
                    metric_type, dim, base_quantization_str, true);
                auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
                auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(
                    dim, BruteForceTestIndex::base_count, metric_type);
                TestIndex::TestBuildIndex(index, dataset, true);
                TestIndex::TestWithAttr(index, dataset, search_param_tmp, false);
                auto index2 = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);

                REQUIRE_NOTHROW(test_serializion_file(*index, *index2, "serialize_bruteforce"));
                TestIndex::TestWithAttr(index2, dataset, search_param_tmp, true);

                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("(PR) BruteForce With Attribute Filter Test", "[ft][BruteForce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceWithAttrFilter(resource);
}

TEST_CASE("(Daily) BruteForce With Attribute Filter Test", "[ft][BruteForce][daily]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(false);
    TestBruteForceWithAttrFilter(resource);
}

static void
TestBruteForceRemoveById(const fixtures::BruteForceResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = 1024 * 1024 * 2;
    auto metric_type = "l2";

    for (auto& dim : resource->dims) {
        auto base_quantization_str = "fp32";
        auto recall = 0.99;
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            metric_type, dim, base_quantization_str);
        auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
        auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(
            dim, BruteForceTestIndex::base_count, metric_type);
        TestIndex::TestContinueAdd(index, dataset, true);
        BruteForceTestIndex::TestGeneral(index, dataset, search_param_tmp, recall);
        for (int i = 0; i < BruteForceTestIndex::base_count; ++i) {
            auto res = index->Remove(dataset->base_->GetIds()[i]);
            auto check_exist = index->CheckIdExist(dataset->base_->GetIds()[i]);
            REQUIRE(res.has_value());
            REQUIRE(res.value());
            REQUIRE(not check_exist);
            auto num = index->GetNumElements();
            REQUIRE(num == BruteForceTestIndex::base_count - i - 1);
        }
        vsag::Options::Instance().set_block_size_limit(origin_size);
    }
}

TEST_CASE("(PR) BruteForce Remove By ID Test", "[ft][BruteForce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceRemoveById(resource);
}

TEST_CASE("(Daily) BruteForce Remove By ID Test", "[ft][BruteForce][daily]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(false);
    TestBruteForceRemoveById(resource);
}

static void
TestBruteForceEstimateMemory(const fixtures::BruteForceResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = 1024 * 1024 * 2;
    uint64_t estimate_count = 1000;
    int64_t dim = 1536;
    for (const auto& metric_type : resource->metric_types) {
        for (auto& [base_quantization_str, recall] : resource->test_cases) {
            vsag::Options::Instance().set_block_size_limit(size);
            auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString(
                metric_type, dim, base_quantization_str);
            auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
            auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(
                dim, BruteForceTestIndex::base_count, metric_type);
            auto val = index->EstimateMemory(1000);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

TEST_CASE("(PR) BruteForce BruteForce Estimate Memory Test", "[ft][BruteForce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceEstimateMemory(resource);
}

TEST_CASE("(Daily) BruteForce BruteForce Estimate Memory Test", "[ft][BruteForce][daily]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(false);
    TestBruteForceEstimateMemory(resource);
}
