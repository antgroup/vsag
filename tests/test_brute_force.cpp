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
#include "vsag/options.h"

namespace fixtures {

class EvenIdFilter : public vsag::Filter {
public:
    bool
    CheckValid(int64_t id) const override {
        return id % 2 == 0;
    }
};

static void
CheckSameRangeSearchResults(const vsag::DatasetPtr& lhs, const vsag::DatasetPtr& rhs) {
    REQUIRE(lhs->GetDim() == rhs->GetDim());
    for (int64_t i = 0; i < lhs->GetDim(); ++i) {
        REQUIRE(lhs->GetIds()[i] == rhs->GetIds()[i]);
        REQUIRE(lhs->GetDistances()[i] == rhs->GetDistances()[i]);
    }
}

class BruteForceTestIndex : public fixtures::TestIndex {
public:
    static std::string
    GenerateBruteForceBuildParametersString(const std::string& metric_type,
                                            int64_t dim,
                                            const std::string& quantization_str = "sq8",
                                            bool use_attr_filter = false);

    static TestMatrix
    GetMatrix(bool sample = true);

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

TestMatrix
BruteForceTestIndex::GetMatrix(bool sample) {
    TestMatrix matrix;
    matrix.base_count = BruteForceTestIndex::base_count;
    if (sample) {
        matrix.dims = fixtures::get_common_used_dims(1, RandomValue(0, 999));
        matrix.quantizers = fixtures::RandomSelect(BruteForceTestIndex::all_test_cases, 3);
        matrix.metrics = fixtures::RandomSelect<std::string>({"ip", "l2", "cosine"}, 1);
    } else {
        matrix.dims = fixtures::get_index_test_dims(3, RandomValue(0, 999));
        matrix.quantizers = BruteForceTestIndex::all_test_cases;
        matrix.metrics = fixtures::RandomSelect<std::string>({"ip", "l2", "cosine"}, 2);
        matrix.base_count = BruteForceTestIndex::base_count * 3;
    }
    return matrix;
}

std::string
BruteForceTestIndex::GenerateBruteForceBuildParametersString(const std::string& metric_type,
                                                             int64_t dim,
                                                             const std::string& quantization_str,
                                                             bool use_attr_filter) {
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
    return fmt::format(parameter_temp, metric_type, dim, quantization_str, use_attr_filter);
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
                             "[ft][factory][bruteforce]") {
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

TEST_CASE("(PR) BruteForce Build & ContinueAdd Test", "[ft][build][bruteforce][pr]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(true);
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestContinueAdd(index, dataset, true);
        fixtures::BruteForceTestIndex::TestGeneral(
            index, dataset, fixtures::search_param_tmp, iter.recall);
    });
}

TEST_CASE("(Daily) BruteForce Build & ContinueAdd Test", "[ft][build][bruteforce][daily]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(false);
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestContinueAdd(index, dataset, true);
        fixtures::BruteForceTestIndex::TestGeneral(
            index, dataset, fixtures::search_param_tmp, iter.recall);
    });
}

TEST_CASE("(PR) BruteForce Build Test", "[ft][build][bruteforce][pr]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(true);
    auto size = GENERATE(1024 * 1024 * 2);
    std::vector<int32_t> search_threads_counts{1, 3};
    constexpr static const char* search_param_tmp2 = R"({{ "parallelism": {} }})";
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestBuildIndex(index, dataset, true);
        for (auto tc : search_threads_counts) {
            auto sp = fmt::format(search_param_tmp2, tc);
            fixtures::BruteForceTestIndex::TestGeneral(index, dataset, sp, iter.recall);
        }
    });
}

TEST_CASE("(PR) BruteForce Parallel RangeSearch Test", "[ft][range_search][bruteforce][pr]") {
    constexpr int64_t dim = 2;
    constexpr int64_t base_count = 8;
    std::vector<int64_t> ids{0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> vectors{0.0F,
                               0.0F,
                               1.0F,
                               0.0F,
                               2.0F,
                               0.0F,
                               3.0F,
                               0.0F,
                               4.0F,
                               0.0F,
                               5.0F,
                               0.0F,
                               6.0F,
                               0.0F,
                               7.0F,
                               0.0F};
    std::vector<float> query_vector{0.0F, 0.0F};

    auto base = vsag::Dataset::Make();
    base->NumElements(base_count)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector.data())->Owner(false);

    auto param =
        fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString("l2", dim, "fp32");
    auto index = fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
    REQUIRE(index->Build(base).has_value());

    auto single = index->RangeSearch(query, 16.0F, "{}", 4).value();
    auto parallel = index->RangeSearch(query, 16.0F, R"({"parallelism": 4})", 4).value();
    fixtures::CheckSameRangeSearchResults(single, parallel);
    REQUIRE_FALSE(index->RangeSearch(query, 16.0F, R"({"parallelism": 4})", 0).has_value());

    auto excessive = index->RangeSearch(query, 16.0F, R"({"parallelism": 32})", 4).value();
    fixtures::CheckSameRangeSearchResults(single, excessive);

    auto filter = std::make_shared<fixtures::EvenIdFilter>();
    auto fs = index->RangeSearch(query, 64.0F, "{}", filter, 3).value();
    auto fp = index->RangeSearch(query, 64.0F, R"({"parallelism": 4})", filter, 3).value();
    fixtures::CheckSameRangeSearchResults(fs, fp);
}

TEST_CASE("(Daily) BruteForce Build Test", "[ft][build][bruteforce][daily]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(false);
    auto size = GENERATE(1024 * 1024 * 2);
    std::vector<int32_t> search_threads_counts{1, 3};
    constexpr static const char* search_param_tmp2 = R"({{ "parallelism": {} }})";
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestBuildIndex(index, dataset, true);
        for (auto tc : search_threads_counts) {
            auto sp = fmt::format(search_param_tmp2, tc);
            fixtures::BruteForceTestIndex::TestGeneral(index, dataset, sp, iter.recall);
        }
    });
}

TEST_CASE("(PR) BruteForce Add Test", "[ft][build][bruteforce][pr]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(true);
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestAddIndex(index, dataset, true);
        if (index->CheckFeature(vsag::SUPPORT_ADD_FROM_EMPTY)) {
            fixtures::BruteForceTestIndex::TestGeneral(
                index, dataset, fixtures::search_param_tmp, iter.recall);
        }
    });
}

TEST_CASE("(Daily) BruteForce Add Test", "[ft][build][bruteforce][daily]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(false);
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestAddIndex(index, dataset, true);
        if (index->CheckFeature(vsag::SUPPORT_ADD_FROM_EMPTY)) {
            fixtures::BruteForceTestIndex::TestGeneral(
                index, dataset, fixtures::search_param_tmp, iter.recall);
        }
    });
}

TEST_CASE("(PR) BruteForce Concurrent Add Test", "[ft][build][bruteforce][concurrent][pr]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(true);
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestConcurrentAdd(index, dataset, true);
        if (index->CheckFeature(vsag::SUPPORT_ADD_CONCURRENT)) {
            fixtures::BruteForceTestIndex::TestGeneral(
                index, dataset, fixtures::search_param_tmp, iter.recall);
        }
    });
}

TEST_CASE("(Daily) BruteForce Concurrent Add Test", "[ft][build][bruteforce][concurrent][daily]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(false);
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestConcurrentAdd(index, dataset, true);
        if (index->CheckFeature(vsag::SUPPORT_ADD_CONCURRENT)) {
            fixtures::BruteForceTestIndex::TestGeneral(
                index, dataset, fixtures::search_param_tmp, iter.recall);
        }
    });
}

TEST_CASE("(PR) BruteForce Serialize File Test", "[ft][serialize][bruteforce][pr]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(true);
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestBuildIndex(index, dataset, true);
        auto index2 =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        fixtures::TestIndex::TestSerializeFile(
            index, index2, dataset, fixtures::search_param_tmp, true);
        index2 = fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        fixtures::TestIndex::TestSerializeBinarySet(
            index, index2, dataset, fixtures::search_param_tmp, true);
        index2 = fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        fixtures::TestIndex::TestSerializeReaderSet(index,
                                                    index2,
                                                    dataset,
                                                    fixtures::search_param_tmp,
                                                    fixtures::BruteForceTestIndex::name,
                                                    true);
    });
}

TEST_CASE("(Daily) BruteForce Serialize File Test", "[ft][serialize][bruteforce][daily]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(false);
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestBuildIndex(index, dataset, true);
        auto index2 =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        fixtures::TestIndex::TestSerializeFile(
            index, index2, dataset, fixtures::search_param_tmp, true);
        index2 = fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        fixtures::TestIndex::TestSerializeBinarySet(
            index, index2, dataset, fixtures::search_param_tmp, true);
        index2 = fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        fixtures::TestIndex::TestSerializeReaderSet(index,
                                                    index2,
                                                    dataset,
                                                    fixtures::search_param_tmp,
                                                    fixtures::BruteForceTestIndex::name,
                                                    true);
    });
}

TEST_CASE("(PR) BruteForce Clone Test", "[ft][clone][bruteforce][pr]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(true);
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestBuildIndex(index, dataset, true);
        auto index2 =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        fixtures::TestIndex::TestClone(index, dataset, fixtures::search_param_tmp);
    });
}

TEST_CASE("(Daily) BruteForce Clone Test", "[ft][clone][bruteforce][daily]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(false);
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestBuildIndex(index, dataset, true);
        auto index2 =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        fixtures::TestIndex::TestClone(index, dataset, fixtures::search_param_tmp);
    });
}

TEST_CASE("(PR) BruteForce Build & ContinueAdd Test With Random Allocator",
          "[ft][build][bruteforce][pr]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(true);
    auto allocator = std::make_shared<fixtures::RandomAllocator>();
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            vsag::Factory::CreateIndex(fixtures::BruteForceTestIndex::name, param, allocator.get());
        if (not index.has_value()) {
            return;
        }
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
    });
}

TEST_CASE("(Daily) BruteForce Build & ContinueAdd Test With Random Allocator",
          "[ft][build][bruteforce][daily]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(false);
    auto allocator = std::make_shared<fixtures::RandomAllocator>();
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            vsag::Factory::CreateIndex(fixtures::BruteForceTestIndex::name, param, allocator.get());
        if (not index.has_value()) {
            return;
        }
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
    });
}

TEST_CASE("(PR) BruteForce GetDistance By ID Test", "[ft][distance][bruteforce][pr]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(true);
    matrix.quantizers = {{"fp32", 0.99f}};
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestBuildIndex(index, dataset, true);
        fixtures::TestIndex::TestCalcDistanceById(index, dataset);
    });
}

TEST_CASE("(Daily) BruteForce GetDistance By ID Test", "[ft][distance][bruteforce][daily]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(false);
    matrix.quantizers = {{"fp32", 0.99f}};
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestBuildIndex(index, dataset, true);
        fixtures::TestIndex::TestCalcDistanceById(index, dataset);
    });
}

TEST_CASE("(PR) BruteForce Duplicate Build Test", "[ft][build][duplicate][bruteforce][pr]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(true);
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestDuplicateAdd(index, dataset);
        fixtures::BruteForceTestIndex::TestGeneral(
            index, dataset, fixtures::search_param_tmp, iter.recall);
    });
}

TEST_CASE("(Daily) BruteForce Duplicate Build Test", "[ft][build][duplicate][bruteforce][daily]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(false);
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestDuplicateAdd(index, dataset);
        fixtures::BruteForceTestIndex::TestGeneral(
            index, dataset, fixtures::search_param_tmp, iter.recall);
    });
}

TEST_CASE("(PR) BruteForce With Attribute Filter Test", "[ft][filter_search][bruteforce][pr]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(true);
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer, true);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestBuildIndex(index, dataset, true);
        fixtures::TestIndex::TestWithAttr(index, dataset, fixtures::search_param_tmp, false);
        auto index2 =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        REQUIRE_NOTHROW(fixtures::test_serializion_file(*index, *index2, "serialize_bruteforce"));
        fixtures::TestIndex::TestWithAttr(index2, dataset, fixtures::search_param_tmp, true);
    });
}

TEST_CASE("(Daily) BruteForce With Attribute Filter Test",
          "[ft][filter_search][bruteforce][daily]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(false);
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer, true);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestBuildIndex(index, dataset, true);
        fixtures::TestIndex::TestWithAttr(index, dataset, fixtures::search_param_tmp, false);
        auto index2 =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        REQUIRE_NOTHROW(fixtures::test_serializion_file(*index, *index2, "serialize_bruteforce"));
        fixtures::TestIndex::TestWithAttr(index2, dataset, fixtures::search_param_tmp, true);
    });
}

TEST_CASE("(PR) BruteForce Mark Remove", "[ft][remove][bruteforce][pr]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(true);
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestMarkRemoveIndex(index, dataset, fixtures::search_param_tmp, true);
        fixtures::BruteForceTestIndex::TestGeneral(
            index, dataset, fixtures::search_param_tmp, iter.recall);
    });
}

TEST_CASE("(Daily) BruteForce Mark Remove", "[ft][remove][bruteforce][daily]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(false);
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestMarkRemoveIndex(index, dataset, fixtures::search_param_tmp, true);
        fixtures::BruteForceTestIndex::TestGeneral(
            index, dataset, fixtures::search_param_tmp, iter.recall);
    });
}

TEST_CASE("(PR) BruteForce RangeSearch After MarkRemove",
          "[ft][remove][range_search][bruteforce][pr]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(true);
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);

        auto train_result = index->Train(dataset->base_);
        REQUIRE(train_result.has_value());
        auto add_results = index->Add(dataset->base_);
        REQUIRE(add_results.has_value());

        auto base_num = dataset->base_->GetNumElements();
        auto base_dim = dataset->base_->GetDim();
        auto ids = dataset->base_->GetIds();

        int64_t remove_count = base_num / 2;
        std::vector<int64_t> remove_ids(ids, ids + remove_count);
        auto remove_result = index->Remove(remove_ids, vsag::RemoveMode::MARK_REMOVE);
        REQUIRE(remove_result.has_value());
        REQUIRE(index->GetNumberRemoved() == remove_count);

        std::unordered_set<int64_t> removed_set(remove_ids.begin(), remove_ids.end());
        auto queries = dataset->range_query_;
        auto query_count = queries->GetNumElements();
        auto radius = dataset->range_radius_;
        for (int64_t q = 0; q < query_count; ++q) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)
                ->Dim(base_dim)
                ->Float32Vectors(queries->GetFloat32Vectors() + q * base_dim)
                ->Owner(false);
            auto res = index->RangeSearch(query, radius[q], fixtures::search_param_tmp);
            REQUIRE(res.has_value());
            auto result_ids = res.value()->GetIds();
            auto result_dim = res.value()->GetDim();
            for (int64_t j = 0; j < result_dim; ++j) {
                REQUIRE(removed_set.count(result_ids[j]) == 0);
            }
        }
    });
}

TEST_CASE("(PR) BruteForce Remove By ID Test", "[ft][remove][bruteforce][pr]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(true);
    matrix.quantizers = {{"fp32", 0.99f}};
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestContinueAdd(index, dataset, true);
        fixtures::BruteForceTestIndex::TestGeneral(
            index, dataset, fixtures::search_param_tmp, iter.recall);
        auto total = static_cast<uint64_t>(dataset->base_->GetNumElements());
        for (uint64_t i = 0; i < total; ++i) {
            auto res = index->Remove(dataset->base_->GetIds()[i]);
            auto check_exist = index->CheckIdExist(dataset->base_->GetIds()[i]);
            REQUIRE(res.has_value());
            REQUIRE(res.value());
            REQUIRE(not check_exist);
            auto num = index->GetNumElements();
            REQUIRE(num == total - i - 1);
        }
    });
}

TEST_CASE("(Daily) BruteForce Remove By ID Test", "[ft][remove][bruteforce][daily]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(false);
    matrix.quantizers = {{"fp32", 0.99f}};
    matrix.ForEach(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, iter.dim, iter.quantizer);
        auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        auto dataset = fixtures::BruteForceTestIndex::pool.GetDatasetAndCreate(
            iter.dim, matrix.base_count, iter.metric);
        fixtures::TestIndex::TestContinueAdd(index, dataset, true);
        fixtures::BruteForceTestIndex::TestGeneral(
            index, dataset, fixtures::search_param_tmp, iter.recall);
        auto total = static_cast<uint64_t>(dataset->base_->GetNumElements());
        for (uint64_t i = 0; i < total; ++i) {
            auto res = index->Remove(dataset->base_->GetIds()[i]);
            auto check_exist = index->CheckIdExist(dataset->base_->GetIds()[i]);
            REQUIRE(res.has_value());
            REQUIRE(res.value());
            REQUIRE(not check_exist);
            auto num = index->GetNumElements();
            REQUIRE(num == total - i - 1);
        }
    });
}

TEST_CASE("(PR) BruteForce BruteForce Estimate Memory Test", "[ft][memory][bruteforce][pr]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(true);
    constexpr int64_t dim = 1536;
    matrix.ForEachDimQuantizer(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, dim, iter.quantizer);
        [[maybe_unused]] auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        [[maybe_unused]] auto val = index->EstimateMemory(1000);
    });
}

TEST_CASE("(Daily) BruteForce BruteForce Estimate Memory Test", "[ft][memory][bruteforce][daily]") {
    auto matrix = fixtures::BruteForceTestIndex::GetMatrix(false);
    constexpr int64_t dim = 1536;
    matrix.ForEachDimQuantizer(fixtures::BruteForceTestIndex::pool, [&](const auto& iter) {
        auto param = fixtures::BruteForceTestIndex::GenerateBruteForceBuildParametersString(
            iter.metric, dim, iter.quantizer);
        [[maybe_unused]] auto index =
            fixtures::TestIndex::TestFactory(fixtures::BruteForceTestIndex::name, param, true);
        [[maybe_unused]] auto val = index->EstimateMemory(1000);
    });
}
