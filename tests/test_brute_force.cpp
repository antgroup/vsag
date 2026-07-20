
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
#include <cstring>
#include <limits>
#include <sstream>

#include "functest.h"
#include "storage/serialization_tags.h"
#include "storage/streaming_serialization_test_utils.h"
#include "test_index.h"
#include "vsag/constants.h"
#include "vsag/options.h"
#include "vsag/search_request.h"

namespace fixtures {

class RejectAllFilter : public vsag::Filter {
public:
    bool
    CheckValid(int64_t id) const override {
        return false;
    }
};

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

namespace {

using vsag::test::InsertUnknownStreamingBlock;
using vsag::test::SetStreamingBlockVersion;

void
RequireStreamTail(std::stringstream& stream, const std::string& expected_tail) {
    std::string tail(expected_tail.size(), '\0');
    stream.read(tail.data(), static_cast<std::streamsize>(tail.size()));
    REQUIRE(tail == expected_tail);
}

}  // namespace

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

TEST_CASE("(PR) BruteForce Build & ContinueAdd Test", "[ft][build][bruteforce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceBuildAndContinueAdd(resource);
}

TEST_CASE("(Daily) BruteForce Build & ContinueAdd Test", "[ft][build][bruteforce][daily]") {
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

TEST_CASE("(PR) BruteForce Build Test", "[ft][build][bruteforce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceBuild(resource);
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

    auto excessive_parallelism =
        index->RangeSearch(query, 16.0F, R"({"parallelism": 32})", 4).value();
    fixtures::CheckSameRangeSearchResults(single, excessive_parallelism);

    auto filter = std::make_shared<fixtures::EvenIdFilter>();
    auto filtered_single = index->RangeSearch(query, 64.0F, "{}", filter, 3).value();
    auto filtered_parallel =
        index->RangeSearch(query, 64.0F, R"({"parallelism": 4})", filter, 3).value();
    fixtures::CheckSameRangeSearchResults(filtered_single, filtered_parallel);
}

TEST_CASE("(Daily) BruteForce Build Test", "[ft][build][bruteforce][daily]") {
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

TEST_CASE("(PR) BruteForce Add Test", "[ft][build][bruteforce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceAdd(resource);
}

TEST_CASE("(Daily) BruteForce Add Test", "[ft][build][bruteforce][daily]") {
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

TEST_CASE("(PR) BruteForce Concurrent Add Test", "[ft][build][bruteforce][concurrent][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceConcurrentAdd(resource);
}

TEST_CASE("(Daily) BruteForce Concurrent Add Test", "[ft][build][bruteforce][concurrent][daily]") {
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

static void
TestBruteForceGetStreamingMetadataOnEmptyIndex() {
    using namespace fixtures;
    auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString("l2", 16, "fp32");
    auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);

    std::stringstream stream;
    REQUIRE(index->SerializeStreaming(stream).has_value());

    auto metadata_result = vsag::Index::GetStreamingMetadata(stream);
    REQUIRE(metadata_result.has_value());
    REQUIRE(metadata_result.value().metadata_json.find("\"_empty\":true") != std::string::npos);
    REQUIRE(metadata_result.value().blocks.empty());
}

static void
TestBruteForceSerializeStreaming() {
    using namespace fixtures;
    auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString("l2", 16, "fp32");
    auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
    auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(16, 100, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);

    std::stringstream stream;
    auto serialize_result = index->SerializeStreaming(stream);
    REQUIRE(serialize_result.has_value());
    auto bytes = stream.str();
    REQUIRE(bytes.substr(0, 8) == vsag::SERIAL_STREAM_MAGIC);

    auto index2 = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
    std::stringstream deserialize_stream(bytes);
    auto deserialize_result = index2->DeserializeStreaming(deserialize_stream);
    REQUIRE(deserialize_result.has_value());
    REQUIRE(index2->GetNumElements() == index->GetNumElements());

    std::stringstream load_stream(bytes);
    auto load_result = vsag::Index::Load(load_stream, R"({"base_io_type": "memory_io"})");
    REQUIRE(load_result.has_value());
    auto index3 = load_result.value();
    REQUIRE(index3->GetNumElements() == index->GetNumElements());

    std::stringstream buffer_load_stream(bytes);
    auto buffer_load_result =
        vsag::Index::Load(buffer_load_stream, R"({"base_io_type": "buffer_io"})");
    REQUIRE(buffer_load_result.has_value());
    REQUIRE(buffer_load_result.value()->GetNumElements() == index->GetNumElements());

    std::stringstream reader_load_stream(bytes);
    auto reader_load_result =
        vsag::Index::Load(reader_load_stream, R"({"base_io_type": "reader_io"})");
    REQUIRE_FALSE(reader_load_result.has_value());

    auto query = get_one_query(dataset->query_, 0);
    auto result = index->KnnSearch(query, 10, search_param_tmp);
    auto result2 = index2->KnnSearch(query, 10, search_param_tmp);
    auto result3 = index3->KnnSearch(query, 10, search_param_tmp);
    REQUIRE(result.has_value());
    REQUIRE(result2.has_value());
    REQUIRE(result3.has_value());
    REQUIRE(result.value()->GetDim() == result2.value()->GetDim());
    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
        REQUIRE(result.value()->GetIds()[i] == result2.value()->GetIds()[i]);
        REQUIRE(result.value()->GetDistances()[i] == result2.value()->GetDistances()[i]);
        REQUIRE(result.value()->GetIds()[i] == result3.value()->GetIds()[i]);
        REQUIRE(result.value()->GetDistances()[i] == result3.value()->GetDistances()[i]);
    }
}

TEST_CASE("BruteForce streaming compatibility",
          "[ft][serialize][streaming][bruteforce][compatibility]") {
    using namespace fixtures;
    auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString("l2", 16, "fp32");
    auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
    auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(16, 100, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);

    std::stringstream stream;
    REQUIRE(index->SerializeStreaming(stream).has_value());
    const auto bytes = stream.str();

    SECTION("skips unknown non-critical block") {
        auto mutated = InsertUnknownStreamingBlock(bytes, false);
        auto restored = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
        std::stringstream deserialize_stream(mutated);
        REQUIRE(restored->DeserializeStreaming(deserialize_stream).has_value());
        REQUIRE(restored->GetNumElements() == index->GetNumElements());
    }

    SECTION("skips unknown non-critical block with unsupported version") {
        auto mutated = InsertUnknownStreamingBlock(bytes, false, 99);
        auto restored = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
        std::stringstream deserialize_stream(mutated);
        REQUIRE(restored->DeserializeStreaming(deserialize_stream).has_value());
        REQUIRE(restored->GetNumElements() == index->GetNumElements());
    }

    SECTION("rejects unknown critical block") {
        auto mutated = InsertUnknownStreamingBlock(bytes, true);
        auto restored = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
        std::stringstream deserialize_stream(mutated);
        REQUIRE_FALSE(restored->DeserializeStreaming(deserialize_stream).has_value());
    }

    SECTION("rejects unsupported critical block version") {
        auto mutated =
            SetStreamingBlockVersion(bytes, vsag::StreamSerializationTag::BASE_CODES, 99);
        auto restored = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
        std::stringstream deserialize_stream(mutated);
        REQUIRE_FALSE(restored->DeserializeStreaming(deserialize_stream).has_value());
    }

    SECTION("supports nested load memory policy") {
        std::stringstream load_stream(bytes);
        auto loaded = vsag::Index::Load(load_stream, R"({"load":{"base_codes":"memory"}})");
        REQUIRE(loaded.has_value());
        REQUIRE(loaded.value()->GetNumElements() == index->GetNumElements());
    }

    SECTION("rejects nested load reader policy for required base codes") {
        std::stringstream load_stream(bytes);
        REQUIRE_FALSE(
            vsag::Index::Load(load_stream, R"({"load":{"base_codes":"reader"}})").has_value());
    }

    SECTION("rejects invalid streaming load JSON as invalid argument") {
        std::stringstream load_stream(bytes);
        auto loaded = vsag::Index::Load(load_stream, "{");
        REQUIRE_FALSE(loaded.has_value());
        REQUIRE(loaded.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("rejects wrong streaming load policy type as invalid argument") {
        std::stringstream load_stream(bytes);
        auto loaded = vsag::Index::Load(load_stream, R"({"load":{"base_codes":false}})");
        REQUIRE_FALSE(loaded.has_value());
        REQUIRE(loaded.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("BruteForce streaming Load skips attribute filter state",
          "[ft][serialize][streaming][bruteforce][compatibility]") {
    using namespace fixtures;
    auto param =
        BruteForceTestIndex::GenerateBruteForceBuildParametersString("l2", 16, "fp32", true);
    auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
    auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(16, 100, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);

    std::stringstream stream;
    REQUIRE(index->SerializeStreaming(stream).has_value());
    std::stringstream load_stream(stream.str());
    auto loaded = vsag::Index::Load(load_stream, R"({"load":{"use_attribute_filter":"skip"}})");
    REQUIRE(loaded.has_value());
    REQUIRE(loaded.value()->GetNumElements() == index->GetNumElements());

    auto query = get_one_query(dataset->query_, 0);
    auto search_result = loaded.value()->KnnSearch(query, 10, search_param_tmp);
    REQUIRE(search_result.has_value());

    vsag::SearchRequest request;
    request.query_ = query;
    request.topk_ = 10;
    request.params_str_ = search_param_tmp;
    request.enable_attribute_filter_ = true;
    request.attribute_filter_str_ = R"(multi_in(term_0, "0", "|"))";
    auto attr_search_result = loaded.value()->SearchWithRequest(request);
    REQUIRE_FALSE(attr_search_result.has_value());
    REQUIRE(attr_search_result.error().type == vsag::ErrorType::INVALID_ARGUMENT);

    auto remove_result = loaded.value()->Remove(dataset->base_->GetIds()[0]);
    REQUIRE(remove_result.has_value());
}

TEST_CASE("BruteForce empty streaming index consumes section end",
          "[ft][serialize][streaming][bruteforce][compatibility]") {
    using namespace fixtures;
    auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString("l2", 16, "fp32");
    auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
    std::stringstream stream;
    REQUIRE(index->SerializeStreaming(stream).has_value());
    stream << "tail";
    auto bytes = stream.str();

    auto restored = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
    std::stringstream deserialize_stream(bytes);
    REQUIRE(restored->DeserializeStreaming(deserialize_stream).has_value());
    RequireStreamTail(deserialize_stream, "tail");

    std::stringstream load_stream(bytes);
    auto loaded = vsag::Index::Load(load_stream, "{}");
    REQUIRE(loaded.has_value());
    RequireStreamTail(load_stream, "tail");
}

TEST_CASE("(PR) BruteForce Serialize File Test", "[ft][serialize][bruteforce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceSerializeFile(resource);
}

TEST_CASE("(PR) BruteForce Streaming Metadata Empty Index Test",
          "[ft][serialize][streaming][bruteforce][pr]") {
    TestBruteForceGetStreamingMetadataOnEmptyIndex();
}

TEST_CASE("(PR) BruteForce Streaming Serialize Test",
          "[ft][serialize][streaming][bruteforce][pr]") {
    TestBruteForceSerializeStreaming();
}

TEST_CASE("(Daily) BruteForce Serialize File Test", "[ft][serialize][bruteforce][daily]") {
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

TEST_CASE("(PR) BruteForce Clone Test", "[ft][clone][bruteforce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceClone(resource);
}

TEST_CASE("(Daily) BruteForce Clone Test", "[ft][clone][bruteforce][daily]") {
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
                //                TestIndex::TestContinueAddIgnoreRequire(index.value(), dataset);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("(PR) BruteForce Build & ContinueAdd Test With Random Allocator",
          "[ft][build][bruteforce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceRandomAllocator(resource);
}

TEST_CASE("(Daily) BruteForce Build & ContinueAdd Test With Random Allocator",
          "[ft][build][bruteforce][daily]") {
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

TEST_CASE("(PR) BruteForce GetDistance By ID Test", "[ft][distance][bruteforce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceCalcDistanceById(resource);
}

TEST_CASE("(Daily) BruteForce GetDistance By ID Test", "[ft][distance][bruteforce][daily]") {
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

TEST_CASE("(PR) BruteForce Duplicate Build Test", "[ft][build][duplicate][bruteforce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceDuplicateBuild(resource);
}

TEST_CASE("(Daily) BruteForce Duplicate Build Test", "[ft][build][duplicate][bruteforce][daily]") {
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

TEST_CASE("(PR) BruteForce With Attribute Filter Test", "[ft][filter_search][bruteforce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceWithAttrFilter(resource);
}

TEST_CASE("(Daily) BruteForce With Attribute Filter Test",
          "[ft][filter_search][bruteforce][daily]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(false);
    TestBruteForceWithAttrFilter(resource);
}

static void
TestBruteForceMarkRemove(const fixtures::BruteForceResourcePtr& resource) {
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
                TestIndex::TestMarkRemoveIndex(index, dataset, search_param_tmp, true);
                BruteForceTestIndex::TestGeneral(index, dataset, search_param_tmp, recall);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("(PR) BruteForce Mark Remove", "[ft][remove][bruteforce][pr]") {
    auto test_index = std::make_shared<fixtures::BruteForceTestIndex>();
    auto resource = test_index->GetResource(true);
    TestBruteForceMarkRemove(resource);
}

TEST_CASE("(Daily) BruteForce Mark Remove", "[ft][remove][bruteforce][daily]") {
    auto test_index = std::make_shared<fixtures::BruteForceTestIndex>();
    auto resource = test_index->GetResource(false);
    TestBruteForceMarkRemove(resource);
}

TEST_CASE("(PR) BruteForce RangeSearch After MarkRemove",
          "[ft][remove][range_search][bruteforce][pr]") {
    // Regression: RangeSearch must exclude documents removed via MARK_REMOVE,
    // mirroring KnnSearch behavior (see create_search_filter usage).
    using namespace fixtures;
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
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

                // Build and add
                auto train_result = index->Train(dataset->base_);
                REQUIRE(train_result.has_value());
                auto add_results = index->Add(dataset->base_);
                REQUIRE(add_results.has_value());

                auto base_num = dataset->base_->GetNumElements();
                auto base_dim = dataset->base_->GetDim();
                auto ids = dataset->base_->GetIds();

                // Mark-remove half of the base data
                int64_t remove_count = base_num / 2;
                std::vector<int64_t> remove_ids(ids, ids + remove_count);
                auto remove_result = index->Remove(remove_ids, vsag::RemoveMode::MARK_REMOVE);
                REQUIRE(remove_result.has_value());
                REQUIRE(index->GetNumberRemoved() == remove_count);

                // RangeSearch from each query; verify no removed id appears.
                // Build a hash set of removed ids for O(1) lookup.
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
                    auto res = index->RangeSearch(query, radius[q], search_param_tmp);
                    REQUIRE(res.has_value());
                    auto result_ids = res.value()->GetIds();
                    auto result_dim = res.value()->GetDim();
                    for (int64_t j = 0; j < result_dim; ++j) {
                        REQUIRE(removed_set.count(result_ids[j]) == 0);
                    }
                }

                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
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

TEST_CASE("(PR) BruteForce Remove By ID Test", "[ft][remove][bruteforce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceRemoveById(resource);
}

TEST_CASE("(Daily) BruteForce Remove By ID Test", "[ft][remove][bruteforce][daily]") {
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

TEST_CASE("(PR) BruteForce BruteForce Estimate Memory Test", "[ft][memory][bruteforce][pr]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(true);
    TestBruteForceEstimateMemory(resource);
}

TEST_CASE("(Daily) BruteForce BruteForce Estimate Memory Test", "[ft][memory][bruteforce][daily]") {
    auto resource = fixtures::BruteForceTestIndex::GetResource(false);
    TestBruteForceEstimateMemory(resource);
}

// BruteForce Reasoning Tests

TEST_CASE("(PR) BruteForce SearchWithRequest Reasoning", "[ft][bruteforce][reasoning][pr]") {
    using namespace fixtures;

    auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString("l2", 16, "fp32");
    auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
    auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(16, 256, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dataset->base_->GetDim())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);

    vsag::SearchRequest req;
    req.topk_ = 5;
    req.params_str_ = "";
    req.query_ = query;
    req.expected_labels_ = {dataset->base_->GetIds()[0]};

    auto result = index->SearchWithRequest(req);
    REQUIRE(result.has_value());
    REQUIRE_FALSE(result.value()->GetReasoning().empty());
    REQUIRE(result.value()->GetReasoning().find("expected_analysis") != std::string::npos);
    REQUIRE(result.value()->GetReasoning().find("missed_targets") != std::string::npos);

    // With RejectAll filter, expected label should be diagnosed as filter_rejected
    req.enable_filter_ = true;
    req.filter_ = std::make_shared<RejectAllFilter>();

    auto empty_result = index->SearchWithRequest(req);
    REQUIRE(empty_result.has_value());
    REQUIRE(empty_result.value()->GetDim() == 0);
    REQUIRE_FALSE(empty_result.value()->GetReasoning().empty());
    REQUIRE(empty_result.value()->GetReasoning().find("missed_targets") != std::string::npos);
    REQUIRE(empty_result.value()->GetReasoning().find("filter_rejected") != std::string::npos);
}

TEST_CASE("(PR) BruteForce Reasoning Found Verification", "[ft][bruteforce][reasoning][pr]") {
    using namespace fixtures;

    auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString("l2", 16, "fp32");
    auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
    auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(16, 256, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dataset->base_->GetDim())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);

    vsag::SearchRequest req;
    req.topk_ = 10;
    req.params_str_ = "";
    req.query_ = query;

    auto baseline = index->SearchWithRequest(req);
    REQUIRE(baseline.has_value());
    REQUIRE(baseline.value()->GetDim() > 0);

    auto* ids = baseline.value()->GetIds();
    int64_t found_label = ids[0];

    req.expected_labels_ = {found_label};
    auto result = index->SearchWithRequest(req);
    REQUIRE(result.has_value());
    REQUIRE_FALSE(result.value()->GetReasoning().empty());

    auto reasoning = result.value()->GetReasoning();
    REQUIRE(reasoning.find("1/1") != std::string::npos);
    REQUIRE(reasoning.find("0 missed") != std::string::npos);
}

TEST_CASE("(PR) BruteForce Reasoning Multiple Labels Mixed", "[ft][bruteforce][reasoning][pr]") {
    using namespace fixtures;

    auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString("l2", 16, "fp32");
    auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
    auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(16, 256, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dataset->base_->GetDim())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);

    vsag::SearchRequest req;
    req.topk_ = 10;
    req.params_str_ = "";
    req.query_ = query;

    auto baseline = index->SearchWithRequest(req);
    REQUIRE(baseline.has_value());
    REQUIRE(baseline.value()->GetDim() > 0);

    auto* ids = baseline.value()->GetIds();
    // Test with a label that is in the top-k result (should be found)
    // and a label from the dataset that is outside the top-k (should be
    // diagnosed as ef_too_small since BruteForce visits all vectors but
    // the top-k heap may evict it).
    int64_t found_label = ids[0];
    int64_t missed_label = dataset->base_->GetIds()[0];
    // Ensure missed_label is not in the top-10 result
    bool missed_in_result = false;
    for (int64_t i = 0; i < baseline.value()->GetDim(); ++i) {
        if (ids[i] == missed_label) {
            missed_in_result = true;
            break;
        }
    }
    if (missed_in_result) {
        // Pick a different label that is not in the result
        for (int64_t i = 0; i < dataset->base_->GetNumElements(); ++i) {
            missed_label = dataset->base_->GetIds()[i];
            missed_in_result = false;
            for (int64_t j = 0; j < baseline.value()->GetDim(); ++j) {
                if (ids[j] == missed_label) {
                    missed_in_result = true;
                    break;
                }
            }
            if (!missed_in_result) {
                break;
            }
        }
    }

    req.expected_labels_ = {found_label, missed_label};
    auto result = index->SearchWithRequest(req);
    REQUIRE(result.has_value());
    REQUIRE_FALSE(result.value()->GetReasoning().empty());

    auto reasoning = result.value()->GetReasoning();
    REQUIRE(reasoning.find("expected_analysis") != std::string::npos);
    // found_label should be found; missed_label may be found or missed
    // depending on the dataset. At minimum the report should be valid.
    if (!missed_in_result) {
        REQUIRE(reasoning.find("missed_targets") != std::string::npos);
    }
}

TEST_CASE("(PR) BruteForce Reasoning With Filter Rejection", "[ft][bruteforce][reasoning][pr]") {
    using namespace fixtures;

    auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString("l2", 16, "fp32");
    auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
    auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(16, 256, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dataset->base_->GetDim())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);

    // Use RejectAllFilter to guarantee that expected labels are rejected.
    // This makes the test deterministic: all expected labels will be
    // diagnosed as filter_rejected since RejectAllFilter rejects everything.
    int64_t target_label = dataset->base_->GetIds()[0];

    vsag::SearchRequest req;
    req.topk_ = 5;
    req.params_str_ = "";
    req.query_ = query;
    req.enable_filter_ = true;
    req.filter_ = std::make_shared<RejectAllFilter>();
    req.expected_labels_ = {target_label};

    auto result = index->SearchWithRequest(req);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == 0);
    REQUIRE_FALSE(result.value()->GetReasoning().empty());

    auto reasoning = result.value()->GetReasoning();
    REQUIRE(reasoning.find("expected_analysis") != std::string::npos);
    REQUIRE(reasoning.find("filter_rejected") != std::string::npos);
}

TEST_CASE("(PR) BruteForce Reasoning Does Not Affect Results", "[ft][bruteforce][reasoning][pr]") {
    using namespace fixtures;

    auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString("l2", 16, "fp32");
    auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
    auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(16, 256, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dataset->base_->GetDim())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);

    vsag::SearchRequest req_without;
    req_without.topk_ = 10;
    req_without.params_str_ = "";
    req_without.query_ = query;

    auto result_without = index->SearchWithRequest(req_without);
    REQUIRE(result_without.has_value());
    REQUIRE(result_without.value()->GetDim() > 0);

    vsag::SearchRequest req_with;
    req_with.topk_ = 10;
    req_with.params_str_ = "";
    req_with.query_ = query;
    req_with.expected_labels_ = {result_without.value()->GetIds()[0]};

    auto result_with = index->SearchWithRequest(req_with);
    REQUIRE(result_with.has_value());
    REQUIRE(result_with.value()->GetDim() == result_without.value()->GetDim());

    auto dim_without = result_without.value()->GetDim();
    for (int64_t i = 0; i < dim_without; ++i) {
        REQUIRE(result_with.value()->GetIds()[i] == result_without.value()->GetIds()[i]);
        REQUIRE(result_with.value()->GetDistances()[i] ==
                result_without.value()->GetDistances()[i]);
    }
}

TEST_CASE("(PR) BruteForce Reasoning Empty Expected Labels", "[ft][bruteforce][reasoning][pr]") {
    using namespace fixtures;

    auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString("l2", 16, "fp32");
    auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
    auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(16, 256, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dataset->base_->GetDim())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);

    vsag::SearchRequest req;
    req.topk_ = 5;
    req.params_str_ = "";
    req.query_ = query;
    req.expected_labels_ = {};

    auto result = index->SearchWithRequest(req);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    // No reasoning analysis when expected_labels is empty
    REQUIRE(result.value()->GetReasoning().find("expected_analysis") == std::string::npos);
}

TEST_CASE("(PR) BruteForce Reasoning No Output When Disabled", "[ft][bruteforce][reasoning][pr]") {
    using namespace fixtures;

    auto param = BruteForceTestIndex::GenerateBruteForceBuildParametersString("l2", 16, "fp32");
    auto index = TestIndex::TestFactory(BruteForceTestIndex::name, param, true);
    auto dataset = BruteForceTestIndex::pool.GetDatasetAndCreate(16, 256, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dataset->base_->GetDim())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);

    // When expected_labels is empty, reasoning is completely disabled:
    // no ReasoningContext is created, and the result should contain no
    // reasoning analysis. This is a deterministic check (no timing).
    vsag::SearchRequest req;
    req.topk_ = 5;
    req.params_str_ = "";
    req.query_ = query;
    req.expected_labels_ = {};

    auto result = index->SearchWithRequest(req);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    REQUIRE(result.value()->GetReasoning().find("expected_analysis") == std::string::npos);

    // Also verify that KnnSearch (which delegates to SearchWithRequest with
    // empty expected_labels) produces identical results with and without
    // the reasoning code path active.
    auto knn_result = index->KnnSearch(query, 5, "", vsag::BitsetPtr(nullptr));
    REQUIRE(knn_result.has_value());
    REQUIRE(knn_result.value()->GetDim() == result.value()->GetDim());
    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
        REQUIRE(result.value()->GetIds()[i] == knn_result.value()->GetIds()[i]);
    }
}
