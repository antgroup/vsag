
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

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstring>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <sstream>
#include <utility>

#include "algorithm/pyramid/pyramid_build_cache.h"
#include "functest.h"
#include "impl/allocator/safe_allocator.h"
#include "storage/serialization_tags.h"
#include "storage/stream_reader.h"
#include "storage/streaming_serialization_test_utils.h"
#include "test_index.h"
#include "vsag/vsag.h"

struct PyramidParam {
    std::vector<int> no_build_levels = std::vector<int>{0, 1, 2};
    std::string base_quantization_type = "fp32";
    std::string precise_quantization_type = "fp32";
    std::string graph_type = "nsw";
    bool use_reorder = false;
    bool support_duplicate = false;
    uint64_t rabitq_bits_per_dim_base = 1;
    bool fast_encode_rabitq = true;
    std::optional<std::string> root_graph_type;
    std::optional<std::string> graph_storage_type;
};

namespace fixtures {
class PyramidTestIndex : public fixtures::TestIndex {
public:
    static std::string
    GeneratePyramidBuildParametersString(const std::string& metric_type,
                                         int64_t dim,
                                         const PyramidParam& param);

    static std::string
    GeneratePyramidSearchParametersString(
        int64_t ef_search,
        double timeout_ms = static_cast<double>(std::numeric_limits<uint32_t>::max()));

    static TestDatasetPool pool;

    static std::vector<int> dims;

    static std::vector<std::vector<int>> levels;

    constexpr static uint64_t base_count = 1000;

    constexpr static const char* search_param_tmp = R"(
        {{
            "pyramid": {{
                "ef_search": {},
                "timeout_ms": {}
            }}
        }})";
};

TestDatasetPool PyramidTestIndex::pool{};
std::vector<int> PyramidTestIndex::dims = fixtures::get_common_used_dims(1);
std::vector<std::vector<int>> PyramidTestIndex::levels{{0, 1}, {0, 2}, {0, 1, 2}};
std::string
PyramidTestIndex::GeneratePyramidBuildParametersString(const std::string& metric_type,
                                                       int64_t dim,
                                                       const PyramidParam& param) {
    constexpr auto parameter_temp = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "max_degree": 32,
            "alpha": 1.2,
            "graph_iter_turn": 15,
            "neighbor_sample_rate": 0.2,
            "no_build_levels": [{}],
            "graph_type": "{}",
            "base_quantization_type": "{}",
            "rabitq_bits_per_dim_base": {},
            "fast_encode_rabitq": {},
            "fast_encode_rabitq_rounds": 6,
            "precise_quantization_type": "{}",
            "use_reorder": {},
            "index_min_size": 28{},
            "support_duplicate": {}
        }}
    }}
    )";
    const auto root_graph_parameter =
        param.root_graph_type.has_value()
            ? fmt::format(",\n            \"root_graph_type\": \"{}\"", *param.root_graph_type)
            : "";
    const auto graph_storage_parameter =
        param.graph_storage_type.has_value()
            ? fmt::format(",\n            \"graph_storage_type\": \"{}\"",
                          *param.graph_storage_type)
            : "";
    auto build_parameters_str = fmt::format(parameter_temp,
                                            metric_type,
                                            dim,
                                            fmt::join(param.no_build_levels, ","),
                                            param.graph_type,
                                            param.base_quantization_type,
                                            param.rabitq_bits_per_dim_base,
                                            param.fast_encode_rabitq,
                                            param.precise_quantization_type,
                                            param.use_reorder,
                                            root_graph_parameter + graph_storage_parameter,
                                            param.support_duplicate);
    return build_parameters_str;
}

std::string
PyramidTestIndex::GeneratePyramidSearchParametersString(int64_t ef_search, double timeout_ms) {
    return fmt::format(search_param_tmp, ef_search, timeout_ms);
}

}  // namespace fixtures

namespace {

auto
MakeDenseDataset(const std::vector<std::array<float, 4>>& vectors,
                 const std::vector<int64_t>& ids,
                 const std::vector<std::string>& paths) -> vsag::DatasetPtr {
    REQUIRE(vectors.size() == ids.size());
    REQUIRE(vectors.size() == paths.size());

    auto dataset = vsag::Dataset::Make();
    auto* raw_vectors = new float[vectors.size() * 4];
    auto* raw_ids = new int64_t[ids.size()];
    auto* raw_paths = new std::string[paths.size()];

    for (size_t i = 0; i < vectors.size(); ++i) {
        std::copy(vectors[i].begin(), vectors[i].end(), raw_vectors + i * 4);
        raw_ids[i] = ids[i];
        raw_paths[i] = paths[i];
    }

    dataset->NumElements(static_cast<int64_t>(vectors.size()))
        ->Dim(4)
        ->Float32Vectors(raw_vectors)
        ->Ids(raw_ids)
        ->Paths(raw_paths)
        ->Owner(true);
    return dataset;
}

auto
MakeSingleQuery(const std::array<float, 4>& vector, const std::string& path) -> vsag::DatasetPtr {
    auto dataset = vsag::Dataset::Make();
    auto* raw_vector = new float[4];
    auto* raw_path = new std::string[1];

    std::copy(vector.begin(), vector.end(), raw_vector);
    raw_path[0] = path;

    dataset->NumElements(1)->Dim(4)->Float32Vectors(raw_vector)->Paths(raw_path)->Owner(true);
    return dataset;
}

auto
CollectIds(const vsag::DatasetPtr& result) -> std::set<int64_t> {
    std::set<int64_t> ids;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        ids.insert(result->GetIds()[i]);
    }
    return ids;
}

uint32_t
SearchAndGetHops(const vsag::IndexPtr& index,
                 const vsag::DatasetPtr& query,
                 int64_t topk,
                 const std::string& search_param) {
    auto result = index->KnnSearch(query, topk, search_param);
    REQUIRE(result.has_value());
    const auto hops = result.value()->GetStatistics({"hops"});
    REQUIRE(hops.size() == 1);
    return static_cast<uint32_t>(std::stoul(hops.front()));
}

void
RequireDistancesNearZero(const vsag::DatasetPtr& result, const std::set<int64_t>& expected_ids) {
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        if (expected_ids.count(result->GetIds()[i]) != 0) {
            REQUIRE(std::abs(result->GetDistances()[i]) <= 1e-6F);
        }
    }
}

using vsag::test::EraseStreamingBlock;
using vsag::test::InsertUnknownStreamingBlock;

class BlockSizeLimitGuard {
public:
    explicit BlockSizeLimitGuard(uint64_t block_size_limit)
        : origin_size_(vsag::Options::Instance().block_size_limit()) {
        vsag::Options::Instance().set_block_size_limit(block_size_limit);
    }

    BlockSizeLimitGuard(const BlockSizeLimitGuard&) = delete;
    BlockSizeLimitGuard&
    operator=(const BlockSizeLimitGuard&) = delete;
    BlockSizeLimitGuard(BlockSizeLimitGuard&&) = delete;
    BlockSizeLimitGuard&
    operator=(BlockSizeLimitGuard&&) = delete;

    ~BlockSizeLimitGuard() {
        vsag::Options::Instance().set_block_size_limit(origin_size_);
    }

private:
    uint64_t origin_size_;
};

}  // namespace

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Build & ContinueAdd Test",
                             "[ft][build][pyramid]") {
    auto metric_type = GENERATE("l2", "ip", "cosine");
    auto use_reorder = GENERATE(true, false);
    auto immutable = GENERATE(true, false);
    PyramidParam pyramid_param;
    pyramid_param.graph_type = GENERATE("nsw", "odescent");
    pyramid_param.no_build_levels = {0, 1, 2};
    pyramid_param.use_reorder = use_reorder;
    if (use_reorder) {
        pyramid_param.base_quantization_type = "rabitq";
        pyramid_param.precise_quantization_type = "fp32";
    }
    const std::string name = "pyramid";
    auto search_param = GeneratePyramidSearchParametersString(100);
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}, graph_type={}, use_reorder={}, immutable={}",
                         metric_type,
                         dim,
                         pyramid_param.graph_type,
                         use_reorder,
                         immutable));
        auto param = GeneratePyramidBuildParametersString(metric_type, dim, pyramid_param);
        auto index = TestFactory(name, param, true);
        REQUIRE(index->GetIndexType() == vsag::IndexType::PYRAMID);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type, /*with_path=*/true);
        TestContinueAdd(index, dataset, true);
        if (immutable) {
            index->SetImmutable();
        }
        TestKnnSearch(index, dataset, search_param, 0.94, true);
        TestFilterSearch(index, dataset, search_param, 0.94, true);
        TestRangeSearch(index, dataset, search_param, 0.94, 10, true);
        TestRangeSearch(index, dataset, search_param, 0.49, 5, true);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid RangeSearch Returns Empty Result",
                             "[ft][pyramid]") {
    PyramidParam pyramid_param;
    const auto param = GeneratePyramidBuildParametersString("l2", 4, pyramid_param);
    auto index = TestFactory("pyramid", param, true);

    auto base = MakeDenseDataset(
        {{{1.0F, 0.0F, 0.0F, 0.0F}}, {{2.0F, 0.0F, 0.0F, 0.0F}}}, {1, 2}, {"a/d/f", "a/d/f"});
    REQUIRE(index->Build(base).has_value());

    auto query = MakeSingleQuery({10.0F, 0.0F, 0.0F, 0.0F}, "a/d/f");
    auto result = index->RangeSearch(query, 0.0F, GeneratePyramidSearchParametersString(20));
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == 0);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Set Immutable",
                             "[ft][immutable][pyramid]") {
    const auto metric_type = "l2";
    PyramidParam pyramid_param;
    const auto param =
        GeneratePyramidBuildParametersString(metric_type, dims.front(), pyramid_param);
    auto index = TestFactory("pyramid", param, true);
    auto dataset =
        pool.GetDatasetAndCreate(dims.front(), base_count, metric_type, /*with_path=*/true);
    TestBuildIndex(index, dataset, true);

    REQUIRE(index->SetImmutable().has_value());
    REQUIRE(index->SetImmutable().has_value());

    auto add_result = index->Add(dataset->base_);
    REQUIRE_FALSE(add_result.has_value());
    REQUIRE(add_result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);

    auto build_result = index->Build(dataset->base_);
    REQUIRE_FALSE(build_result.has_value());
    REQUIRE(build_result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);

    std::vector<int64_t> ids{dataset->base_->GetIds()[0]};
    auto remove_result = index->Remove(ids);
    REQUIRE_FALSE(remove_result.has_value());
    REQUIRE(remove_result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);

    TestKnnSearch(index, dataset, GeneratePyramidSearchParametersString(100), 0.94, true);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid multi-bit RaBitQ fast encoding",
                             "[ft][pyramid][rabitq]") {
    constexpr int64_t dim = 64;
    constexpr uint64_t count = 256;
    const bool fast_encode = GENERATE(false, true);

    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {0, 1, 2};
    pyramid_param.base_quantization_type = "rabitq";
    pyramid_param.precise_quantization_type = "fp32";
    pyramid_param.use_reorder = true;
    pyramid_param.rabitq_bits_per_dim_base = 8;
    pyramid_param.fast_encode_rabitq = fast_encode;

    CAPTURE(fast_encode);
    auto param = GeneratePyramidBuildParametersString("l2", dim, pyramid_param);
    auto index = TestFactory("pyramid", param, true);
    auto dataset = pool.GetDatasetAndCreate(dim, count, "l2", /*with_path=*/true);
    TestBuildIndex(index, dataset, true);
    auto query = fixtures::get_one_query(dataset->query_, 0);
    auto result = index->KnnSearch(query, 5, GeneratePyramidSearchParametersString(100));
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    REQUIRE(result.value()->GetDim() <= 5);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid split RaBitQ builds with ODescent",
                             "[ft][pyramid][rabitq]") {
    constexpr int64_t dim = 64;
    constexpr uint64_t count = 256;
    constexpr auto parameter_temp = R"(
    {{
        "dtype": "float32",
        "metric_type": "l2",
        "dim": {},
        "index_param": {{
            "max_degree": 32,
            "ef_construction": 100,
            "alpha": 1.2,
            "graph_type": "odescent",
            "graph_iter_turn": 3,
            "neighbor_sample_rate": 0.2,
            "no_build_levels": [0],
            "base_quantization_type": "rabitq",
            "rabitq_bits_per_dim_base": 3,
            "precise_quantization_type": "rabitq",
            "rabitq_bits_per_dim_precise": 5,
            "use_reorder": true,
            "index_min_size": 28
        }}
    }})";

    auto index = TestFactory("pyramid", fmt::format(parameter_temp, dim), true);
    auto dataset = pool.GetDatasetAndCreate(dim, count, "l2", /*with_path=*/true);
    TestBuildIndex(index, dataset, true);

    auto query = fixtures::get_one_query(dataset->query_, 0);
    auto result = index->KnnSearch(query, 5, GeneratePyramidSearchParametersString(100));
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    REQUIRE(result.value()->GetDim() <= 5);
    REQUIRE_NOTHROW(index->GetStats());
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid split RaBitQ promotes without raw vectors",
                             "[ft][pyramid][rabitq][add]") {
    constexpr int64_t dim = 64;
    constexpr uint64_t count = 128;
    constexpr auto parameter_temp = R"(
    {{
        "dtype": "float32",
        "metric_type": "l2",
        "dim": {},
        "index_param": {{
            "max_degree": 16,
            "ef_construction": 50,
            "alpha": 1.2,
            "graph_type": "nsw",
            "no_build_levels": [0],
            "base_quantization_type": "rabitq",
            "rabitq_bits_per_dim_base": 3,
            "precise_quantization_type": "rabitq",
            "rabitq_bits_per_dim_precise": 5,
            "use_reorder": true,
            "index_min_size": 8
        }}
    }})";

    auto param = fmt::format(parameter_temp, dim);
    auto index = TestFactory("pyramid", param, true);
    auto dataset = pool.GetDatasetAndCreate(dim, count, "l2", /*with_path=*/true);
    TestBuildIndex(index, dataset, true);

    auto query = fixtures::get_one_query(dataset->query_, 0);
    auto result = index->KnnSearch(query, 5, GeneratePyramidSearchParametersString(100));
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    REQUIRE_NOTHROW(index->GetStats());

    auto restored = TestFactory("pyramid", param, true);
    TestSerializeBinarySet(
        index, restored, dataset, GeneratePyramidSearchParametersString(100), true);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid MRLE RaBitQ Split",
                             "[ft][pyramid][rabitq_split][MRLE]") {
    constexpr int64_t dim = 64;
    constexpr uint64_t count = 256;

    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {0, 1, 2};
    pyramid_param.base_quantization_type = "tq";
    pyramid_param.precise_quantization_type = "rabitq";
    pyramid_param.use_reorder = true;
    pyramid_param.rabitq_bits_per_dim_base = 3;

    auto param_json =
        vsag::JsonType::Parse(GeneratePyramidBuildParametersString("l2", dim, pyramid_param));
    param_json["index_param"]["tq_chain"].SetString("mrle, rabitq");
    param_json["index_param"]["mrle_dim"].SetInt(32);
    param_json["index_param"]["rabitq_bits_per_dim_precise"].SetInt(5);

    auto index = TestFactory("pyramid", param_json.Dump(), true);
    auto dataset = pool.GetDatasetAndCreate(dim, count, "l2", /*with_path=*/true);
    TestBuildIndex(index, dataset, true);
    auto query = fixtures::get_one_query(dataset->query_, 0);
    auto result = index->KnnSearch(query, 5, GeneratePyramidSearchParametersString(100));
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    REQUIRE(result.value()->GetDim() <= 5);
    REQUIRE_NOTHROW(index->GetStats());

    auto restored = TestFactory("pyramid", param_json.Dump(), true);
    TestSerializeBinarySet(
        index, restored, dataset, GeneratePyramidSearchParametersString(100), true);

    std::stringstream stream;
    REQUIRE(index->SerializeStreaming(stream).has_value());
    auto streamed = TestFactory("pyramid", param_json.Dump(), true);
    REQUIRE(streamed->DeserializeStreaming(stream).has_value());
    REQUIRE_NOTHROW(streamed->GetStats());
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Duplicate Path Semantics Same Path",
                             "[ft][build][pyramid]") {
    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {0, 1, 2};
    pyramid_param.support_duplicate = true;

    const auto param = GeneratePyramidBuildParametersString("l2", 4, pyramid_param);
    auto index = TestFactory("pyramid", param, true);

    const std::array<float, 4> shared_vector{1.0F, 2.0F, 3.0F, 4.0F};
    const std::array<float, 4> other_vector{4.0F, 3.0F, 2.0F, 1.0F};
    auto base = MakeDenseDataset(
        {shared_vector, shared_vector, other_vector}, {101, 102, 103}, {"a/d/f", "a/d/f", "b/e/g"});
    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());

    auto query = MakeSingleQuery(shared_vector, "a/d/f");
    auto search_result = index->KnnSearch(query, 3, GeneratePyramidSearchParametersString(20));
    REQUIRE(search_result.has_value());

    auto result = search_result.value();
    auto ids = CollectIds(result);
    REQUIRE(ids.count(101) == 1);
    REQUIRE(ids.count(102) == 1);
    REQUIRE(ids.count(103) == 0);
    RequireDistancesNearZero(result, {101, 102});
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Duplicate Path Semantics Prefix Descendant",
                             "[ft][build][pyramid]") {
    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {0, 1, 2};
    pyramid_param.support_duplicate = true;

    const auto param = GeneratePyramidBuildParametersString("l2", 4, pyramid_param);
    auto index = TestFactory("pyramid", param, true);

    const std::array<float, 4> shared_vector{1.0F, 2.0F, 3.0F, 4.0F};
    const std::array<float, 4> other_vector{4.0F, 3.0F, 2.0F, 1.0F};
    auto base = MakeDenseDataset(
        {shared_vector, shared_vector, other_vector}, {201, 202, 203}, {"a", "a/d/f", "b/e/g"});
    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());

    auto leaf_query = MakeSingleQuery(shared_vector, "a/d/f");
    auto leaf_result = index->KnnSearch(leaf_query, 3, GeneratePyramidSearchParametersString(20));
    REQUIRE(leaf_result.has_value());
    auto leaf_ids = CollectIds(leaf_result.value());
    REQUIRE(leaf_ids.count(202) == 1);
    REQUIRE(leaf_ids.count(201) == 0);
    REQUIRE(leaf_ids.count(203) == 0);
    RequireDistancesNearZero(leaf_result.value(), {202});

    auto prefix_query = MakeSingleQuery(shared_vector, "a");
    auto prefix_result =
        index->KnnSearch(prefix_query, 3, GeneratePyramidSearchParametersString(20));
    REQUIRE(prefix_result.has_value());
    auto prefix_ids = CollectIds(prefix_result.value());
    // Query path `a` only searches node `a`; descendants are visible only if they were folded into
    // node `a`'s duplicate group during insertion.
    REQUIRE(prefix_ids.count(202) == 1);
    REQUIRE(prefix_ids.count(201) == 0);
    REQUIRE(prefix_ids.count(203) == 0);
    RequireDistancesNearZero(prefix_result.value(), {202});
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Duplicate Path Semantics Shared Prefix Visibility",
                             "[ft][build][pyramid]") {
    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {0, 1, 2};
    pyramid_param.support_duplicate = true;

    const auto param = GeneratePyramidBuildParametersString("l2", 4, pyramid_param);
    auto index = TestFactory("pyramid", param, true);

    const std::array<float, 4> shared_vector{1.0F, 2.0F, 3.0F, 4.0F};
    const std::array<float, 4> other_vector{4.0F, 3.0F, 2.0F, 1.0F};
    auto base = MakeDenseDataset(
        {shared_vector, shared_vector, other_vector}, {301, 302, 303}, {"a/d/f", "a/d/g", "b/e/g"});
    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());

    auto query_adf = MakeSingleQuery(shared_vector, "a/d/f");
    auto result_adf = index->KnnSearch(query_adf, 3, GeneratePyramidSearchParametersString(20));
    REQUIRE(result_adf.has_value());
    auto ids_adf = CollectIds(result_adf.value());
    REQUIRE(ids_adf.count(301) == 1);
    REQUIRE(ids_adf.count(302) == 0);
    REQUIRE(ids_adf.count(303) == 0);
    RequireDistancesNearZero(result_adf.value(), {301});

    auto query_adg = MakeSingleQuery(shared_vector, "a/d/g");
    auto result_adg = index->KnnSearch(query_adg, 3, GeneratePyramidSearchParametersString(20));
    REQUIRE(result_adg.has_value());
    auto ids_adg = CollectIds(result_adg.value());
    REQUIRE(ids_adg.count(301) == 0);
    REQUIRE(ids_adg.count(302) == 1);
    REQUIRE(ids_adg.count(303) == 0);
    RequireDistancesNearZero(result_adg.value(), {302});

    auto query_ad = MakeSingleQuery(shared_vector, "a/d");
    auto result_ad = index->KnnSearch(query_ad, 3, GeneratePyramidSearchParametersString(20));
    REQUIRE(result_ad.has_value());
    auto ids_ad = CollectIds(result_ad.value());
    REQUIRE(ids_ad.count(301) == 1);
    REQUIRE(ids_ad.count(302) == 1);
    REQUIRE(ids_ad.count(303) == 0);
    RequireDistancesNearZero(result_ad.value(), {301, 302});
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Duplicate Path Semantics Negative Control",
                             "[ft][build][pyramid]") {
    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {0, 1, 2};
    pyramid_param.support_duplicate = true;

    const auto param = GeneratePyramidBuildParametersString("l2", 4, pyramid_param);
    auto index = TestFactory("pyramid", param, true);

    const std::array<float, 4> shared_vector{1.0F, 2.0F, 3.0F, 4.0F};
    auto base = MakeDenseDataset({shared_vector, shared_vector}, {401, 402}, {"a/d/f", "b/e/g"});
    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());

    auto query_adf = MakeSingleQuery(shared_vector, "a/d/f");
    auto result_adf = index->KnnSearch(query_adf, 2, GeneratePyramidSearchParametersString(20));
    REQUIRE(result_adf.has_value());
    auto ids_adf = CollectIds(result_adf.value());
    REQUIRE(ids_adf.count(401) == 1);
    REQUIRE(ids_adf.count(402) == 0);

    auto query_a = MakeSingleQuery(shared_vector, "a");
    auto result_a = index->KnnSearch(query_a, 2, GeneratePyramidSearchParametersString(20));
    REQUIRE(result_a.has_value());
    auto ids_a = CollectIds(result_a.value());
    REQUIRE(ids_a.count(402) == 0);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid KnnSearch Expands Ef Search To Topk",
                             "[ft][build][pyramid]") {
    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {};

    const auto param = GeneratePyramidBuildParametersString("l2", 4, pyramid_param);
    auto index = TestFactory("pyramid", param, true);

    constexpr int64_t data_count = 24;
    constexpr int64_t topk = 20;
    constexpr int64_t ef_search = 5;
    std::vector<std::array<float, 4>> vectors;
    std::vector<int64_t> ids;
    std::vector<std::string> paths;
    vectors.reserve(data_count);
    ids.reserve(data_count);
    paths.reserve(data_count);
    for (int64_t i = 0; i < data_count; ++i) {
        auto value = static_cast<float>(i);
        vectors.push_back({value, value + 1.0F, value + 2.0F, value + 3.0F});
        ids.push_back(1000 + i);
        paths.emplace_back("all");
    }

    auto base = MakeDenseDataset(vectors, ids, paths);
    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());

    auto query = MakeSingleQuery(vectors.front(), "all");
    auto search_result =
        index->KnnSearch(query, topk, GeneratePyramidSearchParametersString(ef_search));
    REQUIRE(search_result.has_value());
    REQUIRE(search_result.value()->GetDim() == topk);

    auto threshold_result =
        index->KnnSearch(query, topk, R"({"threshold": 0.0, "pyramid": {"ef_search": 5}})");
    REQUIRE(threshold_result.has_value());
    REQUIRE(threshold_result.value()->GetDim() == 1);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid threshold filters before flat-node pruning",
                             "[ft][search][pyramid][threshold][nonfinite]") {
    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {};
    const auto param = GeneratePyramidBuildParametersString("ip", 4, pyramid_param);
    auto index = TestFactory("pyramid", param, true);

    std::vector<std::array<float, 4>> vectors = {
        {std::numeric_limits<float>::max(), 0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F, 0.0F}};
    std::vector<int64_t> ids = {10, 20};
    std::vector<std::string> paths = {"all", "all"};
    REQUIRE(index->Build(MakeDenseDataset(vectors, ids, paths)).has_value());

    std::array<float, 4> query_vector = {std::numeric_limits<float>::max(), 0.0F, 0.0F, 0.0F};
    auto query = MakeSingleQuery(query_vector, "all");
    auto ordinary = index->KnnSearch(query, 2, R"({"pyramid":{"ef_search":2}})");
    REQUIRE(ordinary.has_value());
    REQUIRE(ordinary.value()->GetDim() == 2);
    CAPTURE(ordinary.value()->GetDistances()[0], ordinary.value()->GetDistances()[1]);
    auto result = index->KnnSearch(query, 1, R"({"threshold":1.0,"pyramid":{"ef_search":1}})");
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == 1);
    REQUIRE(result.value()->GetIds()[0] == 20);
    REQUIRE(result.value()->GetDistances()[0] == 1.0F);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Add Test",
                             "[ft][build][pyramid]") {
    auto metric_type = GENERATE("l2");
    std::string base_quantization_str = GENERATE("fp32");
    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {0, 1, 2};
    const std::string name = "pyramid";
    auto search_param = GeneratePyramidSearchParametersString(100);
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}", metric_type, dim));
        auto param = GeneratePyramidBuildParametersString(metric_type, dim, pyramid_param);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type, /*with_path=*/true);
        TestAddIndex(index, dataset, true);
        TestKnnSearch(index, dataset, search_param, 0.94, true);
        TestFilterSearch(index, dataset, search_param, 0.94, true);
        TestRangeSearch(index, dataset, search_param, 0.94, 10, true);
        TestRangeSearch(index, dataset, search_param, 0.49, 5, true);
        TestCalcDistanceById(index, dataset, 1e-5, true);
        TestMultiQueryBatchCalcDistanceById(index, dataset, 1e-5, true);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Multi-Levels Test",
                             "[ft][build][pyramid]") {
    auto metric_type = GENERATE("l2");
    std::string base_quantization_str = GENERATE("fp32");
    const std::string name = "pyramid";
    auto search_param = GeneratePyramidSearchParametersString(100);
    PyramidParam pyramid_param;
    for (auto& dim : dims) {
        for (const auto& level : levels) {
            INFO(fmt::format("metric_type={}, dim={}, no_build_levels={}",
                             metric_type,
                             dim,
                             fmt::join(level, ",")));
            pyramid_param.no_build_levels = level;
            auto param = GeneratePyramidBuildParametersString(metric_type, dim, pyramid_param);
            auto index = TestFactory(name, param, true);
            auto dataset =
                pool.GetDatasetAndCreate(dim, base_count, metric_type, /*with_path=*/true);
            TestContinueAdd(index, dataset, true);
            TestKnnSearch(index, dataset, search_param, 0.94, true);
            TestFilterSearch(index, dataset, search_param, 0.94, true);
            TestRangeSearch(index, dataset, search_param, 0.94, 10, true);
            TestCalcDistanceById(index, dataset, 1e-5, true);
            TestMultiQueryBatchCalcDistanceById(index, dataset, 1e-5, true);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid No Path Test",
                             "[ft][build][pyramid]") {
    auto metric_type = GENERATE("l2");
    std::string base_quantization_str = GENERATE("fp32");
    const std::string name = "pyramid";
    auto search_param = GeneratePyramidSearchParametersString(100);
    constexpr static const char* search_param_with_hops_limit =
        R"({"pyramid":{"ef_search":100,"hops_limit":200}})";
    constexpr static const char* search_param_invalid_hops_limit =
        R"({"pyramid":{"ef_search":100,"hops_limit":50}})";
    constexpr static const char* search_param_for_hops_statistics =
        R"({"pyramid":{"ef_search":1,"hops_limit":2}})";
    constexpr static const char* search_param_without_hops_limit_for_statistics =
        R"({"pyramid":{"ef_search":1}})";
    constexpr static const char* search_param_ignored_hops_limit_for_statistics =
        R"({"pyramid":{"ef_search":1,"hops_limit":1}})";
    PyramidParam pyramid_param;
    std::vector<std::vector<int>> tmp_levels = {{1, 2}, {0, 1, 2}};
    for (auto& dim : dims) {
        for (const auto& level : tmp_levels) {
            INFO(fmt::format("metric_type={}, dim={}, no_build_levels={}",
                             metric_type,
                             dim,
                             fmt::join(level, ",")));
            pyramid_param.no_build_levels = level;
            auto param = GeneratePyramidBuildParametersString(metric_type, dim, pyramid_param);
            auto index = TestFactory(name, param, true);
            auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
            auto tmp_paths = dataset->query_->GetPaths();
            dataset->query_->Paths(nullptr);
            TestContinueAdd(index, dataset, true);
            auto has_root = level[0] != 0;
            TestKnnSearch(index, dataset, search_param, 0.94, has_root);
            if (has_root) {
                TestKnnSearch(index, dataset, search_param_with_hops_limit, 0.9, true);
                TestKnnSearch(index, dataset, search_param_invalid_hops_limit, 0.94, true);

                auto query = fixtures::get_one_query(dataset->query_, 0);
                const auto unlimited_hops = SearchAndGetHops(
                    index, query, dataset->top_k, search_param_without_hops_limit_for_statistics);
                const auto limited_hops = SearchAndGetHops(
                    index, query, dataset->top_k, search_param_for_hops_statistics);
                const auto ignored_hops = SearchAndGetHops(
                    index, query, dataset->top_k, search_param_ignored_hops_limit_for_statistics);
                REQUIRE(unlimited_hops > limited_hops);
                REQUIRE(limited_hops <= 2);
                REQUIRE(ignored_hops == unlimited_hops);
            }
            TestFilterSearch(index, dataset, search_param, 0.94, has_root);
            TestRangeSearch(index, dataset, search_param, 0.94, 10, has_root);
            TestCalcDistanceById(index, dataset, 1e-5, true);
            TestMultiQueryBatchCalcDistanceById(index, dataset, 1e-5, true);
            dataset->query_->Paths(tmp_paths);
        }
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Serialize File",
                             "[ft][pyramid][serialization][serialize]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2");
    auto use_reorder = GENERATE(true, false);
    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {0, 1, 2};
    pyramid_param.use_reorder = use_reorder;
    if (use_reorder) {
        pyramid_param.base_quantization_type = "rabitq";
        pyramid_param.precise_quantization_type = "fp32";
    }
    const std::string name = "pyramid";
    auto search_param = GeneratePyramidSearchParametersString(100);
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}, use_reorder={}", metric_type, dim, use_reorder));
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GeneratePyramidBuildParametersString(metric_type, dim, pyramid_param);
        auto index = TestFactory(name, param, true);
        SECTION("serialize empty index") {
            auto index2 = TestFactory(name, param, true);
            auto serialize_binary = index->Serialize();
            REQUIRE(serialize_binary.has_value());
            auto deserialize_index = index2->Deserialize(serialize_binary.value());
            REQUIRE(deserialize_index.has_value());
        }
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type, /*with_path=*/true);
        TestBuildIndex(index, dataset, true);
        SECTION("serialize/deserialize by binary") {
            auto index2 = TestFactory(name, param, true);
            TestSerializeBinarySet(index, index2, dataset, search_param, true);
        }
        SECTION("serialize/deserialize by readerset") {
            auto index2 = TestFactory(name, param, true);
            TestSerializeReaderSet(index, index2, dataset, search_param, name, true);
        }
        SECTION("serialize/deserialize by file") {
            auto index2 = TestFactory(name, param, true);
            TestSerializeFile(index, index2, dataset, search_param, true);
        }
        SECTION("streaming serialize/deserialize/load") {
            auto index2 = TestFactory(name, param, true);
            std::stringstream stream;
            REQUIRE(index->SerializeStreaming(stream).has_value());
            const auto bytes = stream.str();

            std::stringstream deserialize_stream(bytes);
            REQUIRE(index2->DeserializeStreaming(deserialize_stream).has_value());
            TestKnnSearch(index2, dataset, search_param, 0.94, true);

            std::stringstream load_stream(bytes);
            auto loaded = vsag::Index::Load(load_stream, "{}");
            REQUIRE(loaded.has_value());
            TestKnnSearch(loaded.value(), dataset, search_param, 0.94, true);
        }
    }
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid multi-layer root streaming serialization",
                             "[ft][pyramid][root_graph][streaming]") {
    const auto graph_storage_type = GENERATE(std::string("flat"), std::string("compressed"));
    CAPTURE(graph_storage_type);
    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {1, 2};
    pyramid_param.root_graph_type = "multi_layer";
    pyramid_param.graph_storage_type = graph_storage_type;
    const auto param = GeneratePyramidBuildParametersString("l2", 16, pyramid_param);
    auto index = TestFactory("pyramid", param, true);
    auto dataset = pool.GetDatasetAndCreate(16, 1000, "l2", /*with_path=*/true);
    TestBuildIndex(index, dataset, true);
    const auto search_param = GeneratePyramidSearchParametersString(100);

    SECTION("binary set serialization") {
        auto binary_set = index->Serialize();
        REQUIRE(binary_set.has_value());
        auto restored = TestFactory("pyramid", param, true);
        REQUIRE(restored->Deserialize(binary_set.value()).has_value());
        TestKnnSearch(restored, dataset, search_param, 0.94, true);
    }

    SECTION("streaming serialization and load") {
        std::stringstream stream;
        REQUIRE(index->SerializeStreaming(stream).has_value());
        const auto bytes = stream.str();
        auto restored = TestFactory("pyramid", param, true);
        std::stringstream deserialize_stream(bytes);
        REQUIRE(restored->DeserializeStreaming(deserialize_stream).has_value());
        TestKnnSearch(restored, dataset, search_param, 0.94, true);

        std::stringstream load_stream(bytes);
        auto loaded = vsag::Index::Load(load_stream, "{}");
        REQUIRE(loaded.has_value());
        TestKnnSearch(loaded.value(), dataset, search_param, 0.94, true);
    }
}

TEST_CASE("Pyramid applies root graph type per hierarchy",
          "[ft][pyramid][root_graph][multi_hierarchy]") {
    constexpr int64_t dim = 16;
    constexpr int64_t count = 1000;
    std::vector<float> vectors(count * dim);
    std::vector<int64_t> ids(count);
    std::vector<std::string> paths(count, "");
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = i;
        vectors[i * dim] = static_cast<float>(i);
    }
    const std::string params = R"({
        "dtype":"float32","metric_type":"l2","dim":16,
        "index_param":{
            "base_quantization_type":"fp32","use_reorder":false,
            "graph_type":"nsw","max_degree":32,"ef_construction":100,
            "build_thread_count":4,"index_min_size":1,"no_build_levels":[],
            "hierarchies":[
                {"name":"single","root_graph_type":"single_layer"},
                {"name":"multi","root_graph_type":"multi_layer"}
            ]
        }
    })";
    auto index = vsag::Factory::CreateIndex("pyramid", params);
    REQUIRE(index.has_value());
    auto base = vsag::Dataset::Make()
                    ->NumElements(count)
                    ->Dim(dim)
                    ->Ids(ids.data())
                    ->Float32Vectors(vectors.data())
                    ->Paths("single", paths.data())
                    ->Paths("multi", paths.data())
                    ->Owner(false);
    REQUIRE(index.value()->Build(base).has_value());

    const auto stats = nlohmann::json::parse(index.value()->GetStats());
    REQUIRE(stats["root_graphs"]["single"]["root_graph_type"] == "single_layer");
    REQUIRE(stats["root_graphs"]["single"]["route_graph_count"] == 0);
    REQUIRE(stats["root_graphs"]["multi"]["root_graph_type"] == "multi_layer");
    REQUIRE(stats["root_graphs"]["multi"]["bottom_graph_node_count"] == count);
    REQUIRE(stats["root_graphs"]["multi"]["route_graph_count"].get<uint64_t>() > 0);

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dim)
                     ->Float32Vectors(vectors.data() + 321 * dim)
                     ->Owner(false);
    auto result = index.value()->KnnSearch(
        query, 10, R"({"pyramid":{"ef_search":100,"hierarchies":["multi"]}})");
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == 10);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid streaming compatibility",
                             "[ft][pyramid][serialization][streaming][compatibility]") {
    BlockSizeLimitGuard block_size_limit_guard(1024 * 1024 * 2);
    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {0, 1, 2};
    const auto param = GeneratePyramidBuildParametersString("l2", 16, pyramid_param);
    auto index = TestFactory("pyramid", param, true);
    auto dataset = pool.GetDatasetAndCreate(16, 200, "l2", /*with_path=*/true);
    TestBuildIndex(index, dataset, true);

    std::stringstream stream;
    REQUIRE(index->SerializeStreaming(stream).has_value());
    const auto bytes = stream.str();
    const auto expected_count = dataset->base_->GetNumElements();

    SECTION("skips unknown non-critical block") {
        auto mutated = InsertUnknownStreamingBlock(bytes, false);
        auto restored = TestFactory("pyramid", param, true);
        std::stringstream deserialize_stream(mutated);
        REQUIRE(restored->DeserializeStreaming(deserialize_stream).has_value());
        REQUIRE(restored->GetNumElements() == expected_count);

        std::stringstream load_stream(mutated);
        auto loaded = vsag::Index::Load(load_stream, "{}");
        REQUIRE(loaded.has_value());
        REQUIRE(loaded.value()->GetNumElements() == expected_count);
    }

    SECTION("rejects unknown critical block") {
        auto mutated = InsertUnknownStreamingBlock(bytes, true);
        auto restored = TestFactory("pyramid", param, true);
        std::stringstream deserialize_stream(mutated);
        REQUIRE_FALSE(restored->DeserializeStreaming(deserialize_stream).has_value());

        std::stringstream load_stream(mutated);
        REQUIRE_FALSE(vsag::Index::Load(load_stream, "{}").has_value());
    }

    SECTION("rejects missing required block") {
        auto mutated = EraseStreamingBlock(bytes, vsag::StreamSerializationTag::BASE_CODES);
        auto restored = TestFactory("pyramid", param, true);
        std::stringstream deserialize_stream(mutated);
        REQUIRE_FALSE(restored->DeserializeStreaming(deserialize_stream).has_value());

        std::stringstream load_stream(mutated);
        REQUIRE_FALSE(vsag::Index::Load(load_stream, "{}").has_value());
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex, "Pyramid Clone", "[ft][clone][pyramid]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2");
    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {0, 1, 2};
    const std::string name = "pyramid";
    auto search_param = GeneratePyramidSearchParametersString(100);
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}", metric_type, dim));
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GeneratePyramidBuildParametersString(metric_type, dim, pyramid_param);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type, /*with_path=*/true);
        TestBuildIndex(index, dataset, true);
        TestClone(index, dataset, search_param);
    }
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Export Model",
                             "[ft][export][pyramid]") {
    auto size = GENERATE(1024 * 1024 * 2);
    BlockSizeLimitGuard block_size_limit_guard(size);
    auto metric_type = GENERATE("l2");
    auto use_reorder = GENERATE(true, false);
    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {0, 1, 2};
    pyramid_param.use_reorder = use_reorder;
    if (use_reorder) {
        pyramid_param.base_quantization_type = "rabitq";
        pyramid_param.precise_quantization_type = "fp32";
    }
    const std::string name = "pyramid";
    auto search_param = GeneratePyramidSearchParametersString(100);
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}, use_reorder={}", metric_type, dim, use_reorder));
        auto param = GeneratePyramidBuildParametersString(metric_type, dim, pyramid_param);
        auto index = TestFactory(name, param, true);
        auto index2 = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type, /*with_path=*/true);
        TestBuildIndex(index, dataset, true);
        TestExportModel(index, index2, dataset, search_param);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Build Test With Random Allocator",
                             "[ft][build][pyramid]") {
    auto allocator = std::make_shared<fixtures::RandomAllocator>();
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {0, 1, 2};
    const std::string name = "pyramid";
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}", metric_type, dim));
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GeneratePyramidBuildParametersString(metric_type, dim, pyramid_param);
        auto index = vsag::Factory::CreateIndex(name, param, allocator.get());
        if (not index.has_value()) {
            continue;
        }
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type, /*with_path=*/true);
        TestContinueAddIgnoreRequire(index.value(), dataset, 1);
        vsag::Options::Instance().set_block_size_limit(origin_size);
    }
}
TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Concurrent Test",
                             "[ft][concurrent][pyramid][build]") {
    auto metric_type = GENERATE("l2");
    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {0, 1};
    const std::string name = "pyramid";
    auto search_param = GeneratePyramidSearchParametersString(100);
    for (auto& dim : dims) {
        auto param = GeneratePyramidBuildParametersString(metric_type, dim, pyramid_param);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type, /*with_path=*/true);
        TestConcurrentAdd(index, dataset, true);
        TestConcurrentKnnSearch(index, dataset, search_param, 0.94, true);
        TestCalcDistanceById(index, dataset, 1e-5, true);
        TestMultiQueryBatchCalcDistanceById(index, dataset, 1e-5, true);
    }
    for (auto& dim : dims) {
        auto param = GeneratePyramidBuildParametersString(metric_type, dim, pyramid_param);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type, /*with_path=*/true);
        TestConcurrentAddSearch(index, dataset, search_param, 0.94, true);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid OverTime Test",
                             "[ft][search][pyramid]") {
    auto metric_type = GENERATE("l2");
    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {0, 1};
    const std::string name = "pyramid";
    auto search_param = GeneratePyramidSearchParametersString(100, 20);
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}", metric_type, dim));
        auto param = GeneratePyramidBuildParametersString(metric_type, dim, pyramid_param);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type, /*with_path=*/true);
        TestContinueAdd(index, dataset, true);
        TestSearchOvertime(index, dataset, search_param);
        auto timeout_search_param = GeneratePyramidSearchParametersString(100, 0.0F);

        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(dataset->query_->GetFloat32Vectors())
            ->Paths(dataset->query_->GetPaths())
            ->Owner(false);
        auto res = index->KnnSearch(query, 10, timeout_search_param);
        REQUIRE(res.has_value());
        auto result = res.value();
        REQUIRE(result->GetStatistics() != "{}");
        auto stats = result->GetStatistics({"is_timeout"});
        REQUIRE(stats.size() == 1);
        bool is_timeout = stats[0] == "true";
        REQUIRE(is_timeout);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Duplicate Test",
                             "[ft][concurrent][pyramid][build][duplicate]") {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto metric_type = GENERATE("l2", "cosine");
    auto size = GENERATE(1024 * 1024 * 2);
    auto name = "pyramid";
    auto duplicate_pos = GENERATE("prefix", "suffix", "middle");
    auto search_param = GeneratePyramidSearchParametersString(100);
    std::unordered_map<std::string, float> ratios{
        {"prefix", 0.9}, {"suffix", 0.9}, {"middle", 1.0}};
    auto recall = 0.98F;
    PyramidParam pyramid_param;
    pyramid_param.support_duplicate = true;
    for (auto& dim : dims) {
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GeneratePyramidBuildParametersString(metric_type, dim, pyramid_param);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type, /*with_path=*/true);
        TestIndex::TestBuildDuplicateIndex(index, dataset, duplicate_pos, true);
        TestIndex::TestKnnSearch(index, dataset, search_param, recall, true);
        TestIndex::TestConcurrentKnnSearch(index, dataset, search_param, recall, true);
        TestIndex::TestRangeSearch(index, dataset, search_param, recall, 10, true);
        TestIndex::TestRangeSearch(index, dataset, search_param, recall / 2.0, 5, true);
        TestIndex::TestFilterSearch(index, dataset, search_param, recall, true, true);
        auto index2 = TestIndex::TestFactory(name, param, true);
        TestIndex::TestSerializeFile(index, index2, dataset, search_param, true);

        // query duplicate data
        if (duplicate_pos != std::string("middle")) {
            auto duplicate_data = vsag::Dataset::Make();
            duplicate_data->NumElements(1)
                ->Dim(dataset->base_->GetDim())
                ->SparseVectors(dataset->base_->GetSparseVectors())
                ->Paths(dataset->base_->GetPaths())
                ->Float32Vectors(dataset->base_->GetFloat32Vectors())
                ->Owner(false);
            auto result = index->KnnSearch(duplicate_data, 10, search_param).value();
            REQUIRE(result->GetDim() == 10);
            for (size_t i = 0; i < result->GetDim(); ++i) {
                auto distance = result->GetDistances()[i];
                REQUIRE(std::abs(distance) <= 2e-6);
            }
        }
        vsag::Options::Instance().set_block_size_limit(origin_size);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Duplicate ID Test",
                             "[ft][build][pyramid][duplicate]") {
    auto metric_type = GENERATE("l2");
    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {0, 1};
    const std::string name = "pyramid";
    auto search_param = GeneratePyramidSearchParametersString(100, 20);
    for (auto& dim : dims) {
        auto param = GeneratePyramidBuildParametersString(metric_type, dim, pyramid_param);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type, /*with_path=*/true);
        TestDuplicateAdd(index, dataset);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Analyzer Test",
                             "[ft][pyramid][analyzer][build]") {
    auto metric_type = GENERATE("l2");
    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {0, 1, 2};
    const std::string name = "pyramid";
    auto search_param = GeneratePyramidSearchParametersString(100);
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}", metric_type, dim));
        auto param = GeneratePyramidBuildParametersString(metric_type, dim, pyramid_param);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type, /*with_path=*/true);
        TestBuildIndex(index, dataset, true);

        auto stats_str = index->GetStats();
        REQUIRE(!stats_str.empty());

        auto stats = nlohmann::json::parse(stats_str);
        REQUIRE(stats.contains("total_count"));
        REQUIRE(stats["total_count"].get<int64_t>() == base_count);

        REQUIRE(stats.contains("index_node_structure"));
        REQUIRE(stats.contains("leaf_node_size_distribution"));
        REQUIRE(stats.contains("subindex_quality"));
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid AnalyzeIndexBySearch honors paths and removals",
                             "[ft][pyramid][analyzer]") {
    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {0, 1, 2};

    auto index =
        TestFactory("pyramid", GeneratePyramidBuildParametersString("l2", 4, pyramid_param), true);
    auto base = MakeDenseDataset({{{10.0F, 0.0F, 0.0F, 0.0F}},
                                  {{1.0F, 0.0F, 0.0F, 0.0F}},
                                  {{0.0F, 0.0F, 0.0F, 0.0F}},
                                  {{100.0F, 0.0F, 0.0F, 0.0F}}},
                                 {101, 102, 201, 202},
                                 {"root/a/leaf", "root/a/leaf", "root/b/leaf", "root/b/leaf"});
    REQUIRE(index->Build(base).has_value());

    // The first query has an exact match in the other path. A global ground truth would therefore
    // report only 50% recall for this batch, while path-scoped ground truth reports 100%.
    auto query = MakeDenseDataset({{{0.0F, 0.0F, 0.0F, 0.0F}}, {{100.0F, 0.0F, 0.0F, 0.0F}}},
                                  {0, 1},
                                  {"root/a/leaf", "root/b/leaf"});
    vsag::SearchRequest request;
    request.query_ = query;
    request.topk_ = 1;
    request.params_str_ = GeneratePyramidSearchParametersString(20);

    const auto analyze = [&]() {
        const auto stats_string = index->AnalyzeIndexBySearch(request);
        REQUIRE_FALSE(stats_string.empty());
        auto stats = nlohmann::json::parse(stats_string);
        REQUIRE(stats.contains("avg_distance_query"));
        REQUIRE(stats.contains("recall_query"));
        REQUIRE(stats.contains("time_cost_query"));
        return stats;
    };

    const auto stats_before_remove = analyze();
    REQUIRE(std::abs(stats_before_remove["recall_query"].get<float>() - 1.0F) <= 1e-6F);
    REQUIRE(std::abs(stats_before_remove["avg_distance_query"].get<float>() - 0.5F) <= 1e-6F);

    auto remove_result = index->Remove(std::vector<int64_t>{102}, vsag::RemoveMode::MARK_REMOVE);
    REQUIRE(remove_result.has_value());
    REQUIRE(remove_result.value() == 1);

    // The removed vector was the nearest live candidate in root/a/leaf before removal. Ground
    // truth must now use id 101, just as the actual path-scoped search does.
    const auto stats_after_remove = analyze();
    REQUIRE(std::abs(stats_after_remove["recall_query"].get<float>() - 1.0F) <= 1e-6F);
    REQUIRE(std::abs(stats_after_remove["avg_distance_query"].get<float>() - 50.0F) <= 1e-6F);
}

// ============================================================================
// Multi-Hierarchy Tests
// ============================================================================

namespace {

struct MultiHierarchyFixture {
    static constexpr int64_t NUM = 4;
    static constexpr int64_t DIM = 4;

    std::vector<float> vectors = {
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    std::vector<int64_t> ids = {100, 101, 102, 103};
    std::vector<std::string> site_paths = {"www/news", "www/sports", "www/news", "www/sports"};
    std::vector<std::string> cat_paths = {"tech/ai", "tech/web", "science/bio", "science/phys"};

    vsag::DatasetPtr
    make_base() {
        auto* rv = new float[NUM * DIM];
        auto* ri = new int64_t[NUM];
        auto* rs = new std::string[NUM];
        auto* rc = new std::string[NUM];
        std::copy(vectors.begin(), vectors.end(), rv);
        std::copy(ids.begin(), ids.end(), ri);
        for (int i = 0; i < NUM; ++i) {
            rs[i] = site_paths[i];
            rc[i] = cat_paths[i];
        }
        auto ds = vsag::Dataset::Make();
        ds->NumElements(NUM)
            ->Dim(DIM)
            ->Float32Vectors(rv)
            ->Ids(ri)
            ->Paths("site", rs)
            ->Paths("cat", rc)
            ->Owner(true);
        return ds;
    }

    vsag::DatasetPtr
    make_query(const std::string& hierarchy, const std::string& path) {
        auto* qv = new float[DIM]{1.0f, 0.0f, 0.0f, 0.0f};
        auto* qp = new std::string[1]{path};
        auto ds = vsag::Dataset::Make();
        ds->NumElements(1)->Dim(DIM)->Float32Vectors(qv)->Paths(hierarchy, qp)->Owner(true);
        return ds;
    }

    std::set<int64_t>
    search_ids(const std::shared_ptr<vsag::Index>& index,
               const std::string& hierarchy,
               const std::string& path,
               int64_t k = 4) {
        auto query = make_query(hierarchy, path);
        std::string sp =
            R"({"pyramid": {"ef_search": 100, "hierarchies": [")" + hierarchy + R"("]}})";
        auto result = index->KnnSearch(query, k, sp);
        REQUIRE(result.has_value());
        auto* rids = result.value()->GetIds();
        auto cnt = result.value()->GetDim();
        return std::set<int64_t>(rids, rids + cnt);
    }

    static std::string
    build_param(const std::string& graph_type = "nsw", bool use_reorder = false) {
        std::string reorder_str = use_reorder ? "true" : "false";
        std::string precise = use_reorder ? R"(, "precise_quantization_type": "fp32")" : "";
        std::string base_q = use_reorder ? "rabitq" : "fp32";
        return R"({
            "dtype": "float32", "metric_type": "l2", "dim": 4,
            "index_param": {
                "max_degree": 32, "alpha": 1.2,
                "graph_type": ")" +
               graph_type + R"(",
                "graph_iter_turn": 15, "neighbor_sample_rate": 0.2,
                "base_quantization_type": ")" +
               base_q + R"(",
                "use_reorder": )" +
               reorder_str + precise + R"(,
                "index_min_size": 0, "support_duplicate": false,
                "hierarchies": [
                    {"name": "site", "no_build_levels": [0]},
                    {"name": "cat", "no_build_levels": [0, 1]}
                ]
            }
        })";
    }

    static std::string
    build_reorder_stats_param(const std::string& precise_file_path) {
        return R"({
            "dtype": "float32", "metric_type": "l2", "dim": 4,
            "index_param": {
                "max_degree": 32, "alpha": 1.2,
                "graph_type": "nsw",
                "graph_iter_turn": 15, "neighbor_sample_rate": 0.2,
                "base_quantization_type": "rabitq",
                "base_io_type": "memory_io",
                "use_reorder": true,
                "precise_quantization_type": "fp32",
                "precise_io_type": "buffer_io",
                "precise_file_path": ")" +
               precise_file_path + R"(",
                "index_min_size": 0, "support_duplicate": false,
                "hierarchies": [
                    {"name": "site", "no_build_levels": [0]},
                    {"name": "cat", "no_build_levels": [0, 1]}
                ]
            }
        })";
    }
};

struct MultiPathFixture {
    static constexpr int64_t NUM = 4;
    static constexpr int64_t DIM = 4;

    static vsag::DatasetPtr
    make_base() {
        auto* vectors = new float[NUM * DIM]{1.0F,
                                             0.0F,
                                             0.0F,
                                             0.0F,
                                             0.0F,
                                             1.0F,
                                             0.0F,
                                             0.0F,
                                             0.0F,
                                             0.0F,
                                             1.0F,
                                             0.0F,
                                             0.0F,
                                             0.0F,
                                             0.0F,
                                             1.0F};
        auto* ids = new int64_t[NUM]{10, 11, 12, 13};
        auto* host_paths = new std::string[NUM]{"host/one", "host/two", "host/three", "host/four"};
        auto dataset = vsag::Dataset::Make();
        dataset->NumElements(NUM)
            ->Dim(DIM)
            ->Float32Vectors(vectors)
            ->Ids(ids)
            ->Paths("host", host_paths)
            ->Paths("tag",
                    std::vector<std::vector<std::string>>{
                        {"a/x", "a/y", "a/x", "literal|pipe"}, {"a/x"}, {"other"}, {""}})
            ->Owner(true);
        return dataset;
    }

    static vsag::DatasetPtr
    make_legacy_query(const std::string& hierarchy_name, const std::string& path) {
        auto* vector = new float[DIM]{1.0F, 0.0F, 0.0F, 0.0F};
        auto* paths = new std::string[1]{path};
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(DIM)
            ->Float32Vectors(vector)
            ->Paths(hierarchy_name, paths)
            ->Owner(true);
        return query;
    }

    static vsag::DatasetPtr
    make_structured_query(std::vector<std::string> paths) {
        auto* vector = new float[DIM]{1.0F, 0.0F, 0.0F, 0.0F};
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(DIM)
            ->Float32Vectors(vector)
            ->Paths("tag", std::vector<std::vector<std::string>>{std::move(paths)})
            ->Owner(true);
        return query;
    }

    static std::vector<int64_t>
    search(const std::shared_ptr<vsag::Index>& index,
           const vsag::DatasetPtr& query,
           const std::string& hierarchy_name,
           int64_t parallelism = 1) {
        auto search_param = fmt::format(
            R"({{"pyramid": {{"ef_search": 100, "parallelism": {}, "hierarchies": ["{}"]}}}})",
            parallelism,
            hierarchy_name);
        auto result = index->KnnSearch(query, NUM, search_param);
        REQUIRE(result.has_value());
        std::vector<int64_t> ids;
        const auto count = result.value()->GetDim();
        if (count > 0) {
            ids.assign(result.value()->GetIds(), result.value()->GetIds() + count);
        }
        std::sort(ids.begin(), ids.end());
        return ids;
    }

    static std::string
    build_param(const std::string& graph_type, bool skip_tag_prefix = false) {
        const std::string tag_no_build_levels = skip_tag_prefix ? "[0, 1]" : "[]";
        return R"({
            "dtype": "float32", "metric_type": "l2", "dim": 4,
            "index_param": {
                "max_degree": 16, "alpha": 1.2,
                "graph_type": ")" +
               graph_type + R"(",
                "graph_iter_turn": 15, "neighbor_sample_rate": 0.2,
                "base_quantization_type": "fp32", "use_reorder": false,
                "index_min_size": 0, "support_duplicate": false,
                "hierarchies": [
                    {"name": "tag", "no_build_levels": )" +
               tag_no_build_levels + R"(},
                    {"name": "host", "no_build_levels": []}
                ]
            }
        })";
    }
};

}  // namespace

TEST_CASE("Multi-Hierarchy: NSW Build and Search", "[ft][pyramid][multi_hierarchy]") {
    MultiHierarchyFixture f;
    auto index = vsag::Factory::CreateIndex("pyramid", f.build_param("nsw"));
    REQUIRE(index.has_value());

    auto build_result = index.value()->Build(f.make_base());
    REQUIRE(build_result.has_value());

    // site hierarchy: "www/news" -> ids 100, 102
    auto site_news = f.search_ids(index.value(), "site", "www/news");
    REQUIRE(site_news.count(100) == 1);
    REQUIRE(site_news.count(102) == 1);
    REQUIRE(site_news.count(101) == 0);
    REQUIRE(site_news.count(103) == 0);

    // site hierarchy: "www/sports" -> ids 101, 103
    auto site_sports = f.search_ids(index.value(), "site", "www/sports");
    REQUIRE(site_sports.count(101) == 1);
    REQUIRE(site_sports.count(103) == 1);
    REQUIRE(site_sports.count(100) == 0);

    // cat hierarchy: "tech" -> ids 100, 101
    auto cat_tech = f.search_ids(index.value(), "cat", "tech");
    REQUIRE(cat_tech.count(100) == 1);
    REQUIRE(cat_tech.count(101) == 1);
    REQUIRE(cat_tech.count(102) == 0);

    // cat hierarchy: "science" -> ids 102, 103
    auto cat_science = f.search_ids(index.value(), "cat", "science");
    REQUIRE(cat_science.count(102) == 1);
    REQUIRE(cat_science.count(103) == 1);
    REQUIRE(cat_science.count(100) == 0);
}

TEST_CASE("Multi-Hierarchy: ODescent Build and Search", "[ft][pyramid][multi_hierarchy]") {
    MultiHierarchyFixture f;
    auto index = vsag::Factory::CreateIndex("pyramid", f.build_param("odescent"));
    REQUIRE(index.has_value());

    auto build_result = index.value()->Build(f.make_base());
    REQUIRE(build_result.has_value());

    auto site_news = f.search_ids(index.value(), "site", "www/news");
    REQUIRE(site_news.count(100) == 1);
    REQUIRE(site_news.count(102) == 1);

    auto cat_tech = f.search_ids(index.value(), "cat", "tech");
    REQUIRE(cat_tech.count(100) == 1);
    REQUIRE(cat_tech.count(101) == 1);
}

TEST_CASE("Multi-Hierarchy: Add after Build", "[ft][pyramid][multi_hierarchy]") {
    MultiHierarchyFixture f;
    auto index = vsag::Factory::CreateIndex("pyramid", f.build_param("nsw"));
    REQUIRE(index.has_value());

    // Build with first 2 vectors
    auto* rv = new float[8];
    auto* ri = new int64_t[2]{100, 101};
    auto* rs = new std::string[2]{"www/news", "www/sports"};
    auto* rc = new std::string[2]{"tech/ai", "tech/web"};
    std::copy(f.vectors.begin(), f.vectors.begin() + 8, rv);
    auto base1 = vsag::Dataset::Make();
    base1->NumElements(2)
        ->Dim(4)
        ->Float32Vectors(rv)
        ->Ids(ri)
        ->Paths("site", rs)
        ->Paths("cat", rc)
        ->Owner(true);
    REQUIRE(index.value()->Build(base1).has_value());

    // Add 2 more vectors
    auto* rv2 = new float[8];
    auto* ri2 = new int64_t[2]{102, 103};
    auto* rs2 = new std::string[2]{"www/news", "www/sports"};
    auto* rc2 = new std::string[2]{"science/bio", "science/phys"};
    std::copy(f.vectors.begin() + 8, f.vectors.end(), rv2);
    auto base2 = vsag::Dataset::Make();
    base2->NumElements(2)
        ->Dim(4)
        ->Float32Vectors(rv2)
        ->Ids(ri2)
        ->Paths("site", rs2)
        ->Paths("cat", rc2)
        ->Owner(true);
    REQUIRE(index.value()->Add(base2).has_value());

    // Verify all 4 vectors are searchable
    auto site_news = f.search_ids(index.value(), "site", "www/news");
    REQUIRE(site_news.count(100) == 1);
    REQUIRE(site_news.count(102) == 1);

    auto cat_science = f.search_ids(index.value(), "cat", "science");
    REQUIRE(cat_science.count(102) == 1);
    REQUIRE(cat_science.count(103) == 1);
}

TEST_CASE("Multi-Hierarchy: Serialize and Deserialize", "[ft][pyramid][multi_hierarchy]") {
    MultiHierarchyFixture f;
    auto param_str = f.build_param("nsw");
    auto index = vsag::Factory::CreateIndex("pyramid", param_str);
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(f.make_base()).has_value());

    // Serialize
    auto bs = index.value()->Serialize();
    REQUIRE(bs.has_value());

    // Deserialize into new index
    auto index2 = vsag::Factory::CreateIndex("pyramid", param_str);
    REQUIRE(index2.has_value());
    index2.value()->Deserialize(bs.value());

    // Search deserialized index
    auto site_news = f.search_ids(index2.value(), "site", "www/news");
    REQUIRE(site_news.count(100) == 1);
    REQUIRE(site_news.count(102) == 1);
    REQUIRE(site_news.count(101) == 0);

    auto cat_tech = f.search_ids(index2.value(), "cat", "tech");
    REQUIRE(cat_tech.count(100) == 1);
    REQUIRE(cat_tech.count(101) == 1);
    REQUIRE(cat_tech.count(102) == 0);
}

TEST_CASE("Multi-Hierarchy: Multiple paths per vector",
          "[ft][pyramid][multi_hierarchy][multi_path]") {
    const std::string graph_type = GENERATE("nsw", "odescent");
    CAPTURE(graph_type);
    auto index = vsag::Factory::CreateIndex("pyramid", MultiPathFixture::build_param(graph_type));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(MultiPathFixture::make_base()).has_value());
    REQUIRE(index.value()->GetNumElements() == MultiPathFixture::NUM);

    REQUIRE(MultiPathFixture::search(index.value(),
                                     MultiPathFixture::make_legacy_query("tag", "a/x"),
                                     "tag") == std::vector<int64_t>{10, 11});
    REQUIRE(MultiPathFixture::search(index.value(),
                                     MultiPathFixture::make_legacy_query("tag", "a/y"),
                                     "tag") == std::vector<int64_t>{10});
    REQUIRE(MultiPathFixture::search(index.value(),
                                     MultiPathFixture::make_legacy_query("tag", "a"),
                                     "tag") == std::vector<int64_t>{10, 11});

    for (const auto parallelism : {1, 2}) {
        CAPTURE(parallelism);
        REQUIRE(MultiPathFixture::search(index.value(),
                                         MultiPathFixture::make_legacy_query("tag", "a/x|a/y"),
                                         "tag",
                                         parallelism) == std::vector<int64_t>{10, 11});
    }
    REQUIRE(MultiPathFixture::search(index.value(),
                                     MultiPathFixture::make_structured_query({"a/x", "a/y"}),
                                     "tag") == std::vector<int64_t>{10, 11});
    REQUIRE(MultiPathFixture::search(index.value(),
                                     MultiPathFixture::make_structured_query({"literal|pipe"}),
                                     "tag") == std::vector<int64_t>{10});

    REQUIRE(MultiPathFixture::search(index.value(),
                                     MultiPathFixture::make_structured_query({""}),
                                     "tag") == std::vector<int64_t>{10, 11, 12, 13});

    REQUIRE(MultiPathFixture::search(index.value(),
                                     MultiPathFixture::make_legacy_query("host", "host/three"),
                                     "host") == std::vector<int64_t>{12});
}

TEST_CASE("Multi-Hierarchy: Multiple paths survive serialization",
          "[ft][pyramid][multi_hierarchy][multi_path][serialization]") {
    const auto param = MultiPathFixture::build_param("nsw");
    auto index = vsag::Factory::CreateIndex("pyramid", param);
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(MultiPathFixture::make_base()).has_value());

    auto binary_set = index.value()->Serialize();
    REQUIRE(binary_set.has_value());
    auto restored = vsag::Factory::CreateIndex("pyramid", param);
    REQUIRE(restored.has_value());
    REQUIRE(restored.value()->Deserialize(binary_set.value()).has_value());

    REQUIRE(MultiPathFixture::search(restored.value(),
                                     MultiPathFixture::make_legacy_query("tag", "a/x"),
                                     "tag") == std::vector<int64_t>{10, 11});
    REQUIRE(MultiPathFixture::search(restored.value(),
                                     MultiPathFixture::make_structured_query({"literal|pipe"}),
                                     "tag") == std::vector<int64_t>{10});
}

TEST_CASE("Multi-Hierarchy: Multiple paths below an unbuilt prefix are deduplicated",
          "[ft][pyramid][multi_hierarchy][multi_path]") {
    const std::string graph_type = GENERATE("nsw", "odescent");
    CAPTURE(graph_type);
    auto index =
        vsag::Factory::CreateIndex("pyramid", MultiPathFixture::build_param(graph_type, true));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(MultiPathFixture::make_base()).has_value());

    for (const auto parallelism : {1, 2}) {
        CAPTURE(parallelism);
        REQUIRE(MultiPathFixture::search(index.value(),
                                         MultiPathFixture::make_legacy_query("tag", "a"),
                                         "tag",
                                         parallelism) == std::vector<int64_t>{10, 11});
    }
}

TEST_CASE("Multi-Hierarchy: Add accepts multiple paths per vector",
          "[ft][pyramid][multi_hierarchy][multi_path]") {
    auto index = vsag::Factory::CreateIndex("pyramid", MultiPathFixture::build_param("nsw"));
    REQUIRE(index.has_value());

    auto* base_vector = new float[4]{1.0F, 0.0F, 0.0F, 0.0F};
    auto* base_id = new int64_t[1]{10};
    auto* base_path = new std::string[1]{"seed"};
    auto base = vsag::Dataset::Make();
    base->NumElements(1)
        ->Dim(4)
        ->Float32Vectors(base_vector)
        ->Ids(base_id)
        ->Paths("tag", base_path)
        ->Owner(true);
    REQUIRE(index.value()->Build(base).has_value());

    auto* add_vectors = new float[8]{0.9F, 0.1F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F};
    auto* add_ids = new int64_t[2]{10, 20};
    auto add = vsag::Dataset::Make();
    add->NumElements(2)
        ->Dim(4)
        ->Float32Vectors(add_vectors)
        ->Ids(add_ids)
        ->Paths("tag", std::vector<std::vector<std::string>>{{"ignored"}, {"b/x", "b/y"}})
        ->Owner(true);
    auto add_result = index.value()->Add(add);
    REQUIRE(add_result.has_value());
    REQUIRE(add_result.value() == std::vector<int64_t>{10});

    REQUIRE(MultiPathFixture::search(index.value(),
                                     MultiPathFixture::make_legacy_query("tag", "b/x"),
                                     "tag") == std::vector<int64_t>{20});
    REQUIRE(MultiPathFixture::search(index.value(),
                                     MultiPathFixture::make_legacy_query("tag", "b/y"),
                                     "tag") == std::vector<int64_t>{20});
    REQUIRE(MultiPathFixture::search(
                index.value(), MultiPathFixture::make_legacy_query("tag", "ignored"), "tag")
                .empty());
    REQUIRE(MultiPathFixture::search(index.value(),
                                     MultiPathFixture::make_structured_query({""}),
                                     "tag") == std::vector<int64_t>{10, 20});
}

TEST_CASE("Multi-Hierarchy: Different no_build_levels per hierarchy",
          "[ft][pyramid][multi_hierarchy]") {
    MultiHierarchyFixture f;
    auto index = vsag::Factory::CreateIndex("pyramid", f.build_param("nsw"));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(f.make_base()).has_value());

    // Both hierarchies should still return correct results
    auto site_news = f.search_ids(index.value(), "site", "www/news");
    REQUIRE(site_news.count(100) == 1);
    REQUIRE(site_news.count(102) == 1);

    // cat/tech/ai should find id 100
    auto cat_ai = f.search_ids(index.value(), "cat", "tech/ai");
    REQUIRE(cat_ai.count(100) == 1);
}

TEST_CASE("Multi-Hierarchy: Shared vector base distances", "[ft][pyramid][multi_hierarchy]") {
    MultiHierarchyFixture f;
    auto index = vsag::Factory::CreateIndex("pyramid", f.build_param("nsw"));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(f.make_base()).has_value());

    // id=100 has vector [1,0,0,0]. Query with same vector -> distance = 0.
    auto* qv1 = new float[4]{1.0f, 0.0f, 0.0f, 0.0f};
    auto* qp1 = new std::string[1]{"www/news"};
    auto q_site = vsag::Dataset::Make();
    q_site->NumElements(1)->Dim(4)->Float32Vectors(qv1)->Paths("site", qp1)->Owner(true);
    std::string sp_site = R"({"pyramid": {"ef_search": 100, "hierarchies": ["site"]}})";
    auto r_site = index.value()->KnnSearch(q_site, 4, sp_site);
    REQUIRE(r_site.has_value());

    auto* qv2 = new float[4]{1.0f, 0.0f, 0.0f, 0.0f};
    auto* qp2 = new std::string[1]{"tech"};
    auto q_cat = vsag::Dataset::Make();
    q_cat->NumElements(1)->Dim(4)->Float32Vectors(qv2)->Paths("cat", qp2)->Owner(true);
    std::string sp_cat = R"({"pyramid": {"ef_search": 100, "hierarchies": ["cat"]}})";
    auto r_cat = index.value()->KnnSearch(q_cat, 4, sp_cat);
    REQUIRE(r_cat.has_value());

    // Find distance for id=100 in both results — should be ~0 and identical
    float dist_site = -1, dist_cat = -1;
    for (int64_t i = 0; i < r_site.value()->GetDim(); ++i) {
        if (r_site.value()->GetIds()[i] == 100) {
            dist_site = r_site.value()->GetDistances()[i];
        }
    }
    for (int64_t i = 0; i < r_cat.value()->GetDim(); ++i) {
        if (r_cat.value()->GetIds()[i] == 100) {
            dist_cat = r_cat.value()->GetDistances()[i];
        }
    }
    REQUIRE(dist_site >= 0);
    REQUIRE(dist_cat >= 0);
    REQUIRE(std::abs(dist_site - dist_cat) < 1e-6f);
    REQUIRE(std::abs(dist_site) < 1e-6f);
}

TEST_CASE("Multi-Hierarchy: Partial paths - only some hierarchies provided",
          "[ft][pyramid][multi_hierarchy]") {
    MultiHierarchyFixture f;
    auto index = vsag::Factory::CreateIndex("pyramid", f.build_param("nsw"));
    REQUIRE(index.has_value());

    // Build with only "site" paths, no "cat" paths
    auto* rv = new float[4]{1.0f, 0.0f, 0.0f, 0.0f};
    auto* ri = new int64_t[1]{100};
    auto* rs = new std::string[1]{"www/news"};
    auto base = vsag::Dataset::Make();
    base->NumElements(1)->Dim(4)->Float32Vectors(rv)->Ids(ri)->Paths("site", rs)->Owner(true);

    auto result = index.value()->Build(base);
    REQUIRE(result.has_value());

    // Search in "site" hierarchy -> should find id=100
    auto site_result = f.search_ids(index.value(), "site", "www/news");
    REQUIRE(site_result.count(100) == 1);

    // Search in "cat" hierarchy -> should NOT find id=100
    auto* qv = new float[4]{1.0f, 0.0f, 0.0f, 0.0f};
    auto* qp = new std::string[1]{"tech"};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(4)->Float32Vectors(qv)->Paths("cat", qp)->Owner(true);
    std::string sp = R"({"pyramid": {"ef_search": 100, "hierarchies": ["cat"]}})";
    auto cat_result = index.value()->KnnSearch(query, 4, sp);
    REQUIRE(cat_result.has_value());
    REQUIRE(cat_result.value()->GetDim() == 0);
}

TEST_CASE("Multi-Hierarchy: Partial paths - Add to subset of hierarchies",
          "[ft][pyramid][multi_hierarchy]") {
    MultiHierarchyFixture f;
    auto index = vsag::Factory::CreateIndex("pyramid", f.build_param("nsw"));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(f.make_base()).has_value());

    // Add with only "site" paths, no "cat" paths
    auto* rv = new float[4]{0.5f, 0.5f, 0.0f, 0.0f};
    auto* ri = new int64_t[1]{200};
    auto* rs = new std::string[1]{"www/news"};
    auto add_ds = vsag::Dataset::Make();
    add_ds->NumElements(1)->Dim(4)->Float32Vectors(rv)->Ids(ri)->Paths("site", rs)->Owner(true);

    auto result = index.value()->Add(add_ds);
    REQUIRE(result.has_value());

    // id=200 should be findable in "site"
    auto site_result = f.search_ids(index.value(), "site", "www/news");
    REQUIRE(site_result.count(200) == 1);

    // id=200 should NOT be findable in "cat"
    auto cat_tech = f.search_ids(index.value(), "cat", "tech");
    REQUIRE(cat_tech.count(200) == 0);
}

TEST_CASE("Multi-Hierarchy: Error - unknown hierarchy in search",
          "[ft][pyramid][multi_hierarchy]") {
    MultiHierarchyFixture f;
    auto index = vsag::Factory::CreateIndex("pyramid", f.build_param("nsw"));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(f.make_base()).has_value());

    auto* qv = new float[4]{1.0f, 0.0f, 0.0f, 0.0f};
    auto* qp = new std::string[1]{"anything"};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(4)->Float32Vectors(qv)->Paths("unknown", qp)->Owner(true);

    std::string sp = R"({"pyramid": {"ef_search": 100, "hierarchies": ["unknown"]}})";
    auto result = index.value()->KnnSearch(query, 4, sp);
    REQUIRE_FALSE(result.has_value());
}

TEST_CASE("Multi-Hierarchy: Duplicate label rejected on Add", "[ft][pyramid][multi_hierarchy]") {
    MultiHierarchyFixture f;
    auto index = vsag::Factory::CreateIndex("pyramid", f.build_param("nsw"));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(f.make_base()).has_value());

    // Try to add id=100 again (already exists)
    auto* rv = new float[4]{0.9f, 0.1f, 0.0f, 0.0f};
    auto* ri = new int64_t[1]{100};
    auto* rs = new std::string[1]{"www/news"};
    auto* rc = new std::string[1]{"tech/ai"};
    auto add_ds = vsag::Dataset::Make();
    add_ds->NumElements(1)
        ->Dim(4)
        ->Float32Vectors(rv)
        ->Ids(ri)
        ->Paths("site", rs)
        ->Paths("cat", rc)
        ->Owner(true);

    auto result = index.value()->Add(add_ds);
    REQUIRE(result.has_value());
    // id=100 should be in the failed list
    auto failed = result.value();
    bool found_in_failed = false;
    for (auto id : failed) {
        if (id == 100) {
            found_in_failed = true;
        }
    }
    REQUIRE(found_in_failed);
}

TEST_CASE("Multi-Hierarchy: Legacy single-hierarchy serialize compat",
          "[ft][pyramid][multi_hierarchy][serialization]") {
    // Build with single-hierarchy (no "hierarchies" param) -> serialize -> deserialize -> search
    std::string single_param = R"({
        "dtype": "float32", "metric_type": "l2", "dim": 4,
        "index_param": {
            "max_degree": 32, "alpha": 1.2, "graph_type": "nsw",
            "base_quantization_type": "fp32", "use_reorder": false,
            "index_min_size": 0, "support_duplicate": false,
            "no_build_levels": [0]
        }
    })";

    auto index1 = vsag::Factory::CreateIndex("pyramid", single_param);
    REQUIRE(index1.has_value());

    auto* rv = new float[8]{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    auto* ri = new int64_t[2]{10, 20};
    auto* rp = new std::string[2]{"a/b", "a/c"};
    auto base = vsag::Dataset::Make();
    base->NumElements(2)->Dim(4)->Float32Vectors(rv)->Ids(ri)->Paths(rp)->Owner(true);
    REQUIRE(index1.value()->Build(base).has_value());

    auto bs = index1.value()->Serialize();
    REQUIRE(bs.has_value());

    auto index2 = vsag::Factory::CreateIndex("pyramid", single_param);
    REQUIRE(index2.has_value());
    index2.value()->Deserialize(bs.value());

    auto* qv = new float[4]{1.0f, 0.0f, 0.0f, 0.0f};
    auto* qp = new std::string[1]{"a/b"};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(4)->Float32Vectors(qv)->Paths(qp)->Owner(true);
    std::string sp = R"({"pyramid": {"ef_search": 100}})";
    auto result = index2.value()->KnnSearch(query, 2, sp);
    REQUIRE(result.has_value());

    auto* result_ids = result.value()->GetIds();
    auto cnt = result.value()->GetDim();
    std::set<int64_t> found(result_ids, result_ids + cnt);
    REQUIRE(found.count(10) == 1);
}

TEST_CASE("Multi-Hierarchy: Serialize roundtrip preserves isolation",
          "[ft][pyramid][multi_hierarchy][serialization]") {
    MultiHierarchyFixture f;
    auto param_str = f.build_param("nsw");

    auto index1 = vsag::Factory::CreateIndex("pyramid", param_str);
    REQUIRE(index1.has_value());
    REQUIRE(index1.value()->Build(f.make_base()).has_value());

    auto bs = index1.value()->Serialize();
    REQUIRE(bs.has_value());

    auto index2 = vsag::Factory::CreateIndex("pyramid", param_str);
    REQUIRE(index2.has_value());
    index2.value()->Deserialize(bs.value());

    // Verify isolation: site/www/sports should NOT contain id=100
    auto site_sports = f.search_ids(index2.value(), "site", "www/sports");
    REQUIRE(site_sports.count(100) == 0);
    REQUIRE(site_sports.count(101) == 1);
    REQUIRE(site_sports.count(103) == 1);

    // Verify isolation: cat/science should NOT contain id=100 or id=101
    auto cat_science = f.search_ids(index2.value(), "cat", "science");
    REQUIRE(cat_science.count(100) == 0);
    REQUIRE(cat_science.count(101) == 0);
    REQUIRE(cat_science.count(102) == 1);
    REQUIRE(cat_science.count(103) == 1);
}

TEST_CASE("Multi-Hierarchy: Per-hierarchy build params take effect",
          "[ft][pyramid][multi_hierarchy]") {
    // Create with different max_degree per hierarchy
    std::string param = R"({
        "dtype": "float32", "metric_type": "l2", "dim": 4,
        "index_param": {
            "max_degree": 16, "alpha": 1.0,
            "graph_type": "nsw",
            "base_quantization_type": "fp32",
            "use_reorder": false,
            "index_min_size": 0, "support_duplicate": false,
            "hierarchies": [
                {"name": "site", "no_build_levels": [0], "max_degree": 64, "alpha": 1.5},
                {"name": "cat", "no_build_levels": [0]}
            ]
        }
    })";

    MultiHierarchyFixture f;
    auto index = vsag::Factory::CreateIndex("pyramid", param);
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(f.make_base()).has_value());

    // Both hierarchies should work correctly regardless of different params
    auto site_news = f.search_ids(index.value(), "site", "www/news");
    REQUIRE(site_news.count(100) == 1);
    REQUIRE(site_news.count(102) == 1);

    auto cat_tech = f.search_ids(index.value(), "cat", "tech");
    REQUIRE(cat_tech.count(100) == 1);
    REQUIRE(cat_tech.count(101) == 1);
}

TEST_CASE("Multi-Hierarchy: AnalyzeIndexBySearch honors named query paths",
          "[ft][pyramid][multi_hierarchy][analyzer]") {
    MultiHierarchyFixture f;
    auto index = vsag::Factory::CreateIndex("pyramid", f.build_param("odescent"));
    REQUIRE(index.has_value());

    auto* base_vectors = new float[16]{0.0F,
                                       0.0F,
                                       0.0F,
                                       0.0F,
                                       10.0F,
                                       0.0F,
                                       0.0F,
                                       0.0F,
                                       20.0F,
                                       0.0F,
                                       0.0F,
                                       0.0F,
                                       100.0F,
                                       0.0F,
                                       0.0F,
                                       0.0F};
    auto* base_ids = new int64_t[4]{100, 101, 102, 103};
    auto* site_paths = new std::string[4]{"z", "x", "y", "z"};
    auto base = vsag::Dataset::Make();
    base->NumElements(4)
        ->Dim(4)
        ->Float32Vectors(base_vectors)
        ->Ids(base_ids)
        ->Paths("site", site_paths)
        ->Paths("cat",
                std::vector<std::vector<std::string>>{
                    {"z/zero"}, {"y/leaf"}, {"x/leaf", "y/leaf"}, {"z/hundred"}})
        ->Owner(true);
    REQUIRE(index.value()->Build(base).has_value());

    auto* query_vectors = new float[8]{0.0F, 0.0F, 0.0F, 0.0F, 100.0F, 0.0F, 0.0F, 0.0F};
    auto query = vsag::Dataset::Make();
    query->NumElements(2)
        ->Dim(4)
        ->Float32Vectors(query_vectors)
        ->Paths("cat", std::vector<std::vector<std::string>>{{"x", "y"}, {"x", "y"}})
        ->Owner(true);

    vsag::SearchRequest request;
    request.query_ = query;
    request.topk_ = 2;
    request.params_str_ = R"({"pyramid":{"ef_search":20,"hierarchies":["cat"]}})";

    // Both query rows select x union y. The selected cat scope is ids 101/102 even though id 102
    // belongs to both paths, so it must be counted only once in search and ground truth. The two
    // queries have exact squared L2 distances {100, 400} and {6400, 8100} in that scope.
    const auto stats = nlohmann::json::parse(index.value()->AnalyzeIndexBySearch(request));
    REQUIRE(std::abs(stats["recall_query"].get<float>() - 1.0F) <= 1e-6F);
    REQUIRE(std::abs(stats["avg_distance_query"].get<float>() - 3750.0F) <= 1e-6F);
}

TEST_CASE("Multi-Hierarchy: AnalyzeIndexBySearch topk one has finite quantization metrics",
          "[ft][pyramid][multi_hierarchy][analyzer]") {
    MultiHierarchyFixture f;
    auto index = vsag::Factory::CreateIndex("pyramid", f.build_param("nsw", true));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(f.make_base()).has_value());

    vsag::SearchRequest request;
    request.query_ = f.make_query("cat", "tech/ai");
    request.topk_ = 1;
    request.params_str_ = R"({"pyramid":{"ef_search":20,"hierarchies":["cat"]}})";

    const auto stats = nlohmann::json::parse(index.value()->AnalyzeIndexBySearch(request));
    REQUIRE(std::isfinite(stats["quantization_error_query"].get<float>()));
    REQUIRE(std::isfinite(stats["quantization_inversion_ratio_query"].get<float>()));
    REQUIRE(stats["quantization_inversion_ratio_query"].get<float>() == 0.0F);
}

TEST_CASE("Multi-Hierarchy: Analyzer output format - multi hierarchy",
          "[ft][pyramid][multi_hierarchy][analyzer]") {
    MultiHierarchyFixture f;
    auto index = vsag::Factory::CreateIndex("pyramid", f.build_param("nsw"));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(f.make_base()).has_value());

    auto stats_str = index.value()->GetStats();
    REQUIRE(!stats_str.empty());
    auto stats = nlohmann::json::parse(stats_str);

    REQUIRE(stats.contains("total_count"));
    REQUIRE(stats["total_count"].get<int64_t>() == 4);
    REQUIRE(stats.contains("hierarchy_count"));
    REQUIRE(stats["hierarchy_count"].get<int64_t>() == 2);
    REQUIRE(stats.contains("hierarchies"));
    REQUIRE(stats["hierarchies"].contains("site"));
    REQUIRE(stats["hierarchies"].contains("cat"));
    REQUIRE(stats["hierarchies"]["site"].contains("index_node_structure"));
    REQUIRE(stats["hierarchies"]["site"].contains("leaf_node_size_distribution"));
    REQUIRE(stats["hierarchies"]["site"].contains("subindex_quality"));
    REQUIRE(stats["hierarchies"]["cat"].contains("index_node_structure"));
}

TEST_CASE("Multi-Hierarchy: Analyzer output format - single hierarchy compat",
          "[ft][pyramid][multi_hierarchy][analyzer]") {
    std::string param = R"({
        "dtype": "float32", "metric_type": "l2", "dim": 4,
        "index_param": {
            "max_degree": 32, "alpha": 1.2, "graph_type": "nsw",
            "base_quantization_type": "fp32", "use_reorder": false,
            "index_min_size": 0, "support_duplicate": false,
            "no_build_levels": [0, 1, 2]
        }
    })";
    auto index = vsag::Factory::CreateIndex("pyramid", param);
    REQUIRE(index.has_value());

    auto* rv = new float[8]{1, 0, 0, 0, 0, 1, 0, 0};
    auto* ri = new int64_t[2]{1, 2};
    auto* rp = new std::string[2]{"a/b/c", "a/b/d"};
    auto ds = vsag::Dataset::Make();
    ds->NumElements(2)->Dim(4)->Float32Vectors(rv)->Ids(ri)->Paths(rp)->Owner(true);
    REQUIRE(index.value()->Build(ds).has_value());

    auto stats_str = index.value()->GetStats();
    auto stats = nlohmann::json::parse(stats_str);

    REQUIRE(stats.contains("total_count"));
    REQUIRE(stats.contains("index_node_structure"));
    REQUIRE(stats.contains("leaf_node_size_distribution"));
    REQUIRE(stats.contains("subindex_quality"));
    REQUIRE_FALSE(stats.contains("hierarchies"));
    REQUIRE_FALSE(stats.contains("hierarchy_count"));
}

TEST_CASE("Multi-Hierarchy: Search without hierarchies param in multi-mode errors",
          "[ft][pyramid][multi_hierarchy]") {
    MultiHierarchyFixture f;
    auto index = vsag::Factory::CreateIndex("pyramid", f.build_param("nsw"));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(f.make_base()).has_value());

    auto* qv = new float[4]{1.0f, 0.0f, 0.0f, 0.0f};
    auto* qp = new std::string[1]{"www/news"};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(4)->Float32Vectors(qv)->Paths(qp)->Owner(true);

    std::string sp = R"({"pyramid": {"ef_search": 100}})";
    auto result = index.value()->KnnSearch(query, 4, sp);
    REQUIRE_FALSE(result.has_value());
}

TEST_CASE("Multi-Hierarchy: Multi-hierarchy union/intersection rejected",
          "[ft][pyramid][multi_hierarchy]") {
    MultiHierarchyFixture f;
    auto index = vsag::Factory::CreateIndex("pyramid", f.build_param("nsw"));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(f.make_base()).has_value());

    auto* qv = new float[4]{1.0f, 0.0f, 0.0f, 0.0f};
    auto* qp_s = new std::string[1]{"www/news"};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(4)->Float32Vectors(qv)->Paths("site", qp_s)->Owner(true);

    std::string sp =
        R"({"pyramid": {"ef_search": 100, "hierarchies": ["site", "cat"], "hierarchy_op": "intersection"}})";
    auto result = index.value()->KnnSearch(query, 4, sp);
    REQUIRE_FALSE(result.has_value());
}

TEST_CASE("Multi-Hierarchy: RangeSearch works", "[ft][pyramid][multi_hierarchy]") {
    MultiHierarchyFixture f;
    auto index = vsag::Factory::CreateIndex("pyramid", f.build_param("nsw"));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(f.make_base()).has_value());

    auto* qv = new float[4]{1.0f, 0.0f, 0.0f, 0.0f};
    auto* qp = new std::string[1]{"www/news"};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(4)->Float32Vectors(qv)->Paths("site", qp)->Owner(true);

    std::string sp = R"({"pyramid": {"ef_search": 100, "hierarchies": ["site"]}})";
    auto result = index.value()->RangeSearch(query, 2.0f, sp);
    REQUIRE(result.has_value());
    auto* rids = result.value()->GetIds();
    auto cnt = result.value()->GetDim();
    std::set<int64_t> found(rids, rids + cnt);
    REQUIRE(found.count(100) == 1);
    REQUIRE(found.count(101) == 0);
    REQUIRE(found.count(103) == 0);
}

TEST_CASE("Multi-Hierarchy: Reorder with multi-hierarchy", "[ft][pyramid][multi_hierarchy]") {
    MultiHierarchyFixture f;
    auto index = vsag::Factory::CreateIndex("pyramid", f.build_param("nsw", true));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(f.make_base()).has_value());

    auto site_news = f.search_ids(index.value(), "site", "www/news");
    REQUIRE(site_news.count(100) == 1);
    REQUIRE(site_news.count(102) == 1);
    REQUIRE(site_news.count(101) == 0);

    auto cat_tech = f.search_ids(index.value(), "cat", "tech");
    REQUIRE(cat_tech.count(100) == 1);
    REQUIRE(cat_tech.count(101) == 1);
}

TEST_CASE("Multi-Hierarchy: Reorder statistics include precise IO",
          "[ft][pyramid][multi_hierarchy]") {
    MultiHierarchyFixture f;
    fixtures::TempDir temp_dir("pyramid_reorder_stats");
    auto index = vsag::Factory::CreateIndex(
        "pyramid", f.build_reorder_stats_param(temp_dir.GenerateRandomFile(false)));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(f.make_base()).has_value());

    auto* qv = new float[4]{1.0f, 0.0f, 0.0f, 0.0f};
    auto* qp = new std::string[1]{"www/news"};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(4)->Float32Vectors(qv)->Paths("site", qp)->Owner(true);

    std::string sp = R"({"pyramid": {"ef_search": 100, "hierarchies": ["site"]}})";
    auto knn_result = index.value()->KnnSearch(query, 2, sp);
    REQUIRE(knn_result.has_value());
    auto knn_stats = knn_result.value()->GetStatistics({"io_cnt", "reorder_distance_count"});
    REQUIRE(knn_stats.size() == 2);
    REQUIRE(std::stoul(knn_stats[0]) > 0);
    REQUIRE(std::stoul(knn_stats[1]) > 0);

    auto range_result = index.value()->RangeSearch(query, 2.0f, sp);
    REQUIRE(range_result.has_value());
    auto range_stats = range_result.value()->GetStatistics({"io_cnt", "reorder_distance_count"});
    REQUIRE(range_stats.size() == 2);
    REQUIRE(std::stoul(range_stats[0]) > 0);
    REQUIRE(std::stoul(range_stats[1]) > 0);
}

TEST_CASE("Multi-Hierarchy: Single hierarchy in hierarchies config",
          "[ft][pyramid][multi_hierarchy]") {
    std::string param = R"({
        "dtype": "float32", "metric_type": "l2", "dim": 4,
        "index_param": {
            "max_degree": 32, "alpha": 1.2, "graph_type": "nsw",
            "base_quantization_type": "fp32", "use_reorder": false,
            "index_min_size": 0, "support_duplicate": false,
            "hierarchies": [{"name": "only", "no_build_levels": [0]}]
        }
    })";

    auto index = vsag::Factory::CreateIndex("pyramid", param);
    REQUIRE(index.has_value());

    auto* rv = new float[8]{1, 0, 0, 0, 0, 1, 0, 0};
    auto* ri = new int64_t[2]{10, 20};
    auto* rp = new std::string[2]{"a/b", "a/c"};
    auto ds = vsag::Dataset::Make();
    ds->NumElements(2)->Dim(4)->Float32Vectors(rv)->Ids(ri)->Paths("only", rp)->Owner(true);
    REQUIRE(index.value()->Build(ds).has_value());

    auto* qv = new float[4]{1, 0, 0, 0};
    auto* qp = new std::string[1]{"a/b"};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(4)->Float32Vectors(qv)->Paths("only", qp)->Owner(true);
    std::string sp = R"({"pyramid": {"ef_search": 100, "hierarchies": ["only"]}})";
    auto result = index.value()->KnnSearch(query, 2, sp);
    REQUIRE(result.has_value());
    auto* rids = result.value()->GetIds();
    std::set<int64_t> found(rids, rids + result.value()->GetDim());
    REQUIRE(found.count(10) == 1);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Mark Remove",
                             "[ft][remove][pyramid]") {
    auto metric_type = GENERATE("l2");
    PyramidParam pyramid_param;
    pyramid_param.no_build_levels = {0, 1, 2};
    const std::string name = "pyramid";
    auto search_param = GeneratePyramidSearchParametersString(200);
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}", metric_type, dim));
        auto param = GeneratePyramidBuildParametersString(metric_type, dim, pyramid_param);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type, /*with_path=*/true);
        TestBuildIndex(index, dataset, true);

        auto base_num = dataset->base_->GetNumElements();
        const auto* ids = dataset->base_->GetIds();
        REQUIRE(index->GetNumElements() == base_num);
        REQUIRE(index->GetNumberRemoved() == 0);

        // FORCE_REMOVE is not supported by Pyramid
        auto force_result = index->Remove(ids[0], vsag::RemoveMode::FORCE_REMOVE);
        REQUIRE_FALSE(force_result.has_value());

        // mark remove half of the base data
        int64_t remove_count = base_num / 2;
        std::vector<int64_t> remove_ids(ids, ids + remove_count);
        auto remove_result = index->Remove(remove_ids, vsag::RemoveMode::MARK_REMOVE);
        REQUIRE(remove_result.has_value());
        REQUIRE(remove_result.value() == remove_count);
        REQUIRE(index->GetNumElements() == base_num - remove_count);
        REQUIRE(index->GetNumberRemoved() == remove_count);

        // removing the same ids again should remove nothing
        auto duplicate_remove = index->Remove(remove_ids, vsag::RemoveMode::MARK_REMOVE);
        REQUIRE(duplicate_remove.has_value());
        REQUIRE(duplicate_remove.value() == 0);

        // removed ids must not appear in search results
        for (int64_t i = 0; i < remove_count; ++i) {
            auto query = fixtures::get_one_query(dataset->base_, static_cast<int>(i));
            auto search_result = index->KnnSearch(query, 10, search_param);
            REQUIRE(search_result.has_value());
            for (int64_t j = 0; j < search_result.value()->GetDim(); ++j) {
                REQUIRE(search_result.value()->GetIds()[j] != ids[i]);
            }
        }
    }
}

namespace {

auto
MakePyramidCacheDataset(int64_t count,
                        int64_t dim,
                        const std::vector<float>& vectors,
                        const std::vector<int64_t>& ids,
                        const std::vector<std::string>& paths,
                        const std::vector<std::string>& source_ids) -> vsag::DatasetPtr {
    REQUIRE(vectors.size() == static_cast<size_t>(count * dim));
    REQUIRE(ids.size() == static_cast<size_t>(count));
    REQUIRE(paths.size() == static_cast<size_t>(count));
    REQUIRE(source_ids.size() == static_cast<size_t>(count));

    auto dataset = vsag::Dataset::Make();
    auto raw_vectors = std::make_unique<float[]>(count * dim);
    auto raw_ids = std::make_unique<int64_t[]>(count);
    auto raw_paths = std::make_unique<std::string[]>(count);
    auto raw_source_ids = std::make_unique<std::string[]>(count);

    std::copy(vectors.begin(), vectors.end(), raw_vectors.get());
    std::copy(ids.begin(), ids.end(), raw_ids.get());
    std::copy(paths.begin(), paths.end(), raw_paths.get());
    std::copy(source_ids.begin(), source_ids.end(), raw_source_ids.get());

    dataset->NumElements(count)
        ->Dim(dim)
        ->Float32Vectors(raw_vectors.get())
        ->Ids(raw_ids.get())
        ->Paths(raw_paths.get())
        ->SourceID(raw_source_ids.get())
        ->Owner(true);
    raw_vectors.release();
    raw_ids.release();
    raw_paths.release();
    raw_source_ids.release();
    return dataset;
}

auto
MakePyramidCacheQuery(int64_t dim, const std::vector<float>& query, const std::string& path)
    -> vsag::DatasetPtr {
    REQUIRE(query.size() == static_cast<size_t>(dim));
    auto dataset = vsag::Dataset::Make();
    auto raw_query_vec = std::make_unique<float[]>(dim);
    std::copy(query.begin(), query.end(), raw_query_vec.get());
    auto raw_paths = std::make_unique<std::string[]>(1);
    raw_paths[0] = path;
    dataset->NumElements(1)
        ->Dim(dim)
        ->Float32Vectors(raw_query_vec.get())
        ->Paths(path.empty() ? nullptr : raw_paths.get())
        ->Owner(true);
    raw_query_vec.release();
    if (!path.empty()) {
        raw_paths.release();
    }
    return dataset;
}

}  // namespace

TEST_CASE("Pyramid ExportCache + ImportCache + Build acceleration smoke test",
          "[ft][pyramid][cache][pr]") {
    // End-to-end smoke test mirroring the HGraph cache test, adapted for the
    // Pyramid tree-of-graphs structure: every inserted vector shares a deep
    // primary path, and half also use a second path. The primary leaf becomes
    // a GRAPH that fulfill_cache() walks while structured paths exercise the
    // warm-start tree population path.
    //   (1) Build a baseline Pyramid with N points carrying source_id.
    //   (2) ExportCache to an in-memory stream.
    //   (3) Fresh Pyramid, ImportCache, then Build the same dataset — Build
    //       should automatically take the build_with_cache() warm-start path.
    //   (4) Verify the warmed index is searchable and the first inserted
    //       vector (query == vectors[0]) is returned with ~0 distance.
    constexpr int64_t TEST_DIM = 32;
    constexpr int64_t TEST_COUNT = 200;
    constexpr int64_t TOPK = 10;
    const auto graph_storage_type = GENERATE(std::string("flat"), std::string("compressed"));
    CAPTURE(graph_storage_type);

    // params must include persist_source_id: true so ExportCache produces a
    // usable cache after a Build that recorded source_ids. ef_construction equals
    // max_degree to match the HGraph cache-hit refinement eligibility.
    const auto param = fmt::format(R"(
    {{
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 32,
        "index_param": {{
            "base_quantization_type": "fp32",
            "max_degree": 16,
            "ef_construction": 16,
            "build_thread_count": 8,
            "root_graph_type": "multi_layer",
            "graph_storage_type": "{}",
            "no_build_levels": [1, 2],
            "index_min_size": 28,
            "persist_source_id": true,
            "store_paths": true
        }}
    }}
    )",
                                   graph_storage_type);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> vectors(TEST_DIM * TEST_COUNT);
    for (auto& v : vectors) {
        v = dist(rng);
    }
    std::vector<int64_t> ids(TEST_COUNT);
    for (int64_t i = 0; i < TEST_COUNT; ++i) {
        ids[i] = i + 1;
    }
    // All vectors share a deep path. Levels 1 and 2 remain NO_INDEX, while the root and the
    // depth-3 leaf exceed index_min_size and become GRAPH nodes that fulfill_cache() walks.
    std::vector<std::string> paths(TEST_COUNT, "a/b/c");
    std::vector<std::string> source_ids(TEST_COUNT);
    for (int64_t i = 0; i < TEST_COUNT; ++i) {
        source_ids[i] = fmt::format("pyr_sid_{}", i);
    }

    auto make_dataset = [&]() {
        auto dataset =
            MakePyramidCacheDataset(TEST_COUNT, TEST_DIM, vectors, ids, paths, source_ids);
        std::vector<std::vector<std::string>> structured_paths(TEST_COUNT);
        for (int64_t i = 0; i < TEST_COUNT; ++i) {
            structured_paths[i].push_back(paths[i]);
            if (i % 2 == 0) {
                structured_paths[i].push_back("alternate/cache/leaf");
            }
        }
        dataset->Paths("", std::move(structured_paths));
        return dataset;
    };

    // ---- (1) baseline build ----
    auto baseline = vsag::Factory::CreateIndex("pyramid", param).value();
    auto baseline_build = baseline->Build(make_dataset());
    REQUIRE(baseline_build.has_value());
    REQUIRE(baseline->GetNumElements() == TEST_COUNT);

    // ---- (2) preserve SourceID in the all-in-one module section ----
    auto binary = baseline->Serialize();
    REQUIRE(binary.has_value());
    auto all_in_one = vsag::Factory::CreateIndex("pyramid", param).value();
    REQUIRE(all_in_one->Deserialize(binary.value()).has_value());

    // ---- (3) export cache from the footer-deserialized index ----
    std::stringstream cache_buf;
    auto export_result = all_in_one->ExportCache(cache_buf);
    REQUIRE(export_result.has_value());
    REQUIRE(cache_buf.tellp() > 0);

    // ---- (4) fresh index, import cache, build again ----
    cache_buf.seekg(0);
    auto warmed = vsag::Factory::CreateIndex("pyramid", param).value();
    auto import_result = warmed->ImportCache(cache_buf);
    REQUIRE(import_result.has_value());
    auto warmed_build = warmed->Build(make_dataset());
    REQUIRE(warmed_build.has_value());
    REQUIRE(warmed->GetNumElements() == TEST_COUNT);
    auto restored_data =
        warmed->GetDataByIdsWithFlag(ids.data() + 1, 1, DATA_FLAG_ID | DATA_FLAG_PATH);
    REQUIRE(restored_data.has_value());
    REQUIRE(restored_data.value()->GetPaths() != nullptr);
    REQUIRE(restored_data.value()->GetPaths()[0] == "a/b/c");
    auto warm_stats = vsag::JsonType::Parse(warmed->GetStats());
    REQUIRE(warm_stats["build_cache_hit_nodes"].GetInt() > 0);
    REQUIRE(warm_stats["build_cache_hit_memberships"].GetInt() > 0);
    REQUIRE(warm_stats["build_cache_restored_edges"].GetInt() > 0);
    REQUIRE(warm_stats["root_graphs"]["default"]["route_graph_count"].GetInt() > 0);
    std::vector<float> query_vec(TEST_DIM);
    std::copy(vectors.begin(), vectors.begin() + TEST_DIM, query_vec.begin());
    const auto* search_param = R"({"pyramid": {"ef_search": 50}})";
    const auto require_self_at_path = [&](const std::string& path) {
        auto query = MakePyramidCacheQuery(TEST_DIM, query_vec, path);
        auto search_result = warmed->KnnSearch(query, TOPK, search_param);
        REQUIRE(search_result.has_value());
        auto knn = search_result.value();
        REQUIRE(knn->GetNumElements() == 1);
        REQUIRE(knn->GetDim() > 0);
        REQUIRE(knn->GetDim() <= TOPK);
        bool found_self = false;
        for (int64_t i = 0; i < knn->GetDim(); ++i) {
            if (knn->GetIds()[i] == ids[0]) {
                found_self = true;
                REQUIRE(knn->GetDistances()[i] < 1e-4F);
                break;
            }
        }
        REQUIRE(found_self);
    };
    // Verify both memberships survive cache import and warm-start tree population.
    require_self_at_path("a/b/c");
    require_self_at_path("alternate/cache/leaf");
}

TEST_CASE("Pyramid Build Cache clamps rows to the current maximum degree",
          "[ft][pyramid][cache][pr]") {
    constexpr int64_t test_dim = 32;
    constexpr int64_t test_count = 200;
    const auto source_param = fmt::format(R"({{
        "dtype": "float32",
        "metric_type": "l2",
        "dim": {},
        "index_param": {{
            "base_quantization_type": "fp32",
            "max_degree": 32,
            "ef_construction": 64,
            "no_build_levels": [0, 1, 2],
            "index_min_size": 28,
            "persist_source_id": true
        }}
    }})",
                                          test_dim);
    const auto target_param = fmt::format(R"({{
        "dtype": "float32",
        "metric_type": "l2",
        "dim": {},
        "index_param": {{
            "base_quantization_type": "fp32",
            "max_degree": 8,
            "ef_construction": 32,
            "no_build_levels": [0, 1, 2],
            "index_min_size": 28,
            "persist_source_id": true
        }}
    }})",
                                          test_dim);

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> vectors(test_dim * test_count);
    for (auto& value : vectors) {
        value = dist(rng);
    }
    std::vector<int64_t> ids(test_count);
    std::vector<std::string> paths(test_count, "a/b/c");
    std::vector<std::string> source_ids(test_count);
    for (int64_t i = 0; i < test_count; ++i) {
        ids[i] = i + 1;
        source_ids[i] = fmt::format("pyr_degree_sid_{}", i);
    }
    auto make_dataset = [&]() {
        return MakePyramidCacheDataset(test_count, test_dim, vectors, ids, paths, source_ids);
    };

    auto source = vsag::Factory::CreateIndex("pyramid", source_param).value();
    REQUIRE(source->Build(make_dataset()).has_value());
    std::stringstream cache;
    REQUIRE(source->ExportCache(cache).has_value());

    cache.seekg(0);
    auto warmed = vsag::Factory::CreateIndex("pyramid", target_param).value();
    REQUIRE(warmed->ImportCache(cache).has_value());
    REQUIRE(warmed->Build(make_dataset()).has_value());
    auto stats = vsag::JsonType::Parse(warmed->GetStats());
    REQUIRE(stats["build_cache_hit_nodes"].GetInt() > 0);

    std::vector<float> query_vector(test_dim);
    std::copy_n(vectors.data(), test_dim, query_vector.data());
    auto query = MakePyramidCacheQuery(test_dim, query_vector, "a/b/c");
    auto result = warmed->KnnSearch(query, 10, R"({"pyramid":{"ef_search":50}})");
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
}

TEST_CASE("Pyramid ExportCache + ImportCache + Build miss-only path", "[ft][pyramid][cache][pr]") {
    // Force every node to take the *missed* branch of build_with_cache by
    // using a disjoint source_id set on the warmed build. The expected
    // hit-rate is 0 and missed_nodes equals the index size.
    constexpr int64_t TEST_DIM = 32;
    constexpr int64_t TEST_COUNT = 200;
    constexpr int64_t TOPK = 10;

    const auto* param = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 32,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 16,
            "ef_construction": 50,
            "no_build_levels": [0, 1, 2],
            "index_min_size": 28,
            "persist_source_id": true
        }
    }
    )";

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> vectors(TEST_DIM * TEST_COUNT);
    for (auto& v : vectors) {
        v = dist(rng);
    }
    std::vector<int64_t> ids(TEST_COUNT);
    for (int64_t i = 0; i < TEST_COUNT; ++i) {
        ids[i] = i + 1;
    }
    std::vector<std::string> paths(TEST_COUNT, "a/b/c");
    std::vector<std::string> source_ids_a(TEST_COUNT);
    std::vector<std::string> source_ids_b(TEST_COUNT);
    for (int64_t i = 0; i < TEST_COUNT; ++i) {
        source_ids_a[i] = fmt::format("pyr_A_{}", i);
        source_ids_b[i] = fmt::format("pyr_B_{}", i);
    }

    auto make_dataset = [&](const std::vector<std::string>& sids) {
        return MakePyramidCacheDataset(TEST_COUNT, TEST_DIM, vectors, ids, paths, sids);
    };

    // ---- (1) baseline build with source_id "pyr_A_*" ----
    auto baseline = vsag::Factory::CreateIndex("pyramid", param).value();
    auto baseline_build = baseline->Build(make_dataset(source_ids_a));
    REQUIRE(baseline_build.has_value());
    REQUIRE(baseline->GetNumElements() == TEST_COUNT);

    // ---- (2) export cache (only contains "pyr_A_*" source_ids) ----
    std::stringstream cache_buf;
    auto export_result = baseline->ExportCache(cache_buf);
    REQUIRE(export_result.has_value());
    REQUIRE(cache_buf.tellp() > 0);

    // ---- (3) fresh index, import cache, build with disjoint source_ids ----
    cache_buf.seekg(0);
    auto warmed = vsag::Factory::CreateIndex("pyramid", param).value();
    auto import_result = warmed->ImportCache(cache_buf);
    REQUIRE(import_result.has_value());
    auto warmed_build = warmed->Build(make_dataset(source_ids_b));
    REQUIRE(warmed_build.has_value());
    REQUIRE(warmed->GetNumElements() == TEST_COUNT);

    // ---- (4) baseline cache hit-rate must be 100% (full overlap) ----
    auto baseline_stats_str = baseline->GetStats();
    auto baseline_parsed = vsag::JsonType::Parse(baseline_stats_str);
    // baseline was built cold -> skipped_reason
    REQUIRE(baseline_parsed.Contains("build_cache_hit_rate"));
    REQUIRE(baseline_parsed["build_cache_hit_rate"].Contains("skipped_reason"));

    // Disjoint source IDs fall back to the regular cold builder before cache restoration.
    auto warmed_stats_str = warmed->GetStats();
    INFO(warmed_stats_str);
    auto warmed_parsed = vsag::JsonType::Parse(warmed_stats_str);
    REQUIRE(warmed_parsed.Contains("build_cache_hit_rate"));
    REQUIRE(warmed_parsed["build_cache_hit_rate"].Contains("skipped_reason"));
}

TEST_CASE("Pyramid GetStats reports build cache hit-rate", "[ft][pyramid][cache][pr]") {
    // Verify that the build-time warm-start hit-rate computed during
    // build_with_cache() is surfaced through GetStats():
    //   (1) A normal Build() (no imported cache) reports a skipped_reason.
    //   (2) A Build() after ImportCache() reports a numeric hit-rate plus
    //       the hit / missed node counts, and miss + hit equals the warm
    //       build's vector count for this single-hierarchy setup.
    constexpr int64_t TEST_DIM = 32;
    constexpr int64_t TEST_COUNT = 200;

    const auto* param = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 32,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 16,
            "ef_construction": 50,
            "no_build_levels": [0, 1, 2],
            "index_min_size": 28,
            "persist_source_id": true
        }
    }
    )";

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> vectors(TEST_DIM * TEST_COUNT);
    for (auto& v : vectors) {
        v = dist(rng);
    }
    std::vector<int64_t> ids(TEST_COUNT);
    for (int64_t i = 0; i < TEST_COUNT; ++i) {
        ids[i] = i + 1;
    }
    std::vector<std::string> paths(TEST_COUNT, "a/b/c");
    std::vector<std::string> source_ids(TEST_COUNT);
    for (int64_t i = 0; i < TEST_COUNT; ++i) {
        source_ids[i] = fmt::format("pyr_stat_{}", i);
    }
    auto make_dataset = [&]() {
        return MakePyramidCacheDataset(TEST_COUNT, TEST_DIM, vectors, ids, paths, source_ids);
    };

    // ---- (1) no cache -> skipped_reason in stats ----
    auto no_cache = vsag::Factory::CreateIndex("pyramid", param).value();
    auto cold_build = no_cache->Build(make_dataset());
    REQUIRE(cold_build.has_value());
    auto cold_stats_str = no_cache->GetStats();
    auto cold_parsed = vsag::JsonType::Parse(cold_stats_str);
    REQUIRE(cold_parsed.Contains("build_cache_hit_rate"));
    REQUIRE(cold_parsed["build_cache_hit_rate"].Contains("skipped_reason"));

    // ---- (2) with cache -> numeric hit-rate + hit/missed counts ----
    std::stringstream cache_buf;
    REQUIRE(no_cache->ExportCache(cache_buf).has_value());
    cache_buf.seekg(0);
    auto warmed = vsag::Factory::CreateIndex("pyramid", param).value();
    REQUIRE(warmed->ImportCache(cache_buf).has_value());
    REQUIRE(warmed->Build(make_dataset()).has_value());

    auto warm_stats_str = warmed->GetStats();
    INFO(warm_stats_str);
    auto warm_parsed = vsag::JsonType::Parse(warm_stats_str);
    REQUIRE(warm_parsed.Contains("build_cache_hit_rate"));
    REQUIRE(warm_parsed.Contains("build_cache_hit_nodes"));
    REQUIRE(warm_parsed.Contains("build_cache_missed_nodes"));
    REQUIRE(warm_parsed.Contains("build_cache_hit_memberships"));
    REQUIRE(warm_parsed.Contains("build_cache_missed_memberships"));
    REQUIRE(warm_parsed.Contains("build_cache_restored_edges"));
    REQUIRE(warm_parsed["build_cache_hit_memberships"].GetInt() > 0);
    REQUIRE(warm_parsed["build_cache_restored_edges"].GetInt() > 0);
    const float hit_rate = warm_parsed["build_cache_hit_rate"].GetFloat();
    REQUIRE(hit_rate > 0.0F);
    REQUIRE(hit_rate <= 1.0F);
    const auto hit_nodes = warm_parsed["build_cache_hit_nodes"].GetInt();
    const auto missed_nodes = warm_parsed["build_cache_missed_nodes"].GetInt();
    REQUIRE(hit_nodes + missed_nodes == TEST_COUNT);
}

TEST_CASE("Pyramid duplicate cache roundtrips and membership changes",
          "[ft][pyramid][cache][duplicate_cache][pr]") {
    const auto root = GENERATE(std::string("single_layer"), std::string("multi_layer"));
    const auto change = GENERATE(std::string("reorder"),
                                 std::string("remove-representative"),
                                 std::string("remove-alias"),
                                 std::string("move-path"),
                                 std::string("all-identical"),
                                 std::string("no-edges"),
                                 std::string("repeated-label"));
    const auto use_cache = GENERATE(false, true);
    const auto serialized = GENERATE(false, true);
    CAPTURE(root, change, use_cache, serialized);
    constexpr int64_t dim = 8;
    constexpr int64_t count = 120;
    const auto param = fmt::format(R"({{
        "dtype":"float32", "metric_type":"l2", "dim":8,
        "index_param":{{"base_quantization_type":"fp32", "max_degree":16,
        "ef_construction":64, "build_thread_count":1, "root_graph_type":"{}",
        "index_min_size":8, "persist_source_id":true, "store_paths":true,
        "support_duplicate":true}} }})",
                                   root);
    std::vector<float> vectors(count * dim);
    std::vector<int64_t> ids(count);
    std::vector<std::string> paths(count, "a/b");
    std::vector<std::string> sources(count);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = 1000 + i;
        sources[i] = "duplicate-source-" + std::to_string(i);
        const auto group = change == "all-identical" ? 0 : i / 10;
        for (int64_t d = 0; d < dim; ++d) {
            vectors[i * dim + d] = static_cast<float>(group * (d + 1));
        }
    }
    auto dataset = [&]() {
        return MakePyramidCacheDataset(ids.size(), dim, vectors, ids, paths, sources);
    };
    auto baseline = vsag::Factory::CreateIndex("pyramid", param).value();
    REQUIRE(baseline->Build(dataset()).has_value());
    std::stringstream cache;
    REQUIRE(baseline->ExportCache(cache).has_value());
    if (change == "remove-representative" || change == "remove-alias") {
        const uint64_t offset = change == "remove-representative" ? 0 : 1;
        ids.erase(ids.begin() + offset);
        paths.erase(paths.begin() + offset);
        sources.erase(sources.begin() + offset);
        vectors.erase(vectors.begin() + offset * dim, vectors.begin() + (offset + 1) * dim);
    } else if (change == "move-path") {
        paths[0] = "other/path";
    } else if (change == "no-edges") {
        ids.resize(10);
        paths.resize(10);
        sources.resize(10);
        vectors.resize(10 * dim);
    } else if (change == "repeated-label") {
        ids[1] = ids[0];
    }
    // Reverse input IDs without changing SourceID/vector identity; the actual
    // representative is now not necessarily the minimum inner ID in its group.
    std::reverse(ids.begin(), ids.end());
    std::reverse(paths.begin(), paths.end());
    std::reverse(sources.begin(), sources.end());
    for (uint64_t i = 0; i < ids.size() / 2; ++i) {
        for (int64_t d = 0; d < dim; ++d) {
            std::swap(vectors[i * dim + d], vectors[(ids.size() - i - 1) * dim + d]);
        }
    }
    for (int round = 0; round < 2; ++round) {
        auto warm = vsag::Factory::CreateIndex("pyramid", param).value();
        if (use_cache) {
            cache.seekg(0);
            REQUIRE(warm->ImportCache(cache).has_value());
        }
        auto built = warm->Build(dataset());
        REQUIRE(built.has_value());
        if (change == "repeated-label") {
            REQUIRE_FALSE(built.value().empty());
            return;
        }
        REQUIRE(built.value().empty());
        REQUIRE(warm->GetNumElements() == static_cast<int64_t>(ids.size()));
        if (use_cache && change == "remove-representative" && round == 0) {
            const auto stats = vsag::JsonType::Parse(warm->GetStats());
            // Only the affected ten-member class loses reuse; its nine survivors
            // are inserted normally. The other eleven classes keep their cache rows.
            REQUIRE(stats["build_cache_hit_nodes"].GetInt() == 110);
            REQUIRE(stats["build_cache_missed_nodes"].GetInt() == 9);
            REQUIRE(stats["build_cache_restored_edges"].GetInt() > 0);
        }
        if (use_cache && change == "reorder") {
            auto stats = vsag::JsonType::Parse(warm->GetStats());
            REQUIRE(stats["build_cache_hit_nodes"].GetInt() == static_cast<int64_t>(ids.size()));
            REQUIRE(stats["build_cache_restored_edges"].GetInt() > 0);
        }
        auto binary = warm->Serialize();
        REQUIRE(binary.has_value());
        auto restored = vsag::Factory::CreateIndex("pyramid", param).value();
        REQUIRE(restored->Deserialize(binary.value()).has_value());
        if (!serialized) {
            restored = warm;
        }
        for (uint64_t i = 0; i < ids.size(); ++i) {
            std::vector<float> vector(vectors.begin() + i * dim, vectors.begin() + (i + 1) * dim);
            for (const auto& path : {paths[i], std::string{}}) {
                auto query = MakePyramidCacheQuery(dim, vector, path);
                auto result =
                    restored->KnnSearch(query, ids.size(), R"({"pyramid":{"ef_search":256}})");
                REQUIRE(result.has_value());
                std::vector<int64_t> expected;
                for (uint64_t j = 0; j < ids.size(); ++j) {
                    if (path.empty() || paths[j] == path) {
                        expected.push_back(ids[j]);
                    }
                }
                REQUIRE(result.value()->GetDim() == static_cast<int64_t>(expected.size()));
                std::vector<int64_t> actual;
                for (int64_t j = 0; j < result.value()->GetDim(); ++j) {
                    const auto label = result.value()->GetIds()[j];
                    actual.push_back(label);
                    auto found = std::find(ids.begin(), ids.end(), label);
                    REQUIRE(found != ids.end());
                    const uint64_t offset = std::distance(ids.begin(), found);
                    float distance = 0;
                    for (int64_t d = 0; d < dim; ++d) {
                        const auto delta = vector[d] - vectors[offset * dim + d];
                        distance += delta * delta;
                    }
                    REQUIRE(std::abs(result.value()->GetDistances()[j] - distance) < 0.01F);
                }
                std::sort(expected.begin(), expected.end());
                std::sort(actual.begin(), actual.end());
                REQUIRE(actual == expected);
            }
        }
        cache.str("");
        cache.clear();
        REQUIRE(restored->ExportCache(cache).has_value());
    }
}

static void
CheckLargePyramidDuplicateCache(const std::string& root,
                                const std::string& storage,
                                int cold_threads,
                                int warm_threads = 4,
                                int copies = 2,
                                bool production_construction = false) {
    CAPTURE(root, storage, cold_threads);
    constexpr int64_t groups = 1050;
    const int64_t count = groups * copies;
    constexpr int64_t dim = 2;
    const auto param = fmt::format(R"({{
        "dtype":"float32", "metric_type":"l2", "dim":2,
        "index_param":{{"base_quantization_type":"fp32", "max_degree":16,
        "ef_construction":32, "build_thread_count":4, "root_graph_type":"{}",
        "graph_storage_type":"{}", "index_min_size":8,
        "persist_source_id":true, "store_paths":true, "support_duplicate":true}} }})",
                                   root,
                                   storage);
    std::vector<float> vectors(count * dim);
    std::vector<int64_t> ids(count);
    std::vector<std::string> paths(count, "a/b");
    std::vector<std::string> sources(count);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = 10000 + i;
        sources[i] = "large-duplicate-" + std::to_string(i);
        vectors[i * dim] = static_cast<float>(i / copies);
        vectors[i * dim + 1] = static_cast<float>((i / copies) % 17);
    }
    auto dataset = MakePyramidCacheDataset(count, dim, vectors, ids, paths, sources);
    auto cold_param = vsag::JsonType::Parse(param);
    if (production_construction) {
        cold_param["index_param"]["max_degree"].SetInt(64);
        cold_param["index_param"]["ef_construction"].SetInt(500);
    }
    cold_param["index_param"]["build_thread_count"].SetInt(cold_threads);
    cold_param["index_param"]["support_duplicate"].SetBool(copies == 2);
    auto cold = vsag::Factory::CreateIndex("pyramid", cold_param.Dump()).value();
    REQUIRE(cold->Build(dataset).has_value());
    std::stringstream cache;
    REQUIRE(cold->ExportCache(cache).has_value());
    // Inspect the interchange representation, not private dense storage high-water IDs.
    // Every group has exactly one physical row and every edge targets another owner.
    {
        auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
        vsag::PyramidBuildCache exported(allocator.get());
        std::stringstream snapshot(cache.str());
        vsag::IOStreamReader reader(snapshot);
        exported.Deserialize(reader);
        for (const auto& path : {std::string{}, std::string("a"), std::string("a/b")}) {
            auto* graph = exported.GetGraphCache("", path);
            const auto* owners = exported.GetGroupOwners("", path);
            REQUIRE(graph != nullptr);
            REQUIRE(owners != nullptr);
            REQUIRE(owners->size() == static_cast<uint64_t>(count));
            uint64_t representatives = 0;
            for (uint64_t member = 0; member < owners->size(); ++member) {
                const auto owner = (*owners)[member];
                REQUIRE(owner < owners->size());
                REQUIRE((*owners)[owner] == owner);
                if (owner != member) {
                    continue;
                }
                ++representatives;
                const auto* row = graph->FindNeighborInnerIds(graph->source_ids_[member]);
                REQUIRE(row != nullptr);
                REQUIRE(row->size() <= static_cast<uint64_t>(production_construction ? 65 : 17));
                REQUIRE(row->front() == member);
                for (uint64_t edge = 1; edge < row->size(); ++edge) {
                    REQUIRE((*row)[edge] < owners->size());
                    REQUIRE((*owners)[(*row)[edge]] == (*row)[edge]);
                }
            }
            REQUIRE(representatives == static_cast<uint64_t>(groups));
        }
    }
    auto warm_param = vsag::JsonType::Parse(param);
    if (production_construction) {
        warm_param["index_param"]["max_degree"].SetInt(64);
        warm_param["index_param"]["ef_construction"].SetInt(500);
    }
    warm_param["index_param"]["build_thread_count"].SetInt(warm_threads);
    warm_param["index_param"]["support_duplicate"].SetBool(copies == 2);
    auto warm = vsag::Factory::CreateIndex("pyramid", warm_param.Dump()).value();
    REQUIRE(warm->ImportCache(cache).has_value());
    REQUIRE(warm->Build(dataset).has_value());
    const auto stats = vsag::JsonType::Parse(warm->GetStats());
    const auto& root_stats = stats["root_graphs"]["default"];
    REQUIRE(stats["build_cache_hit_nodes"].GetInt() == count);
    REQUIRE(stats["build_cache_restored_edges"].GetInt() > 0);
    if (cold_threads == 1) {
        // Dense routed roots report the high-water ID, not occupied row count.
        REQUIRE(root_stats["bottom_graph_node_count"].GetInt() ==
                (root == "multi_layer" ? count - 1 : groups));
    }
    if (root == "multi_layer") {
        REQUIRE(root_stats["route_graph_count"].GetInt() > 0);
    } else {
        REQUIRE(root_stats["route_graph_count"].GetInt() == 0);
    }
    auto binary = warm->Serialize();
    REQUIRE(binary.has_value());
    auto restored = vsag::Factory::CreateIndex("pyramid", warm_param.Dump()).value();
    REQUIRE(restored->Deserialize(binary.value()).has_value());
    for (const auto& index : {cold, warm, restored}) {
        CAPTURE(index == cold ? "cold" : (index == warm ? "warm" : "restored"));
        for (const auto offset : {int64_t{0}, int64_t{1024}, count - 2}) {
            for (const auto& path : {std::string("a/b"), std::string{}}) {
                CAPTURE(offset, path);
                std::vector<float> vector(vectors.begin() + offset * dim,
                                          vectors.begin() + (offset + 1) * dim);
                auto query = MakePyramidCacheQuery(dim, vector, path);
                auto result = index->KnnSearch(query, count, R"({"pyramid":{"ef_search":4096}})");
                REQUIRE(result.has_value());
                if (result.value()->GetDim() != count) {
                    std::vector<bool> found(count, false);
                    for (int64_t j = 0; j < result.value()->GetDim(); ++j) {
                        found[result.value()->GetIds()[j] - 10000] = true;
                    }
                    uint64_t partial_groups = 0;
                    std::string missing;
                    for (int64_t j = 0; j < count; ++j) {
                        if (!found[j]) {
                            missing += std::to_string(j) + ",";
                        }
                        if (j % 2 == 0 && found[j] != found[j + 1]) {
                            ++partial_groups;
                        }
                    }
                    WARN("missing IDs=" << missing << " partial_groups=" << partial_groups);
                    auto range =
                        index->RangeSearch(query, 1e12F, R"({"pyramid":{"ef_search":4096}})");
                    REQUIRE(range.has_value());
                    WARN("Range count=" << range.value()->GetDim());
                }
                REQUIRE(result.value()->GetDim() == count);
                std::vector<int64_t> actual;
                for (int64_t j = 0; j < count; ++j) {
                    const auto label = result.value()->GetIds()[j];
                    REQUIRE(label >= 10000);
                    REQUIRE(label < 10000 + count);
                    actual.push_back(label);
                    const auto id = label - 10000;
                    float distance = 0.0F;
                    for (int64_t d = 0; d < dim; ++d) {
                        const auto delta = vector[d] - vectors[id * dim + d];
                        distance += delta * delta;
                    }
                    REQUIRE(result.value()->GetDistances()[j] == distance);
                }
                std::sort(actual.begin(), actual.end());
                REQUIRE(actual == ids);
            }
        }
    }
    // Rehydrate transient admission from physical representatives after deserialize.
    // Also exercise the existing cold and cache-restored registries with incremental Add.
    if (copies == 2) {
        std::vector<float> extra_vector(vectors.begin(), vectors.begin() + dim);
        std::vector<int64_t> extra_ids{900000};
        std::vector<std::string> extra_paths{"a/b"};
        std::vector<std::string> extra_sources{"incremental-alias"};
        auto extra =
            MakePyramidCacheDataset(1, dim, extra_vector, extra_ids, extra_paths, extra_sources);
        for (const auto& index : {cold, warm, restored}) {
            REQUIRE(index->Add(extra).has_value());
            auto query = MakePyramidCacheQuery(dim, extra_vector, "a/b");
            auto result = index->KnnSearch(query, 3, R"({"pyramid":{"ef_search":256}})");
            REQUIRE(result.has_value());
            REQUIRE(result.value()->GetDim() == 3);
            std::vector<int64_t> actual(result.value()->GetIds(), result.value()->GetIds() + 3);
            std::sort(actual.begin(), actual.end());
            REQUIRE(actual == std::vector<int64_t>{10000, 10001, 900000});
        }
    }
}

TEST_CASE("Pyramid duplicate cache restores more than 1024 representatives",
          "[ft][pyramid][cache][duplicate_cache_large][pr]") {
    const auto root = GENERATE(std::string("single_layer"), std::string("multi_layer"));
    const auto storage = GENERATE(std::string("flat"), std::string("compressed"));
    CheckLargePyramidDuplicateCache(root, storage, 1);
}

TEST_CASE("Pyramid parallel duplicate admission preserves connectivity and cache hits",
          "[ft][pyramid][cache][duplicate_connectivity][pr][concurrent]") {
    const auto root = GENERATE(std::string("single_layer"), std::string("multi_layer"));
    const auto storage = GENERATE(std::string("flat"), std::string("compressed"));
    const auto threads = GENERATE(4, 32);
    CheckLargePyramidDuplicateCache(root, storage, threads, threads);
}

TEST_CASE("Pyramid production construction duplicate connectivity diagnostic",
          "[ft][pyramid][cache][duplicate_connectivity_production][pr][concurrent]") {
    const auto root = GENERATE(std::string("single_layer"), std::string("multi_layer"));
    const auto storage = GENERATE(std::string("flat"), std::string("compressed"));
    CheckLargePyramidDuplicateCache(root, storage, 32, 32, 2, true);
}

namespace {

struct DuplicateClassFixture {
    static constexpr int64_t dim = 2;
    std::vector<int64_t> ids;
    std::vector<float> vectors;
    std::vector<std::string> paths;
    std::vector<std::string> sources;
    std::string params;

    explicit DuplicateClassFixture(const std::string& root) {
        params = fmt::format(R"({{
            "dtype":"float32", "metric_type":"l2", "dim":2,
            "index_param":{{"base_quantization_type":"fp32", "max_degree":16,
            "ef_construction":64, "build_thread_count":1, "root_graph_type":"{}",
            "index_min_size":8, "persist_source_id":true, "store_paths":true,
            "support_duplicate":true}} }})",
                             root);
        for (int64_t i = 0; i < 100; ++i) {
            ids.push_back(1000 + i);
            sources.push_back("focused-class-" + std::to_string(i));
            paths.push_back("a/b");
            vectors.push_back(static_cast<float>(i / 10));
            vectors.push_back(static_cast<float>((i / 10) * 2));
        }
    }

    void
    CopyVector(uint64_t target, uint64_t source) {
        for (int64_t d = 0; d < dim; ++d) {
            vectors[target * dim + d] = vectors[source * dim + d];
        }
    }

    auto
    Dataset() const {
        return MakePyramidCacheDataset(ids.size(), dim, vectors, ids, paths, sources);
    }

    void
    Check(const vsag::IndexPtr& index) const {
        // Full population, not top-k tie order: enforce every unique label and exact FP32 L2.
        for (const auto& path : {std::string{}, std::string("a/b")}) {
            for (const auto offset : {uint64_t{0}, uint64_t{10}, uint64_t{20}}) {
                CAPTURE(path, offset);
                std::vector<float> vector(vectors.begin() + offset * dim,
                                          vectors.begin() + (offset + 1) * dim);
                auto query = MakePyramidCacheQuery(dim, vector, path);
                auto result =
                    index->KnnSearch(query, ids.size(), R"({"pyramid":{"ef_search":256}})");
                REQUIRE(result.has_value());
                REQUIRE(result.value()->GetDim() == static_cast<int64_t>(ids.size()));
                std::vector<int64_t> actual;
                for (int64_t j = 0; j < result.value()->GetDim(); ++j) {
                    const auto label = result.value()->GetIds()[j];
                    const auto found = std::find(ids.begin(), ids.end(), label);
                    REQUIRE(found != ids.end());
                    const auto row = static_cast<uint64_t>(found - ids.begin());
                    float distance = 0.0F;
                    for (int64_t d = 0; d < dim; ++d) {
                        const auto delta = vector[d] - vectors[row * dim + d];
                        distance += delta * delta;
                    }
                    REQUIRE(result.value()->GetDistances()[j] == distance);
                    actual.push_back(label);
                }
                std::sort(actual.begin(), actual.end());
                auto expected = ids;
                std::sort(expected.begin(), expected.end());
                REQUIRE(actual == expected);
            }
        }
    }
};

}  // namespace

TEST_CASE("Pyramid cache class demotion and membership threshold",
          "[ft][pyramid][cache][duplicate_class_validation][pr]") {
    const auto root = GENERATE(std::string("single_layer"), std::string("multi_layer"));
    const auto change = GENERATE(std::string("three-merged-classes"),
                                 std::string("split-matches-stable"),
                                 std::string("exactly-80-percent"),
                                 std::string("79-percent"));
    CAPTURE(root, change);
    DuplicateClassFixture fixture(root);
    // For threshold cases: classes 0/1/2 have sizes 10/11/9; all others have size 10.
    // This allows 80 and 79 surviving memberships with all 100 SourceIDs retained.
    if (change == "exactly-80-percent" || change == "79-percent") {
        fixture.CopyVector(20, 10);
    }
    auto cold = vsag::Factory::CreateIndex("pyramid", fixture.params).value();
    REQUIRE(cold->Build(fixture.Dataset()).has_value());
    std::stringstream cache;
    REQUIRE(cold->ExportCache(cache).has_value());
    const auto old_sources = fixture.sources;
    const auto old_ids = fixture.ids;
    int64_t expected_hits = 0;
    if (change == "three-merged-classes") {
        // A third collision must also demote: retaining it incorrectly reaches the 80% gate.
        for (uint64_t i = 10; i < 30; ++i) {
            fixture.CopyVector(i, 0);
        }
    } else if (change == "split-matches-stable") {
        fixture.CopyVector(1, 10);
        expected_hits = 90;
    } else {
        // Split entire classes by changing one alias, not their physical representative.
        fixture.vectors[1 * fixture.dim] = 100.0F;
        const uint64_t second_alias = change == "exactly-80-percent" ? 31 : 11;
        fixture.vectors[second_alias * fixture.dim] = 101.0F;
        expected_hits = change == "exactly-80-percent" ? 80 : 0;
    }
    REQUIRE(fixture.sources == old_sources);
    REQUIRE(fixture.ids == old_ids);
    auto warm = vsag::Factory::CreateIndex("pyramid", fixture.params).value();
    REQUIRE(warm->ImportCache(cache).has_value());
    REQUIRE(warm->Build(fixture.Dataset()).has_value());
    const auto stats = vsag::JsonType::Parse(warm->GetStats());
    REQUIRE(stats["build_cache_hit_nodes"].GetInt() == expected_hits);
    REQUIRE(stats["build_cache_missed_nodes"].GetInt() == 100 - expected_hits);
    if (expected_hits != 0) {
        REQUIRE(stats["build_cache_restored_edges"].GetInt() > 0);
    } else {
        REQUIRE(stats["build_cache_restored_edges"].GetInt() == 0);
    }
    fixture.Check(warm);
    auto serialized = warm->Serialize();
    REQUIRE(serialized.has_value());
    auto restored = vsag::Factory::CreateIndex("pyramid", fixture.params).value();
    REQUIRE(restored->Deserialize(serialized.value()).has_value());
    fixture.Check(restored);
}

TEST_CASE("Pyramid cached owner survives filtered empty row and equal misses",
          "[ft][pyramid][cache][duplicate_class_validation][pr]") {
    const auto root = GENERATE(std::string("single_layer"), std::string("multi_layer"));
    CAPTURE(root);
    DuplicateClassFixture fixture(root);
    auto cold = vsag::Factory::CreateIndex("pyramid", fixture.params).value();
    REQUIRE(cold->Build(fixture.Dataset()).has_value());
    std::stringstream exported;
    REQUIRE(cold->ExportCache(exported).has_value());
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidBuildCache cache(allocator.get());
    vsag::IOStreamReader reader(exported);
    cache.Deserialize(reader);
    for (const auto& path : {std::string{}, std::string("a"), std::string("a/b")}) {
        auto* graph = cache.GetGraphCache("", path);
        const auto* owners = cache.GetGroupOwners("", path);
        REQUIRE(graph != nullptr);
        REQUIRE(owners != nullptr);
        auto owner_of = [&](const std::string& source) {
            const auto found =
                std::find(graph->source_ids_.begin(), graph->source_ids_.end(), source);
            REQUIRE(found != graph->source_ids_.end());
            return (*owners)[static_cast<uint64_t>(found - graph->source_ids_.begin())];
        };
        const auto invalid_owner = owner_of(fixture.sources[0]);
        const auto stable_owner = owner_of(fixture.sources[10]);
        REQUIRE(invalid_owner != stable_owner);
        // Valid wire row, but its sole edge will be filtered when class 0 is demoted.
        auto& row = graph->neighbors_.at(graph->source_ids_[stable_owner]);
        row.clear();
        row.push_back(stable_owner);
        row.push_back(invalid_owner);
    }
    std::stringstream modified;
    vsag::IOStreamWriter writer(modified);
    cache.Serialize(writer);
    fixture.CopyVector(1, 10);  // Demote class 0; its miss must join retained class 1.
    auto warm = vsag::Factory::CreateIndex("pyramid", fixture.params).value();
    REQUIRE(warm->ImportCache(modified).has_value());
    REQUIRE(warm->Build(fixture.Dataset()).has_value());
    const auto stats = vsag::JsonType::Parse(warm->GetStats());
    REQUIRE(stats["build_cache_hit_nodes"].GetInt() == 90);
    REQUIRE(stats["build_cache_missed_nodes"].GetInt() == 10);
    fixture.Check(warm);
    auto binary = warm->Serialize();
    REQUIRE(binary.has_value());
    auto restored = vsag::Factory::CreateIndex("pyramid", fixture.params).value();
    REQUIRE(restored->Deserialize(binary.value()).has_value());
    fixture.Check(restored);
    std::vector<float> added(fixture.vectors.begin() + 10 * fixture.dim,
                             fixture.vectors.begin() + 11 * fixture.dim);
    auto one = MakePyramidCacheDataset(1, fixture.dim, added, {2000}, {"a/b"}, {"focused-added"});
    REQUIRE(warm->Add(one).has_value());
    REQUIRE(restored->Add(one).has_value());
    fixture.ids.push_back(2000);
    fixture.sources.push_back("focused-added");
    fixture.paths.push_back("a/b");
    fixture.vectors.insert(fixture.vectors.end(), added.begin(), added.end());
    fixture.Check(warm);
    fixture.Check(restored);
    auto after_add = restored->Serialize();
    REQUIRE(after_add.has_value());
    auto final_index = vsag::Factory::CreateIndex("pyramid", fixture.params).value();
    REQUIRE(final_index->Deserialize(after_add.value()).has_value());
    fixture.Check(final_index);
}

TEST_CASE("Pyramid cached source component remains reachable from routed entries",
          "[ft][pyramid][cache][duplicate_class_validation][duplicate_return_connectivity][pr]") {
    const auto root = GENERATE(std::string("single_layer"), std::string("multi_layer"));
    const auto storage = GENERATE(std::string("flat"), std::string("compressed"));
    CAPTURE(root, storage);
    DuplicateClassFixture fixture(root);
    auto params = vsag::JsonType::Parse(fixture.params);
    params["index_param"]["graph_storage_type"].SetString(storage);
    fixture.params = params.Dump();
    auto cold = vsag::Factory::CreateIndex("pyramid", fixture.params).value();
    REQUIRE(cold->Build(fixture.Dataset()).has_value());
    std::stringstream exported;
    REQUIRE(cold->ExportCache(exported).has_value());
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidBuildCache cache(allocator.get());
    vsag::IOStreamReader reader(exported);
    cache.Deserialize(reader);
    for (const auto& path : {std::string{}, std::string("a"), std::string("a/b")}) {
        auto* graph = cache.GetGraphCache("", path);
        const auto* owners = cache.GetGroupOwners("", path);
        REQUIRE(graph != nullptr);
        REQUIRE(owners != nullptr);
        std::vector<vsag::InnerIdType> representatives;
        for (vsag::InnerIdType id = 0; id < owners->size(); ++id) {
            if ((*owners)[id] == id) {
                representatives.push_back(id);
            }
        }
        REQUIRE(representatives.size() == 10);
        // The first cached owner can reach every other owner, but nothing can return
        // to it. This passes a forward-only repair and fails routed-entry reachability.
        for (uint64_t i = 0; i < representatives.size(); ++i) {
            const auto id = representatives[i];
            auto& row = graph->neighbors_.at(graph->source_ids_[id]);
            row.clear();
            row.push_back(id);
            row.push_back(representatives[i == 0 ? 1 : (i == 9 ? 1 : i + 1)]);
        }
    }
    std::stringstream modified;
    vsag::IOStreamWriter writer(modified);
    cache.Serialize(writer);
    auto warm = vsag::Factory::CreateIndex("pyramid", fixture.params).value();
    REQUIRE(warm->ImportCache(modified).has_value());
    REQUIRE(warm->Build(fixture.Dataset()).has_value());
    REQUIRE(vsag::JsonType::Parse(warm->GetStats())["build_cache_hit_nodes"].GetInt() == 100);
    fixture.Check(warm);
    std::stringstream checked;
    REQUIRE(warm->ExportCache(checked).has_value());
    vsag::PyramidBuildCache restored(allocator.get());
    vsag::IOStreamReader checked_reader(checked);
    restored.Deserialize(checked_reader);
    for (const auto& path : {std::string{}, std::string("a"), std::string("a/b")}) {
        const auto* graph = restored.GetGraphCache("", path);
        const auto* owners = restored.GetGroupOwners("", path);
        REQUIRE(graph != nullptr);
        REQUIRE(owners != nullptr);
        for (vsag::InnerIdType start = 0; start < owners->size(); ++start) {
            if ((*owners)[start] != start) {
                continue;
            }
            std::vector<bool> seen(owners->size());
            std::vector<vsag::InnerIdType> queue{start};
            seen[start] = true;
            for (uint64_t offset = 0; offset < queue.size(); ++offset) {
                const auto id = queue[offset];
                const auto* row = graph->FindNeighborInnerIds(graph->source_ids_[id]);
                REQUIRE(row != nullptr);
                REQUIRE(row->size() <= 17);
                for (uint64_t edge = 1; edge < row->size(); ++edge) {
                    const auto target = (*row)[edge];
                    REQUIRE(target < owners->size());
                    REQUIRE((*owners)[target] == target);
                    if (not seen[target]) {
                        seen[target] = true;
                        queue.push_back(target);
                    }
                }
            }
            REQUIRE(queue.size() == 10);
        }
    }
}

TEST_CASE("Pyramid warm cache preserves raw vectors and precise distances",
          "[ft][pyramid][cache][raw_vector][duplicate_raw_cache][pr]") {
    const auto duplicate = GENERATE(false, true);
    const auto root = GENERATE(std::string("single_layer"), std::string("multi_layer"));
    DuplicateClassFixture fixture(root);
    auto params = vsag::JsonType::Parse(fixture.params);
    params["index_param"]["support_duplicate"].SetBool(duplicate);
    params["index_param"]["base_quantization_type"].SetString("sq8");
    params["index_param"]["store_raw_vector"].SetBool(true);
    fixture.params = params.Dump();
    auto cold = vsag::Factory::CreateIndex("pyramid", fixture.params).value();
    REQUIRE(cold->Build(fixture.Dataset()).has_value());
    std::stringstream cache;
    REQUIRE(cold->ExportCache(cache).has_value());
    auto warm = vsag::Factory::CreateIndex("pyramid", fixture.params).value();
    REQUIRE(warm->ImportCache(cache).has_value());
    REQUIRE(warm->Build(fixture.Dataset()).has_value());
    REQUIRE(vsag::JsonType::Parse(warm->GetStats())["build_cache_hit_nodes"].GetInt() > 0);
    auto binary = warm->Serialize();
    REQUIRE(binary.has_value());
    auto restored = vsag::Factory::CreateIndex("pyramid", fixture.params).value();
    REQUIRE(restored->Deserialize(binary.value()).has_value());
    for (const auto& index : {cold, warm, restored}) {
        auto raw = index->GetRawVectorByIds(fixture.ids.data(), fixture.ids.size(), nullptr);
        REQUIRE(raw.has_value());
        REQUIRE(std::equal(
            fixture.vectors.begin(), fixture.vectors.end(), raw.value()->GetFloat32Vectors()));
        for (uint64_t row = 0; row < fixture.ids.size(); ++row) {
            auto distance = index->CalcDistanceById(fixture.vectors.data(), fixture.ids[row], true);
            REQUIRE(distance.has_value());
            float expected = 0;
            for (int64_t d = 0; d < fixture.dim; ++d) {
                const auto delta = fixture.vectors[d] - fixture.vectors[row * fixture.dim + d];
                expected += delta * delta;
            }
            REQUIRE(distance.value() == expected);
        }
    }
}

namespace {
struct DuplicateMultiPathFixture {
    static constexpr int64_t DIM = 2;
    std::vector<int64_t> ids;
    std::vector<float> vectors;
    std::vector<std::string> sources;
    std::vector<std::vector<std::string>> paths;

    DuplicateMultiPathFixture() {
        for (int64_t i = 0; i < 120; ++i) {
            ids.push_back(1000 + i);
            const auto group = i / 2;
            vectors.push_back(static_cast<float>(group));
            vectors.push_back(static_cast<float>(group * 2));
            sources.push_back("multipath-source-" + std::to_string(i));
            paths.push_back(i % 2 == 0
                                ? std::vector<std::string>{"a/x", "a/y", "a/x", "literal|pipe"}
                                : std::vector<std::string>{"a/x", "b/z"});
        }
    }

    auto
    Dataset() -> vsag::DatasetPtr {
        return vsag::Dataset::Make()
            ->NumElements(static_cast<int64_t>(ids.size()))
            ->Dim(DIM)
            ->Ids(ids.data())
            ->Float32Vectors(vectors.data())
            ->SourceID(sources.data())
            ->Paths("tag", paths)
            ->Owner(false);
    }

    void
    Reverse() {
        std::reverse(ids.begin(), ids.end());
        std::reverse(sources.begin(), sources.end());
        std::reverse(paths.begin(), paths.end());
        for (uint64_t left = 0; left < ids.size() / 2; ++left) {
            const auto right = ids.size() - 1 - left;
            for (int64_t d = 0; d < DIM; ++d) {
                std::swap(vectors[left * DIM + d], vectors[right * DIM + d]);
            }
        }
    }

    void
    Check(const vsag::IndexPtr& index) const {
        const std::vector<std::vector<std::string>> queries{
            {""}, {"a"}, {"a/x"}, {"a/y"}, {"b/z"}, {"a/x", "a/y", "a/x"}, {"literal|pipe"}};
        const float query_vector[2]{0.0F, 0.0F};
        for (const auto& query_paths : queries) {
            CAPTURE(query_paths);
            std::vector<int64_t> expected;
            for (uint64_t row = 0; row < ids.size(); ++row) {
                bool matches = false;
                for (const auto& prefix : query_paths) {
                    for (const auto& path : paths[row]) {
                        matches = matches || prefix.empty() || path == prefix ||
                                  path.rfind(prefix + "/", 0) == 0;
                    }
                }
                if (matches) {
                    expected.push_back(ids[row]);
                }
            }
            auto query = vsag::Dataset::Make()
                             ->NumElements(1)
                             ->Dim(DIM)
                             ->Float32Vectors(query_vector)
                             ->Paths("tag", std::vector<std::vector<std::string>>{query_paths})
                             ->Owner(false);
            auto result =
                index->KnnSearch(query,
                                 static_cast<int64_t>(ids.size()),
                                 R"({"pyramid":{"ef_search":512,"hierarchies":["tag"]}})");
            REQUIRE(result.has_value());
            REQUIRE(result.value()->GetDim() == static_cast<int64_t>(expected.size()));
            std::vector<int64_t> actual;
            for (int64_t offset = 0; offset < result.value()->GetDim(); ++offset) {
                const auto label = result.value()->GetIds()[offset];
                const auto found = std::find(ids.begin(), ids.end(), label);
                REQUIRE(found != ids.end());
                const auto row = static_cast<uint64_t>(found - ids.begin());
                const auto x = vectors[row * DIM];
                const auto y = vectors[row * DIM + 1];
                REQUIRE(result.value()->GetDistances()[offset] == x * x + y * y);
                actual.push_back(label);
            }
            std::sort(actual.begin(), actual.end());
            std::sort(expected.begin(), expected.end());
            REQUIRE(actual == expected);  // Includes duplicate-label exclusion in path unions.
        }
    }
};
}  // namespace

TEST_CASE("Pyramid duplicate cache preserves multipath memberships",
          "[ft][pyramid][cache][duplicate_multipath_cache][concurrent][pr]") {
    const auto root = GENERATE(std::string("single_layer"), std::string("multi_layer"));
    const auto threads = GENERATE(4, 32);
    const auto move_owner = GENERATE(false, true);
    CAPTURE(root, threads, move_owner);
    auto params = vsag::JsonType::Parse(R"({
        "dtype":"float32", "metric_type":"l2", "dim":2,
        "index_param":{"graph_type":"nsw", "base_quantization_type":"fp32",
            "use_reorder":false, "index_min_size":8, "max_degree":16,
            "ef_construction":64, "support_duplicate":true,
            "persist_source_id":true, "store_paths":true,
            "hierarchies":[{"name":"tag", "no_build_levels":[]} ]}})");
    params["index_param"]["build_thread_count"].SetInt(threads);
    // Match established root_graph_type handling in existing duplicate-cache fixtures.
    params["index_param"]["root_graph_type"].SetString(root);
    const auto build_params = params.Dump();
    DuplicateMultiPathFixture fixture;
    auto day1 = vsag::Factory::CreateIndex("pyramid", build_params).value();
    REQUIRE(day1->Build(fixture.Dataset()).has_value());
    fixture.Check(day1);
    std::stringstream exported;
    REQUIRE(day1->ExportCache(exported).has_value());
    const auto cache_bytes = exported.str();
    if (move_owner) {
        auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
        vsag::PyramidBuildCache metadata(allocator.get());
        std::stringstream input(cache_bytes);
        vsag::IOStreamReader reader(input);
        metadata.Deserialize(reader);
        const auto* graph = metadata.GetGraphCache("tag", "a/x");
        const auto* owners = metadata.GetGroupOwners("tag", "a/x");
        REQUIRE(graph != nullptr);
        REQUIRE(owners != nullptr);
        uint64_t owner = 0;
        while (owner < owners->size() && (*owners)[owner] != owner) {
            ++owner;
        }
        REQUIRE(owner < owners->size());
        const auto found =
            std::find(fixture.sources.begin(), fixture.sources.end(), graph->source_ids_[owner]);
        REQUIRE(found != fixture.sources.end());
        const auto row = static_cast<uint64_t>(found - fixture.sources.begin());
        // Drop only this actual owner's a/x membership, retaining its other paths and SourceID.
        auto& paths = fixture.paths[row];
        paths.erase(std::remove(paths.begin(), paths.end(), "a/x"), paths.end());
        REQUIRE(not paths.empty());
    }
    fixture.Reverse();
    auto cold = vsag::Factory::CreateIndex("pyramid", build_params).value();
    REQUIRE(cold->Build(fixture.Dataset()).has_value());
    fixture.Check(cold);
    auto warm = vsag::Factory::CreateIndex("pyramid", build_params).value();
    std::stringstream cache_input(cache_bytes);
    REQUIRE(warm->ImportCache(cache_input).has_value());
    REQUIRE(warm->Build(fixture.Dataset()).has_value());
    fixture.Check(warm);
    const auto stats = vsag::JsonType::Parse(warm->GetStats());
    REQUIRE(stats["build_cache_hit_nodes"].GetInt() > 0);
    REQUIRE(stats["build_cache_restored_edges"].GetInt() > 0);
    if (not move_owner) {
        REQUIRE(stats["build_cache_hit_nodes"].GetInt() == 120);
    }
    auto serialized = warm->Serialize();
    REQUIRE(serialized.has_value());
    auto restored = vsag::Factory::CreateIndex("pyramid", build_params).value();
    REQUIRE(restored->Deserialize(serialized.value()).has_value());
    fixture.Check(restored);
    float added_vector[2]{0.0F, 0.0F};
    int64_t added_id = 9999;
    std::string added_source = "multipath-added";
    const std::vector<std::string> added_paths{"a/x", "a/y", "a/x"};
    auto added = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(2)
                     ->Float32Vectors(added_vector)
                     ->Ids(&added_id)
                     ->SourceID(&added_source)
                     ->Paths("tag", std::vector<std::vector<std::string>>{added_paths})
                     ->Owner(false);
    REQUIRE(restored->Add(added).has_value());
    fixture.ids.push_back(added_id);
    fixture.sources.push_back(added_source);
    fixture.vectors.insert(fixture.vectors.end(), {0.0F, 0.0F});
    fixture.paths.push_back(added_paths);
    fixture.Check(restored);
    auto after_add = restored->Serialize();
    REQUIRE(after_add.has_value());
    auto roundtrip = vsag::Factory::CreateIndex("pyramid", build_params).value();
    REQUIRE(roundtrip->Deserialize(after_add.value()).has_value());
    fixture.Check(roundtrip);
}
