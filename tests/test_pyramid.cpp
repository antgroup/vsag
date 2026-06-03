
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

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <nlohmann/json.hpp>
#include <set>

#include "functest.h"
#include "test_index.h"
#include "vsag/vsag.h"

struct PyramidParam {
    std::vector<int> no_build_levels = std::vector<int>{0, 1, 2};
    std::string base_quantization_type = "fp32";
    std::string precise_quantization_type = "fp32";
    std::string graph_type = "nsw";
    bool use_reorder = false;
    bool support_duplicate = false;
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
std::vector<int> PyramidTestIndex::dims = fixtures::get_common_used_dims(1, RandomValue(0, 999));
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
            "precise_quantization_type": "{}",
            "use_reorder": {},
            "index_min_size": 28,
            "support_duplicate": {}
        }}
    }}
    )";
    auto build_parameters_str = fmt::format(parameter_temp,
                                            metric_type,
                                            dim,
                                            fmt::join(param.no_build_levels, ","),
                                            param.graph_type,
                                            param.base_quantization_type,
                                            param.precise_quantization_type,
                                            param.use_reorder,
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

void
RequireDistancesNearZero(const vsag::DatasetPtr& result, const std::set<int64_t>& expected_ids) {
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        if (expected_ids.count(result->GetIds()[i]) != 0) {
            REQUIRE(std::abs(result->GetDistances()[i]) <= 1e-6F);
        }
    }
}

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
        INFO(fmt::format("metric_type={}, dim={}, use_reorder={}, immutable={}",
                         metric_type,
                         dim,
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
            TestFilterSearch(index, dataset, search_param, 0.94, has_root);
            TestRangeSearch(index, dataset, search_param, 0.94, 10, has_root);
            TestCalcDistanceById(index, dataset, 1e-5, true);
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
    }
    vsag::Options::Instance().set_block_size_limit(origin_size);
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
