
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

}  // namespace

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Build & ContinueAdd Test",
                             "[ft][pyramid]") {
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
                             "[ft][pyramid]") {
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
                             "[ft][pyramid]") {
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
                             "[ft][pyramid]") {
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
                             "[ft][pyramid]") {
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

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex, "Pyramid Add Test", "[ft][pyramid]") {
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
                             "[ft][pyramid]") {
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

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex, "Pyramid No Path Test", "[ft][pyramid]") {
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
                             "[ft][pyramid][serialization]") {
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

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex, "Pyramid Clone", "[ft][pyramid]") {
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
                             "Pyramid Build Test With Random Allocator",
                             "[ft][pyramid]") {
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
                             "[ft][pyramid][concurrent]") {
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

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex, "Pyramid OverTime Test", "[ft][pyramid]") {
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
                             "[ft][pyramid][concurrent]") {
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
                             "[ft][pyramid]") {
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
                             "[ft][pyramid][analyzer]") {
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
