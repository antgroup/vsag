
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

#include "fixtures/test_dataset_pool.h"
#include "fixtures/test_logger.h"
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

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Build & ContinueAdd Test",
                             "[ft][pyramid]") {
    auto metric_type = GENERATE("l2", "ip", "cosine");
    auto use_reorder = GENERATE(true, false);
    auto immutable = GENERATE(true, false);
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
        TestKnnSearch(index, dataset, search_param, 0.96, true);
        TestFilterSearch(index, dataset, search_param, 0.96, true);
        TestRangeSearch(index, dataset, search_param, 0.96, 10, true);
        TestRangeSearch(index, dataset, search_param, 0.49, 5, true);
    }
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
        TestKnnSearch(index, dataset, search_param, 0.96, true);
        TestFilterSearch(index, dataset, search_param, 0.96, true);
        TestRangeSearch(index, dataset, search_param, 0.96, 10, true);
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
            TestKnnSearch(index, dataset, search_param, 0.96, true);
            TestFilterSearch(index, dataset, search_param, 0.96, true);
            TestRangeSearch(index, dataset, search_param, 0.96, 10, true);
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
            TestKnnSearch(index, dataset, search_param, 0.96, has_root);
            TestFilterSearch(index, dataset, search_param, 0.96, has_root);
            TestRangeSearch(index, dataset, search_param, 0.96, 10, has_root);
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
        TestConcurrentKnnSearch(index, dataset, search_param, 0.96, true);
        TestCalcDistanceById(index, dataset, 1e-5, true);
    }
    for (auto& dim : dims) {
        auto param = GeneratePyramidBuildParametersString(metric_type, dim, pyramid_param);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type, /*with_path=*/true);
        TestConcurrentAddSearch(index, dataset, search_param, 0.96, true);
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
