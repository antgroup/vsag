
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

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <fstream>
#include <string>
#include <vector>

#include "functest.h"
#include "test_index.h"
#include "vsag/vsag.h"

namespace fixtures {

struct MCIParam {
    std::string base_quantization_type = "sq8";
    std::string base_codes_type = "flatten";
    uint64_t max_degree = 16;
    uint64_t mcs = 32;
    uint64_t clique_max = 12;
    float alpha = 1.2F;
    float join_ratio_threshold = 0.6F;
    uint64_t added_mct = 3;
    bool use_reorder = false;
    bool use_hgraph_hybrid = false;
    float hgraph_valid_ratio_threshold = 0.5F;
    std::string hgraph_index_path;
    int64_t hgraph_ef_search = 32;
};

class MCITestIndex : public fixtures::TestIndex {
public:
    static TestDatasetPool pool;
    static std::vector<int> dims;

    constexpr static uint64_t base_count = 800;
    constexpr static const char* search_param_tmp = R"(
        {{
            "mci": {{
                "ef_search": {},
                "seed_count": {},
                "rabitq_one_bit_search": {}
            }}
        }})";

    static std::string
    GenerateBuildParametersString(const std::string& metric_type,
                                  int64_t dim,
                                  const MCIParam& param) {
        constexpr auto parameter_temp = R"(
        {{
            "dtype": "float32",
            "metric_type": "{}",
            "dim": {},
            "index_param": {{
                "base_quantization_type": "{}",
                "base_codes_type": "{}",
                "max_degree": {},
                "mcs": {},
                "clique_max": {},
                "alpha": {},
                "join_ratio_threshold": {},
                "added_mct": {},
                "use_reorder": {},
                "use_hgraph_hybrid": {},
                "hgraph_valid_ratio_threshold": {},
                "hgraph_index_path": "{}",
                "hgraph_ef_search": {}{}
            }}
        }})";

        const std::string hgraph_index_param_block = param.use_hgraph_hybrid ? std::string(R"(,
                "hgraph_index_param": {
                    "base_quantization_type": "fp32",
                    "graph_type": "odescent",
                    "max_degree": 16,
                    "alpha": 1.2,
                    "graph_iter_turn": 15,
                    "neighbor_sample_rate": 0.3
                })")
                                                                             : std::string("");

        return fmt::format(parameter_temp,
                           metric_type,
                           dim,
                           param.base_quantization_type,
                           param.base_codes_type,
                           param.max_degree,
                           param.mcs,
                           param.clique_max,
                           param.alpha,
                           param.join_ratio_threshold,
                           param.added_mct,
                           param.use_reorder,
                           param.use_hgraph_hybrid,
                           param.hgraph_valid_ratio_threshold,
                           param.hgraph_index_path,
                           param.hgraph_ef_search,
                           hgraph_index_param_block);
    }

    static std::string
    GenerateSearchParametersString(int64_t ef_search,
                                   int64_t seed_count,
                                   bool rabitq_one_bit_search = false) {
        return fmt::format(
            search_param_tmp, ef_search, seed_count, rabitq_one_bit_search ? "true" : "false");
    }
};

TestDatasetPool MCITestIndex::pool{};
std::vector<int> MCITestIndex::dims = fixtures::get_common_used_dims(1, RandomValue(0, 999));

}  // namespace fixtures

TEST_CASE_PERSISTENT_FIXTURE(fixtures::MCITestIndex,
                             "MCI Build & Search Test",
                             "[ft][build][search][mci]") {
    auto metric_type = GENERATE("l2", "ip", "cosine");
    auto base_quantization_type = GENERATE("fp32", "sq8");
    fixtures::MCIParam mci_param;
    mci_param.base_quantization_type = base_quantization_type;

    const std::string name = "mci";
    auto search_param = GenerateSearchParametersString(80, 32);
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}, base_quantization_type={}",
                         metric_type,
                         dim,
                         base_quantization_type));
        auto param = GenerateBuildParametersString(metric_type, dim, mci_param);
        auto index = TestFactory(name, param, true);
        REQUIRE(index->GetIndexType() == vsag::IndexType::MCI);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestBuildIndex(index, dataset, true);
        TestKnnSearch(index, dataset, search_param, 0.85, true);
        TestRangeSearch(index, dataset, search_param, 0.85, 10, true);
        TestRangeSearch(index, dataset, search_param, 0.49, 5, true);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::MCITestIndex,
                             "MCI Filter Search Test",
                             "[ft][filter][search][mci]") {
    auto metric_type = GENERATE("l2", "cosine");
    fixtures::MCIParam mci_param;
    const std::string name = "mci";
    auto search_param = GenerateSearchParametersString(120, 48);
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}", metric_type, dim));
        auto param = GenerateBuildParametersString(metric_type, dim, mci_param);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestBuildIndex(index, dataset, true);
        TestFilterSearch(index, dataset, search_param, 0.85, true);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::MCITestIndex,
                             "MCI Incremental Add Test",
                             "[ft][add][search][mci]") {
    auto metric_type = GENERATE("l2", "ip", "cosine");
    fixtures::MCIParam mci_param;
    const std::string name = "mci";
    auto search_param = GenerateSearchParametersString(120, 48);
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}", metric_type, dim));
        auto param = GenerateBuildParametersString(metric_type, dim, mci_param);
        auto index = TestFactory(name, param, true);
        REQUIRE(index->GetIndexType() == vsag::IndexType::MCI);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        // Build on the first half, then incrementally Add the rest one-by-one so
        // the clique index is maintained via incremental_update_clique.
        TestContinueAddIgnoreRequire(index, dataset, 0.5);
        REQUIRE(index->GetNumElements() == static_cast<int64_t>(base_count));
        TestKnnSearch(index, dataset, search_param, 0.85, true);
        TestFilterSearch(index, dataset, search_param, 0.85, true);
        auto reloaded = TestFactory(name, param, true);
        TestSerializeBinarySet(index, reloaded, dataset, search_param, true);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::MCITestIndex,
                             "MCI HGraph Hybrid Batch Add Test",
                             "[ft][add][search][mci]") {
    auto metric_type = GENERATE("l2");
    fixtures::MCIParam mci_param;
    mci_param.use_hgraph_hybrid = true;
    const std::string name = "mci";
    auto search_param = GenerateSearchParametersString(120, 48);
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}", metric_type, dim));
        auto param = GenerateBuildParametersString(metric_type, dim, mci_param);
        auto index = TestFactory(name, param, true);
        REQUIRE(index->GetIndexType() == vsag::IndexType::MCI);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);

        const auto prefix_count = static_cast<int64_t>(base_count / 2);
        const auto* paths = dataset->base_->GetPaths();
        auto prefix = vsag::Dataset::Make();
        prefix->Dim(dim)
            ->Ids(dataset->base_->GetIds())
            ->NumElements(prefix_count)
            ->Paths(paths)
            ->Float32Vectors(dataset->base_->GetFloat32Vectors())
            ->Owner(false);
        auto build_result = index->Build(prefix);
        REQUIRE(build_result.has_value());

        auto tail = vsag::Dataset::Make();
        tail->Dim(dim)
            ->Ids(dataset->base_->GetIds() + prefix_count)
            ->NumElements(static_cast<int64_t>(base_count) - prefix_count)
            ->Paths(paths == nullptr ? nullptr : paths + prefix_count)
            ->Float32Vectors(dataset->base_->GetFloat32Vectors() + prefix_count * dim)
            ->Owner(false);
        auto add_result = index->Add(tail);
        REQUIRE(add_result.has_value());
        REQUIRE(add_result.value().empty());
        REQUIRE(index->GetNumElements() == static_cast<int64_t>(base_count));

        TestKnnSearch(index, dataset, search_param, 0.85, true);
        TestFilterSearch(index, dataset, search_param, 0.85, true);

        auto serialize_binary = index->Serialize();
        REQUIRE(serialize_binary.has_value());
        auto reloaded = TestFactory(name, param, true);
        auto deserialize_result = reloaded->Deserialize(serialize_binary.value());
        REQUIRE(deserialize_result.has_value());
        REQUIRE(reloaded->GetNumElements() == static_cast<int64_t>(base_count));

        auto query = fixtures::get_one_query(dataset->query_, 0);
        auto result = reloaded->KnnSearch(query, 10, search_param);
        REQUIRE(result.has_value());
        auto stats = result.value()->GetStatistics({"mci_hybrid_route"});
        REQUIRE(stats.size() == 1);
        REQUIRE(stats[0].find("hgraph") != std::string::npos);
        TestKnnSearch(reloaded, dataset, search_param, 0.85, true);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::MCITestIndex,
                             "MCI Reorder Test",
                             "[ft][build][search][reorder][mci]") {
    auto metric_type = GENERATE("l2");
    fixtures::MCIParam mci_param;
    mci_param.base_quantization_type = "rabitq";
    mci_param.use_reorder = true;
    const std::string name = "mci";
    auto search_param = GenerateSearchParametersString(120, 48);
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}", metric_type, dim));
        auto param = GenerateBuildParametersString(metric_type, dim, mci_param);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestBuildIndex(index, dataset, true);
        TestKnnSearch(index, dataset, search_param, 0.85, true);
        auto rabitq_search_param =
            GenerateSearchParametersString(120, 48, /*rabitq_one_bit_search=*/true);
        TestKnnSearch(index, dataset, rabitq_search_param, 0.85, true);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::MCITestIndex,
                             "MCI Concurrent Knn Search Test",
                             "[ft][concurrent][mci]") {
    auto metric_type = GENERATE("l2");
    fixtures::MCIParam mci_param;
    const std::string name = "mci";
    auto search_param = GenerateSearchParametersString(120, 48);
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}", metric_type, dim));
        auto param = GenerateBuildParametersString(metric_type, dim, mci_param);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestBuildIndex(index, dataset, true);
        TestConcurrentKnnSearch(index, dataset, search_param, 0.85, true);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::MCITestIndex, "MCI Serialize Test", "[ft][serialize][mci]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip");
    auto use_reorder = GENERATE(true, false);
    fixtures::MCIParam mci_param;
    mci_param.use_reorder = use_reorder;
    if (use_reorder) {
        mci_param.base_quantization_type = "rabitq";
    }
    const std::string name = "mci";
    auto search_param = GenerateSearchParametersString(120, 48);
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}, use_reorder={}", metric_type, dim, use_reorder));
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GenerateBuildParametersString(metric_type, dim, mci_param);
        auto index = TestFactory(name, param, true);
        SECTION("serialize empty index") {
            auto index2 = TestFactory(name, param, true);
            auto serialize_binary = index->Serialize();
            REQUIRE(serialize_binary.has_value());
            auto deserialize_index = index2->Deserialize(serialize_binary.value());
            REQUIRE(deserialize_index.has_value());
        }
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
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

TEST_CASE_PERSISTENT_FIXTURE(fixtures::MCITestIndex,
                             "MCI Hybrid Overlay With Decoupled HGraph",
                             "[ft][hybrid][mci]") {
    auto metric_type = GENERATE("l2");
    auto dim = dims.front();
    INFO(fmt::format("metric_type={}, dim={}", metric_type, dim));

    fixtures::TempDir dir("mci_hybrid");
    const std::string hgraph_path = dir.path + "hgraph.index";

    constexpr auto hgraph_params_tmp = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "base_quantization_type": "fp32",
            "graph_type": "odescent",
            "max_degree": 16,
            "alpha": 1.2,
            "graph_iter_turn": 15,
            "neighbor_sample_rate": 0.3
        }}
    }})";
    auto hgraph_params = fmt::format(hgraph_params_tmp, metric_type, dim);

    auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
    auto hgraph_index = vsag::Factory::CreateIndex("hgraph", hgraph_params);
    REQUIRE(hgraph_index.has_value());
    auto hgraph_build_result = hgraph_index.value()->Build(dataset->base_);
    REQUIRE(hgraph_build_result.has_value());
    {
        std::ofstream output(hgraph_path, std::ios::binary);
        REQUIRE(output.is_open());
        auto serialize_result = hgraph_index.value()->Serialize(output);
        REQUIRE(serialize_result.has_value());
    }

    fixtures::MCIParam mci_param;
    mci_param.use_hgraph_hybrid = true;
    mci_param.hgraph_index_path = hgraph_path;
    auto mci_build_param = GenerateBuildParametersString(metric_type, dim, mci_param);
    auto mci_index = TestFactory("mci", mci_build_param, true);
    TestBuildIndex(mci_index, dataset, true);

    // Persist and reload the MCI index; deserialization should also reopen the
    // companion HGraph file referenced by hgraph_index_path.
    const std::string mci_path = dir.path + "mci.index";
    {
        std::ofstream output(mci_path, std::ios::binary);
        REQUIRE(output.is_open());
        auto serialize_result = mci_index->Serialize(output);
        REQUIRE(serialize_result.has_value());
    }
    auto reloaded = TestFactory("mci", mci_build_param, true);
    {
        std::ifstream input(mci_path, std::ios::binary);
        REQUIRE(input.is_open());
        auto deserialize_result = reloaded->Deserialize(input);
        REQUIRE(deserialize_result.has_value());
    }
    REQUIRE(reloaded->GetNumElements() == static_cast<int64_t>(base_count));

    // Hybrid routing should still produce reasonable filtered search results.
    auto search_param = GenerateSearchParametersString(120, 48);
    TestFilterSearch(reloaded, dataset, search_param, 0.85, true);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::MCITestIndex,
                             "MCI Build with Random Allocator",
                             "[ft][build][mci]") {
    auto allocator = std::make_shared<fixtures::RandomAllocator>();
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2");
    fixtures::MCIParam mci_param;
    const std::string name = "mci";
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}", metric_type, dim));
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GenerateBuildParametersString(metric_type, dim, mci_param);
        auto index = vsag::Factory::CreateIndex(name, param, allocator.get());
        if (not index.has_value()) {
            continue;
        }
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        auto build_result = index.value()->Build(dataset->base_);
        // Random allocator may inject failures; just check there is no crash.
        (void)build_result;
        vsag::Options::Instance().set_block_size_limit(origin_size);
    }
}
