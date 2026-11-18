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
#include <functional>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include "hgraph.h"
#include "impl/allocator/safe_allocator.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"
#include "vsag/dataset.h"
#include "vsag/factory.h"
#include "vsag/filter.h"

namespace {

vsag::DatasetPtr
MakeBuildDataset(std::vector<int64_t>& ids,
                 std::vector<float>& vectors,
                 int64_t count,
                 int64_t dim) {
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(count)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    return dataset;
}

}  // namespace

TEST_CASE("HGraph batch build stays searchable without an upper level",
          "[ut][hgraph][build][odescent]") {
    // Upper levels drive the entry point of a batch-built HGraph. The level sampler draws from a
    // fixed-seed engine, but mapping those draws to uniforms is implementation defined: on libc++
    // six points can all fall on level zero. The entry point then stayed INVALID_ENTRY_POINT, so
    // Build succeeded while every search returned an empty result set.
    constexpr int64_t dim = 2;
    constexpr int64_t total = 6;
    constexpr const char* params = R"(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 2,
            "index_param": {
                "base_quantization_type": "fp32",
                "graph_type": "odescent",
                "max_degree": 6,
                "alpha": 1.2,
                "graph_iter_turn": 6,
                "neighbor_sample_rate": 0.3
            }
        })";
    std::vector<int64_t> ids{0, 1, 2, 3, 4, 5};
    std::vector<float> vectors{0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6};

    auto index = vsag::Factory::CreateIndex("hgraph", params);
    REQUIRE(index.has_value());
    auto built = index.value()->Build(MakeBuildDataset(ids, vectors, total, dim));
    REQUIRE(built.has_value());
    REQUIRE(built.value().empty());
    REQUIRE(index.value()->GetNumElements() == total);

    auto result = index.value()->KnnSearch(MakeBuildDataset(ids, vectors, 1, dim),
                                           total,
                                           R"({"hgraph":{"ef_search":100,"use_mci":false}})",
                                           vsag::FilterPtr{});
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == total);
}

namespace vsag {
namespace {

// Schedule a competing reverse-edge update at the first point where the forward writer
// releases its node lock. This makes the lost-update interleaving reproducible without sleeps.
class CacheRefineInterleavingMutex final : public MutexArray {
public:
    CacheRefineInterleavingMutex(MutexArrayPtr wrapped, std::function<void()> update)
        : wrapped_(std::move(wrapped)), update_(std::move(update)) {
    }

    void
    Lock(uint32_t id) override {
        wrapped_->Lock(id);
    }
    void
    Unlock(uint32_t id) override {
        wrapped_->Unlock(id);
        if (id == 0 && update_) {
            std::thread reverse_worker(std::exchange(update_, {}));
            reverse_worker.join();
        }
    }
    void
    SharedLock(uint32_t id) override {
        wrapped_->SharedLock(id);
    }
    void
    SharedUnlock(uint32_t id) override {
        wrapped_->SharedUnlock(id);
    }
    void
    Resize(uint32_t count) override {
        wrapped_->Resize(count);
    }
    uint64_t
    GetMemoryUsage() override {
        return wrapped_->GetMemoryUsage();
    }

private:
    MutexArrayPtr wrapped_;
    std::function<void()> update_;
};

}  // namespace

class HGraphCacheTestAccess {
public:
    static void
    DisableUpperLevels(HGraph& graph) {
        graph.mult_ = 0.0;
    }

    static void
    RefineWithCompetingReverseEdge(HGraph& graph,
                                   const DatasetPtr& data,
                                   bool self_entry,
                                   bool& updated) {
        // 1 is the entry point. Node 2 has selected node 0, but its reverse edge 0 -> 2
        // has not been installed yet. It is the only edge that can make node 2 reachable.
        Vector<InnerIdType> neighbors(graph.allocator_);
        neighbors.push_back(1);
        graph.bottom_graph_->InsertNeighborsById(0, neighbors);
        neighbors.clear();
        graph.bottom_graph_->InsertNeighborsById(1, neighbors);
        neighbors.push_back(0);
        graph.bottom_graph_->InsertNeighborsById(2, neighbors);
        graph.entry_point_id_ = 1;
        graph.route_graphs_.clear();

        auto original_mutex = graph.neighbors_mutex_;
        graph.neighbors_mutex_ = std::make_shared<CacheRefineInterleavingMutex>(
            original_mutex, [&graph, original_mutex, &updated]() {
                // This models the reverse-edge transaction of another refinement worker.
                LockGuard lock(original_mutex, 0);
                Vector<InnerIdType> current(graph.allocator_);
                graph.bottom_graph_->GetNeighbors(0, current);
                current.push_back(2);
                graph.bottom_graph_->InsertNeighborsById(0, current);
                updated = true;
            });
        graph.searcher_->SetMutexArray(graph.neighbors_mutex_);
        graph.refine_nodes_two_phase(data,
                                     {0},
                                     "regression_refine",
                                     1,
                                     50,
                                     self_entry,
                                     graph.basic_flatten_codes_,
                                     {{0, 0}});
    }

    static void
    SetNeighbors(HGraph& graph, InnerIdType id, const std::vector<InnerIdType>& ids) {
        Vector<InnerIdType> neighbors(graph.allocator_);
        neighbors.insert(neighbors.end(), ids.begin(), ids.end());
        graph.bottom_graph_->InsertNeighborsById(id, neighbors);
    }

    static std::vector<InnerIdType>
    Neighbors(const HGraph& graph, InnerIdType id) {
        Vector<InnerIdType> neighbors(graph.allocator_);
        graph.bottom_graph_->GetNeighbors(id, neighbors);
        return {neighbors.begin(), neighbors.end()};
    }
};

}  // namespace vsag

TEST_CASE("HGraph cache refinement preserves a competing reverse edge",
          "[ut][hgraph][cache][concurrent]") {
    const bool self_entry = GENERATE(false, true);
    vsag::IndexCommonParam common;
    common.dim_ = 2;
    common.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto params = vsag::HGraph::CheckAndMappingExternalParam(vsag::JsonType::Parse(R"({
        "base_quantization_type":"fp32", "max_degree":8,
        "ef_construction":50, "build_thread_count":1
    })"),
                                                             common);
    vsag::HGraph graph(params, common);
    std::vector<int64_t> ids{10, 20, 30};
    std::vector<float> vectors{0, 0, 2, 0, 0, 1};
    auto data = MakeBuildDataset(ids, vectors, 3, 2);
    REQUIRE(graph.Build(data).empty());
    REQUIRE(graph.GetNumElements() == 3);

    bool updated = false;
    vsag::HGraphCacheTestAccess::RefineWithCompetingReverseEdge(graph, data, self_entry, updated);
    REQUIRE(updated);
    CHECK(graph.GetNumElements() == 3);
    CHECK(graph.CalcDistanceById(vectors.data() + 4, ids[2]) == 0.0F);
    const auto neighbors = vsag::HGraphCacheTestAccess::Neighbors(graph, 0);
    INFO("forward neighbors of inner ID 0: " << Catch::rangeToString(neighbors));
    CHECK(std::find(neighbors.begin(), neighbors.end(), 2) != neighbors.end());

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(2)->Float32Vectors(vectors.data() + 4)->Owner(false);
    std::stringstream serialized;
    vsag::IOStreamWriter writer(serialized);
    graph.Serialize(writer);
    vsag::HGraph restored(params, common);
    vsag::IOStreamReader reader(serialized);
    restored.Deserialize(reader);
    for (const auto* index : {&graph, &restored}) {
        const auto result = index->KnnSearch(query, 3, R"({"hgraph":{"ef_search":50}})", nullptr);
        REQUIRE(result->GetDim() > 0);
        INFO("nearest label=" << result->GetIds()[0] << " distance=" << result->GetDistances()[0]);
        CHECK(result->GetIds()[0] == ids[2]);
        CHECK(result->GetDistances()[0] == 0.0F);
        CHECK(result->GetDim() == 3);
    }
}

TEST_CASE("HGraph cache refinement joins missed and warm components",
          "[ut][hgraph][cache][concurrent]") {
    const auto threads = GENERATE(1, 4);
    // 0: empty entry point; 1: empty entry plus split warm graph; 2: removals, all remaining
    // rows hit the cache but the removed articulation points split the warm graph;
    // 3: retain the complete connected cache as the self-entry control;
    // 4: retain only the entry point; its empty warm seed is classified as a miss.
    const auto scenario = GENERATE(0, 1, 2, 3, 4);
    const uint64_t source_count = scenario == 0 ? 4 : 6;
    const uint64_t count = scenario == 2 ? 4 : source_count;
    vsag::IndexCommonParam common;
    common.dim_ = 2;
    common.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto json = vsag::JsonType::Parse(R"({
        "base_quantization_type":"fp32", "max_degree":16,
        "ef_construction":50, "build_thread_count":1
    })");
    auto source_params = vsag::HGraph::CheckAndMappingExternalParam(json, common);
    vsag::HGraph source(source_params, common);
    vsag::HGraphCacheTestAccess::DisableUpperLevels(source);
    std::vector<float> source_vectors{0, 0, 1, 0, 0, 1, 1, 1, 2, 0, 2, 1};
    std::vector<int64_t> source_ids{10, 20, 30, 40, 50, 60};
    std::vector<std::string> source_keys{"A_0", "A_1", "A_2", "A_3", "A_4", "A_5"};
    source_vectors.resize(source_count * 2);
    source_ids.resize(source_count);
    source_keys.resize(source_count);
    auto source_data = MakeBuildDataset(source_ids, source_vectors, source_count, 2);
    source_data->SourceID(source_keys.data());
    REQUIRE(source.Build(source_data).empty());
    if (scenario != 0) {
        // A connected source graph whose warm induced subgraph becomes two pairs when
        // old rows 0 and 1 are replaced or removed. All edges fit below the degree limit.
        const std::vector<std::vector<vsag::InnerIdType>> adjacency{
            {1, 2, 4}, {0}, {0, 3}, {2}, {0, 5}, {4}};
        for (uint64_t i = 0; i < source_count; ++i) {
            vsag::HGraphCacheTestAccess::SetNeighbors(source, i, adjacency[i]);
        }
    }
    REQUIRE(source
                .KnnSearch(MakeBuildDataset(source_ids, source_vectors, 1, 2),
                           source_count,
                           R"({"hgraph":{"ef_search":50}})",
                           nullptr)
                ->GetDim() == source_count);
    std::stringstream cache;
    source.ExportCache(cache);

    if (threads > 1) {
        common.thread_pool_ = vsag::SafeThreadPool::FactoryDefaultThreadPool();
        common.thread_pool_->SetPoolSize(threads);
    }
    json["build_thread_count"].SetInt(threads);
    auto params = vsag::HGraph::CheckAndMappingExternalParam(json, common);
    vsag::HGraph warmed(params, common);
    // With no upper levels, the first row is the entry point on every standard library.
    vsag::HGraphCacheTestAccess::DisableUpperLevels(warmed);
    std::vector<float> vectors = source_vectors;
    std::vector<int64_t> ids{101, 102, 103, 104, 105, 106};
    std::vector<std::string> keys{"B_0", "B_1", "A_2", "A_3", "A_4", "A_5"};
    if (scenario == 2) {
        vectors.erase(vectors.begin(), vectors.begin() + 4);
        keys.erase(keys.begin(), keys.begin() + 2);
    } else if (scenario == 3) {
        keys = source_keys;
    } else if (scenario == 4) {
        keys = {"A_0", "B_1", "B_2", "B_3", "B_4", "B_5"};
    }
    ids.resize(count);
    keys.resize(count);
    auto data = MakeBuildDataset(ids, vectors, count, 2);
    data->SourceID(keys.data());
    warmed.ImportCache(cache);
    REQUIRE(warmed.Build(data).empty());
    REQUIRE(warmed.GetNumElements() == count);
    const auto stats = vsag::JsonType::Parse(warmed.GetStats());
    const uint64_t hits = scenario == 4 ? 0 : (scenario >= 2 ? count : count - 2);
    CHECK(stats["build_cache_hit_nodes"].GetInt() == hits);
    CHECK(stats["build_cache_missed_nodes"].GetInt() == count - hits);
    CHECK(stats["connect_components"].GetInt() == 1);
    for (uint64_t i = 0; i < count; ++i) {
        const auto neighbors = vsag::HGraphCacheTestAccess::Neighbors(warmed, i);
        CHECK(std::find(neighbors.begin(), neighbors.end(), i) == neighbors.end());
    }

    std::stringstream serialized;
    vsag::IOStreamWriter writer(serialized);
    warmed.Serialize(writer);
    vsag::HGraph restored(params, common);
    vsag::IOStreamReader reader(serialized);
    restored.Deserialize(reader);
    for (const auto* index : {&warmed, &restored}) {
        for (uint64_t i = 0; i < ids.size(); ++i) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)->Dim(2)->Float32Vectors(vectors.data() + i * 2)->Owner(false);
            const auto result =
                index->KnnSearch(query, count, R"({"hgraph":{"ef_search":50}})", nullptr);
            REQUIRE(result->GetDim() > 0);
            INFO("scenario=" << scenario << " query label=" << ids[i] << " nearest label="
                             << result->GetIds()[0] << " distance=" << result->GetDistances()[0]);
            CHECK(result->GetIds()[0] == ids[i]);
            CHECK(result->GetDistances()[0] == 0.0F);
            CHECK(result->GetDim() == count);
        }
    }
}
