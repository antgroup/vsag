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
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <numeric>
#include <random>
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

TEST_CASE("Alpha-CNG builds and reloads an FP32 HGraph", "[ut][hgraph][alpha_cng]") {
    constexpr int64_t dim = 8;
    constexpr int64_t count = 73;
    const std::string params = R"({"dtype":"float32","metric_type":"l2","dim":8,
        "index_param":{"graph_type":"alpha_cng","alpha_cng_tau":0.02,"max_degree":8,
        "alpha":1.06,"adaptive_pruning_adjust_step":0.04,
        "graph_iter_turn":2,"build_thread_count":2,"build_block_size":16}})";
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 100);
    ids[15] = ids[2];
    std::vector<float> rows(count * dim);
    std::mt19937 rng(73);
    for (auto& value : rows) {
        value = std::uniform_real_distribution<float>{}(rng);
    }
    auto data = MakeBuildDataset(ids, rows, count, dim);
    auto created = vsag::Factory::CreateIndex("hgraph", params);
    REQUIRE(created.has_value());
    auto built = created.value()->Build(data);
    REQUIRE(built.has_value());
    REQUIRE(built.value() == std::vector<int64_t>{ids[2]});
    auto query = MakeBuildDataset(ids, rows, 1, dim);
    auto search = [&](const auto& index) {
        auto result = index->KnnSearch(query, count - 1, R"({"hgraph":{"ef_search":100}})");
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetDim() == count - 1);
        REQUIRE(result.value()->GetIds()[0] == ids.front());
        return std::vector<int64_t>(result.value()->GetIds(),
                                    result.value()->GetIds() + result.value()->GetDim());
    };
    const auto expected = search(created.value());
    std::stringstream bytes;
    REQUIRE(created.value()->Serialize(bytes).has_value());
    auto loaded = vsag::Factory::CreateIndex("hgraph", params).value();
    REQUIRE(loaded->Deserialize(bytes).has_value());
    REQUIRE(search(loaded) == expected);
    auto first_add = vsag::Factory::CreateIndex("hgraph", params).value();
    REQUIRE(first_add->Add(MakeBuildDataset(ids, rows, 1, dim)).has_value());
    REQUIRE(first_add->GetNumElements() == 1);
    auto next = vsag::Dataset::Make()
                    ->NumElements(1)
                    ->Dim(dim)
                    ->Ids(ids.data() + 1)
                    ->Float32Vectors(rows.data() + dim)
                    ->Owner(false);
    REQUIRE(first_add->Add(next).has_value());
    REQUIRE(first_add->GetNumElements() == 2);
    for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
             {"0.02", "-1"},
             {"0.04", "-0.01"},
             {"1.06", "0.07"},
             {"\"l2\"", "\"ip\""},
             {"\"alpha_cng\"", "\"nsw\""},
             {"\"max_degree\":8", "\"base_quantization_type\":\"sq8\""}}) {
        auto invalid = params;
        invalid.replace(invalid.find(from), from.size(), to);
        REQUIRE_FALSE(vsag::Factory::CreateIndex("hgraph", invalid).has_value());
    }
}

TEST_CASE("Alpha-CNG reloads RaBitQ indexes after bulk build and after tuning FP32",
          "[ut][hgraph][alpha_cng][sq8_reorder]") {
    const bool split = GENERATE(false, true);
    const bool tune = GENERATE(false, true);
    constexpr int64_t count = 80;
    constexpr int64_t dim = 64;
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 100);
    std::mt19937 engine(735);
    std::uniform_real_distribution<float> sample(-1.0F, 1.0F);
    std::vector<float> rows(count * dim);
    std::generate(rows.begin(), rows.end(), [&]() { return sample(engine); });
    auto data = MakeBuildDataset(ids, rows, count, dim);
    const std::string start = R"({"dtype":"float32","metric_type":"l2","dim":64,"index_param":{)";
    const std::string graph = R"("graph_type":"alpha_cng","alpha_cng_tau":0.02,"max_degree":12,)";
    const std::string codes =
        std::string(
            R"("base_quantization_type":"rabitq","use_reorder":true,"rabitq_bits_per_dim_base":3,)") +
        (split ? R"("precise_quantization_type":"rabitq","rabitq_bits_per_dim_precise":5)"
               : R"("precise_quantization_type":"sq8")");
    const std::string quantized = start + graph + codes + "}}";
    const auto original = tune ? start + graph + R"("base_quantization_type":"fp32"}})" : quantized;
    auto index = vsag::Factory::CreateIndex("hgraph", original).value();
    REQUIRE(index->Build(data).has_value());
    if (tune) {
        auto result = index->Tune(quantized, true);
        REQUIRE(result.has_value());
        REQUIRE(result.value());
    }
    const auto query = MakeBuildDataset(ids, rows, 1, dim);
    const std::string search = split ? R"({"hgraph":{"ef_search":80,"rabitq_one_bit_search":true}})"
                                     : R"({"hgraph":{"ef_search":80}})";
    const auto before = index->KnnSearch(query, 10, search).value();
    REQUIRE(before->GetDim() == 10);
    REQUIRE(before->GetIds()[0] == ids.front());
    std::stringstream storage;
    REQUIRE(index->Serialize(storage).has_value());
    auto restored = vsag::Factory::CreateIndex("hgraph", quantized).value();
    REQUIRE(restored->Deserialize(storage).has_value());
    const auto after = restored->KnnSearch(query, 10, search).value();
    REQUIRE(after->GetDim() == before->GetDim());
    REQUIRE(std::equal(before->GetIds(), before->GetIds() + 10, after->GetIds()));
    REQUIRE(std::equal(before->GetDistances(), before->GetDistances() + 10, after->GetDistances()));
}

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

class CacheCandidateScheduleMutex final : public MutexArray {
    MutexArrayPtr wrapped_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool first_ready_ = false, prefix_ready_ = false, first_block_done_ = false;
    bool first_lock_ = true, start_gate_ = true, prefix_gate_ = true, trace_gate_ = true;
    std::function<void(int)> trace_;
    uint32_t block_size_;
    void
    wait(std::unique_lock<std::mutex>& lock, const std::function<bool()>& ready) {
        if (not cv_.wait_for(lock, std::chrono::seconds(10), ready)) {
            throw std::runtime_error(
                "cache candidate schedule timed out: requires two contiguous refine blocks, "
                "with node 0 paused until the second block completes its first 32 nodes");
        }
    }

public:
    CacheCandidateScheduleMutex(MutexArrayPtr wrapped,
                                uint32_t block_size,
                                std::function<void(int)> trace)
        : wrapped_(std::move(wrapped)), trace_(std::move(trace)), block_size_(block_size) {
    }
    void
    Lock(uint32_t id) override {
        {
            std::unique_lock lock(mutex_);
            if (id == 0 && first_lock_) {
                first_lock_ = false;
                first_ready_ = true;
                cv_.notify_all();
                wait(lock, [&] { return prefix_ready_; });
            }
        }
        wrapped_->Lock(id);
    }
    void
    Unlock(uint32_t id) override {
        wrapped_->Unlock(id);
        if (id == block_size_ - 1) {
            std::lock_guard lock(mutex_);
            first_block_done_ = true;
            cv_.notify_all();
        }
    }
    void
    SharedLock(uint32_t id) override {
        {
            std::unique_lock lock(mutex_);
            if (id == block_size_ && start_gate_) {
                start_gate_ = false;
                wait(lock, [&] { return first_ready_; });
            }
            if (id == block_size_ + 32 && prefix_gate_) {
                prefix_gate_ = false;
                trace_(0);
                prefix_ready_ = true;
                cv_.notify_all();
                wait(lock, [&] { return first_block_done_; });
            }
            if (id == 1 && trace_gate_) {
                trace_gate_ = false;
                trace_(1);
            }
        }
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
};

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
    ScheduleStaleCandidate(HGraph& graph, bool& full_entry, bool& orphaned) {
        // Match the failing GCC level entry while keeping the witness independent
        // of the standard library's level distribution and upper-layer RNG
        // implementation.
        // This data/entry witness needs two 128-node blocks: 127 finishes block one,
        // 128 starts block two, and 160 gates it after the prefix 128..159. Fail
        // explicitly if blocking changes instead of silently losing the race witness.
        REQUIRE(HGraph::cache_refine_block_size_ == 128);
        graph.mult_ = 0.0;
        graph.entry_point_id_ = 66;
        graph.neighbors_mutex_ = std::make_shared<CacheCandidateScheduleMutex>(
            graph.neighbors_mutex_,
            HGraph::cache_refine_block_size_,
            [&graph, &full_entry, &orphaned](int stage) {
                Vector<InnerIdType> neighbors(graph.allocator_);
                graph.bottom_graph_->GetNeighbors(graph.entry_point_id_, neighbors);
                if (stage == 0) {
                    full_entry = neighbors.size() == graph.bottom_graph_->MaximumDegree();
                } else {
                    orphaned = std::find(neighbors.begin(), neighbors.end(), 0) == neighbors.end();
                }
            });
        graph.searcher_->SetMutexArray(graph.neighbors_mutex_);
    }

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

TEST_CASE("HGraph cache build connects a miss rejected by stale reverse candidates",
          "[ut][hgraph][cache][concurrent]") {
    const auto threads = GENERATE(2, 4);
    const auto degree_limit = GENERATE(4, 16);
    constexpr int64_t count = 200;
    constexpr int64_t dim = 32;
    vsag::IndexCommonParam common;
    common.dim_ = dim;
    common.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto json = vsag::JsonType::Parse(R"({
        "base_quantization_type":"fp32", "max_degree":16,
        "ef_construction":50, "build_thread_count":1
    })");
    json["max_degree"].SetInt(degree_limit);
    auto source_params = vsag::HGraph::CheckAndMappingExternalParam(json, common);
    vsag::HGraph source(source_params, common);
    std::mt19937 rng(7);
    std::vector<float> vectors(count * dim);
    for (auto& value : vectors) {
        // Pin the original GCC uniform[-1,1] input without relying on the library's
        // uniform_real_distribution implementation on other platforms.
        value = std::ldexp(static_cast<float>(rng()), -31) - 1.0F;
    }
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 1);
    std::vector<std::string> source_ids(count);
    for (int64_t i = 0; i < count; ++i) {
        source_ids[i] = "A_" + std::to_string(i);
    }
    auto data = MakeBuildDataset(ids, vectors, count, dim);
    data->SourceID(source_ids.data());
    REQUIRE(source.Build(data).empty());
    std::stringstream cache;
    source.ExportCache(cache);
    for (int64_t i = 0; i < count; ++i) {
        source_ids[i] = "B_" + std::to_string(i);
    }
    json["build_thread_count"].SetInt(threads);
    auto target_params = vsag::HGraph::CheckAndMappingExternalParam(json, common);
    vsag::HGraph target(target_params, common);
    target.ImportCache(cache);
    bool full_entry = false;
    bool orphaned = false;
    vsag::HGraphCacheTestAccess::ScheduleStaleCandidate(target, full_entry, orphaned);
    REQUIRE(target.Build(data).empty());
    // Node 0 selected the empty entry. The other worker completed nodes128..159
    // before its forward write; the now-full entry pruned its only proposed
    // backlink. Nodes 160..199 wait until the first block finishes, making this a
    // real two-worker schedule.
    if (degree_limit == 16) {
        REQUIRE(full_entry);
        REQUIRE(orphaned);
    }
    const auto stats = vsag::JsonType::Parse(target.GetStats());
    CHECK(stats["build_cache_hit_nodes"].GetUint64() == 0);
    CHECK(stats["build_cache_missed_nodes"].GetUint64() == count);
    std::vector<uint64_t> incoming(count);
    for (uint64_t id = 0; id < count; ++id) {
        const auto neighbors = vsag::HGraphCacheTestAccess::Neighbors(target, id);
        CHECK(neighbors.size() <= static_cast<uint64_t>(degree_limit));
        for (auto neighbor : neighbors) {
            REQUIRE(neighbor < count);
            CHECK(neighbor != id);
            ++incoming[neighbor];
        }
    }
    for (auto degree : incoming) {
        CHECK(degree > 0);
    }
    // A saturated degree-four graph also exercises repairs that must preserve the
    // entry's existing spanning tree. Every miss must be reachable after the build.
    std::vector<bool> reachable(count);
    std::vector<vsag::InnerIdType> frontier{66};
    reachable[66] = true;
    for (uint64_t cursor = 0; cursor < frontier.size(); ++cursor) {
        for (auto neighbor : vsag::HGraphCacheTestAccess::Neighbors(target, frontier[cursor])) {
            if (not reachable[neighbor]) {
                reachable[neighbor] = true;
                frontier.push_back(neighbor);
            }
        }
    }
    CHECK(frontier.size() == count);
    std::stringstream serialized;
    vsag::IOStreamWriter writer(serialized);
    target.Serialize(writer);
    vsag::HGraph restored(target_params, common);
    vsag::IOStreamReader reader(serialized);
    restored.Deserialize(reader);
    for (const auto* index : {&target, &restored}) {
        // Preserve the original ef50 self-query assertion. The degree16 witness also
        // checks the complete query set; degree4 checks the same first query plus the
        // full structural reachability invariant above, without assuming exact ANN recall.
        const auto query_count = degree_limit == 16 ? count : 1;
        for (int64_t i = 0; i < query_count; ++i) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data() + i * dim)->Owner(false);
            CHECK(index->CalcDistanceById(vectors.data() + i * dim, ids[i]) == 0.0F);
            auto result = index->KnnSearch(query, 10, R"({"hgraph":{"ef_search":50}})", nullptr);
            bool found_self = false;
            for (int64_t k = 0; k < result->GetDim(); ++k) {
                if (result->GetIds()[k] == ids[i]) {
                    CHECK(result->GetDistances()[k] < 1e-4F);
                    found_self = true;
                }
            }
            INFO("query label=" << ids[i]);
            CHECK(found_self);
        }
    }
}
