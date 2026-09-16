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

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_set>

#include "fixtures/functest.h"
#include "test_index.h"
#include "vsag/options.h"

namespace {

constexpr int64_t DIM = 16;
constexpr int64_t MAX_DEGREE = 8;
constexpr int64_t EF_CONSTRUCTION = 30;
constexpr int64_t EF_SEARCH = 20;

vsag::IndexPtr
CreateForceRemoveHGraphIndex(const nlohmann::json& extra = {}) {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    vsag::Options::Instance().set_block_size_limit(1024 * 1024 * 50);

    nlohmann::json index_param;
    index_param["base_quantization_type"] = "fp32";
    index_param["max_degree"] = MAX_DEGREE;
    index_param["ef_construction"] = EF_CONSTRUCTION;
    index_param["build_thread_count"] = 0;  // deterministic level assignment
    index_param["use_reverse_edges"] = true;
    index_param["support_force_remove"] = true;
    for (auto& [key, val] : extra.items()) {
        index_param[key] = val;
    }

    nlohmann::json param;
    param["dtype"] = "float32";
    param["metric_type"] = "l2";
    param["dim"] = DIM;
    param["index_param"] = index_param;

    auto index = vsag::Factory::CreateIndex("hgraph", param.dump());
    REQUIRE(index.has_value());

    vsag::Options::Instance().set_block_size_limit(origin_size);
    return index.value();
}

// Verify that after force-removing elements from the tail of the index,
// re-adding elements works correctly (no crash, valid search results).
bool
TestRemoveFromEndThenAdd(int64_t build_count, int64_t remove_count, uint64_t seed) {
    auto index = CreateForceRemoveHGraphIndex();

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distrib(0.1F, 0.9F);

    std::vector<int64_t> ids(static_cast<size_t>(build_count));
    std::vector<float> vectors(DIM * static_cast<size_t>(build_count));
    for (int64_t i = 0; i < build_count; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < DIM * build_count; ++i) {
        vectors[i] = distrib(rng);
    }

    // Build
    auto base_dataset = vsag::Dataset::Make();
    base_dataset->Dim(DIM)
        ->NumElements(build_count)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto build_result = index->Build(base_dataset);
    if (not build_result.has_value()) {
        return false;
    }

    // Remove the last `remove_count` labels — these are the highest labels,
    // which map to the highest inner_ids when using Build with sequential label
    // assignment. Removing from the tail hits the swap-tail path in
    // force_remove_one, which can leave entry_point_id_ stale when the
    // deleted element was the unique upper-level entry point.
    std::vector<int64_t> remove_ids(static_cast<size_t>(remove_count));
    for (int64_t i = 0; i < remove_count; ++i) {
        remove_ids[i] = build_count - 1 - i;
    }
    auto remove_result = index->Remove(remove_ids, vsag::RemoveMode::FORCE_REMOVE);
    if (not remove_result.has_value()) {
        return false;
    }

    // Re-add new labels beyond the original range.
    int64_t add_count = 3;
    std::vector<int64_t> new_ids(static_cast<size_t>(add_count));
    std::vector<float> new_vectors(DIM * static_cast<size_t>(add_count));
    for (int64_t i = 0; i < add_count; ++i) {
        new_ids[i] = build_count + i;
    }
    for (int64_t i = 0; i < DIM * add_count; ++i) {
        new_vectors[i] = distrib(rng);
    }

    auto add_dataset = vsag::Dataset::Make();
    add_dataset->Dim(DIM)
        ->NumElements(add_count)
        ->Ids(new_ids.data())
        ->Float32Vectors(new_vectors.data())
        ->Owner(false);
    auto add_result = index->Add(add_dataset);
    if (not add_result.has_value()) {
        return false;
    }

    // Search should work and return all added elements.
    std::string search_param = nlohmann::json{{"hgraph", {{"ef_search", EF_SEARCH}}}}.dump();
    auto query = vsag::Dataset::Make();
    query->Dim(DIM)->NumElements(1)->Float32Vectors(new_vectors.data())->Owner(false);
    auto search_result = index->KnnSearch(
        query, static_cast<int64_t>(build_count - remove_count + add_count), search_param);
    if (not search_result.has_value()) {
        return false;
    }
    auto result_count = search_result.value()->GetDim();
    if (result_count <= 0) {
        return false;
    }

    // At least one of the newly added labels should appear in the results.
    const auto* result_ids = search_result.value()->GetIds();
    for (int64_t j = 0; j < result_count; ++j) {
        for (int64_t k = 0; k < add_count; ++k) {
            if (result_ids[j] == new_ids[k]) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

// ---------------------------------------------------------------------------
// The following test cases verify the fix for a stale-entry-point defect:
//
// force_remove_one calls find_new_entry_point, which only searches
// route_graphs_. When the deleted element is the sole upper-level node and
// also the tail (swap_id == inner_id), the entry point stays pointing at
// the deleted ID.  A subsequent Add uses that stale entry point in its
// bottom-graph probe, reading out-of-range codes, which leads to an AVX512
// load on a null or invalid pointer.
//
// Reproducing deterministically requires building with build_thread_count=0
// (so level assignment is seeded) and force-removing from the tail.  The
// tests use a range of sizes and seeds to cover common configurations.
// ---------------------------------------------------------------------------

TEST_CASE("HGraph stale entry point after tail force-remove (small)",
          "[ft][hgraph][force_remove][stale_ep]") {
    fixtures::logger::LoggerReplacer _;

    constexpr int kSeeds = 5;
    for (uint64_t seed = 100; seed < 100 + kSeeds; ++seed) {
        CAPTURE(seed);
        // 12 elements: small enough that ~1-2 get upper level
        // remove 10 to leave 2, then add 3 back.
        bool ok = TestRemoveFromEndThenAdd(12, 10, seed);
        REQUIRE(ok);
    }
}

TEST_CASE("HGraph stale entry point after tail force-remove (medium)",
          "[ft][hgraph][force_remove][stale_ep]") {
    fixtures::logger::LoggerReplacer _;

    constexpr int kSeeds = 5;
    for (uint64_t seed = 200; seed < 200 + kSeeds; ++seed) {
        CAPTURE(seed);
        // 30 elements, remove 25, add back 5.
        bool ok = TestRemoveFromEndThenAdd(30, 25, seed);
        REQUIRE(ok);
    }
}

TEST_CASE("HGraph stale entry point after removing almost all (degenerate)",
          "[ft][hgraph][force_remove][stale_ep][degenerate]") {
    fixtures::logger::LoggerReplacer _;

    constexpr int kSeeds = 5;
    for (uint64_t seed = 300; seed < 300 + kSeeds; ++seed) {
        CAPTURE(seed);
        // Build 10, remove 9 (leave 1), then add 3.
        bool ok = TestRemoveFromEndThenAdd(10, 9, seed);
        REQUIRE(ok);
    }
}

TEST_CASE("HGraph stale entry point after sequential tail force-removes",
          "[ft][hgraph][force_remove][stale_ep]") {
    fixtures::logger::LoggerReplacer _;

    auto index = CreateForceRemoveHGraphIndex();

    constexpr int64_t kN = 20;
    std::vector<int64_t> ids(kN);
    std::vector<float> vectors(DIM * kN);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distrib(0.1F, 0.9F);
    for (int64_t i = 0; i < kN; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < DIM * kN; ++i) {
        vectors[i] = distrib(rng);
    }

    auto base_dataset = vsag::Dataset::Make();
    base_dataset->Dim(DIM)
        ->NumElements(kN)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    REQUIRE(index->Build(base_dataset).has_value());

    // Force-remove elements one at a time from the end.  Each removal
    // triggers find_new_entry_point, move_id (swap-tail), and shrink_to_fit.
    // After enough removals the entry point may be stale.
    for (int64_t i = kN - 1; i > 3; --i) {
        auto remove_result = index->Remove(ids[i], vsag::RemoveMode::FORCE_REMOVE);
        REQUIRE(remove_result.has_value());
        REQUIRE(remove_result.value() > 0);
    }

    REQUIRE(index->GetNumElements() >= 3);

    // Add new element — the probe uses entry_point_id_ to start searching.
    // A stale entry point causes an out-of-bounds read, which will segfault
    // or produce wrong results.
    int64_t new_id = kN;
    auto add_dataset = vsag::Dataset::Make();
    add_dataset->Dim(DIM)
        ->NumElements(1)
        ->Ids(&new_id)
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto add_result = index->Add(add_dataset);
    REQUIRE(add_result.has_value());

    std::string search_param = nlohmann::json{{"hgraph", {{"ef_search", EF_SEARCH}}}}.dump();
    auto query = vsag::Dataset::Make();
    query->Dim(DIM)->NumElements(1)->Float32Vectors(vectors.data())->Owner(false);
    auto search_result = index->KnnSearch(query, 10, search_param);
    REQUIRE(search_result.has_value());
    REQUIRE(search_result.value()->GetDim() > 0);
}

TEST_CASE("HGraph stale entry point with concurrent Add (stress)",
          "[ft][hgraph][force_remove][stale_ep][concurrent][stress]") {
    fixtures::logger::LoggerReplacer _;

    constexpr int64_t kStressN = 40;
    constexpr int kThreads = 4;
    constexpr int kAddBatch = 20;

    std::vector<float> vectors(DIM * (kStressN + kAddBatch));
    std::mt19937 rng(55);
    std::uniform_real_distribution<float> distrib(0.1F, 0.9F);
    for (int64_t i = 0; i < DIM * (kStressN + kAddBatch); ++i) {
        vectors[i] = distrib(rng);
    }

    auto index = CreateForceRemoveHGraphIndex();
    std::vector<int64_t> ids(kStressN);
    for (int64_t i = 0; i < kStressN; ++i) {
        ids[i] = i;
    }

    auto base_dataset = vsag::Dataset::Make();
    base_dataset->Dim(DIM)
        ->NumElements(kStressN)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    REQUIRE(index->Build(base_dataset).has_value());

    // Concurrent: one thread force-removes from the tail, another adds.
    std::atomic<bool> stop{false};
    std::atomic<int> add_ok{0};
    std::atomic<int> remove_ok{0};

    auto remover = [&](int64_t start_label) {
        int64_t label = start_label;
        while (not stop.load()) {
            auto result = index->Remove(label, vsag::RemoveMode::FORCE_REMOVE);
            if (result.has_value() && result.value() > 0) {
                remove_ok.fetch_add(1);
            }
            --label;
            if (label < 0) {
                break;
            }
        }
    };

    auto adder = [&](int64_t start_label) {
        int64_t label = start_label;
        while (not stop.load()) {
            auto dataset = vsag::Dataset::Make();
            dataset->Dim(DIM)
                ->NumElements(1)
                ->Ids(&label)
                ->Float32Vectors(&vectors[(label % (kStressN + kAddBatch)) * DIM])
                ->Owner(false);
            auto result = index->Add(dataset);
            if (result.has_value()) {
                add_ok.fetch_add(1);
            }
            ++label;
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    };

    std::vector<std::thread> threads;
    threads.emplace_back(remover, kStressN - 1);
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back(adder, kStressN + i * 100);
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));
    stop.store(true);

    for (auto& t : threads) {
        t.join();
    }

    // If we got here without crashing, the test passes.
    // (A segfault during Add→probe→flatten_read→SIMD_load would kill the process.)
    REQUIRE(add_ok.load() > 0);
    REQUIRE(remove_ok.load() > 0);
}

TEST_CASE("HGraph stale entry point: remove EP tail then Add single",
          "[ft][hgraph][force_remove][stale_ep][regression]") {
    fixtures::logger::LoggerReplacer _;

    auto index = CreateForceRemoveHGraphIndex({{"max_degree", 4}});

    constexpr int64_t kN = 8;
    std::vector<int64_t> ids(kN);
    std::vector<float> vectors(DIM * kN);
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> distrib(0.1F, 0.9F);
    for (int64_t i = 0; i < kN; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < DIM * kN; ++i) {
        vectors[i] = distrib(rng);
    }

    auto base_dataset = vsag::Dataset::Make();
    base_dataset->Dim(DIM)
        ->NumElements(kN)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    REQUIRE(index->Build(base_dataset).has_value());

    // After force-removing all but 1 element (ids 0 through kN-2),
    // the remaining element must be a valid entry point.
    for (int64_t i = kN - 1; i > 0; --i) {
        auto remove_result = index->Remove(ids[i], vsag::RemoveMode::FORCE_REMOVE);
        REQUIRE(remove_result.has_value());
        REQUIRE(remove_result.value() > 0);
    }

    REQUIRE(index->GetNumElements() == 1);

    // Add back one element — if entry_point_id_ is stale, this will
    // read out of bounds and crash.
    int64_t new_id = kN;
    auto add_dataset = vsag::Dataset::Make();
    add_dataset->Dim(DIM)
        ->NumElements(1)
        ->Ids(&new_id)
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto add_result = index->Add(add_dataset);
    REQUIRE(add_result.has_value());

    // Verify the newly added element is searchable.
    std::string search_param = nlohmann::json{{"hgraph", {{"ef_search", EF_SEARCH}}}}.dump();
    auto query = vsag::Dataset::Make();
    query->Dim(DIM)->NumElements(1)->Float32Vectors(vectors.data())->Owner(false);
    auto search_result = index->KnnSearch(query, 2, search_param);
    REQUIRE(search_result.has_value());
    REQUIRE(search_result.value()->GetDim() >= 1);

    bool found_new = false;
    const auto* result_ids = search_result.value()->GetIds();
    for (int64_t j = 0; j < search_result.value()->GetDim(); ++j) {
        if (result_ids[j] == new_id) {
            found_new = true;
            break;
        }
    }
    REQUIRE(found_new);
}
