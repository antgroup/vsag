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
#include <mutex>
#include <nlohmann/json.hpp>
#include <thread>
#include <unordered_set>

#include "fixtures/functest.h"
#include "test_index.h"
#include "vsag/options.h"
#include "vsag/vsag.h"

namespace {

constexpr int64_t DIM = 16;
constexpr int64_t NUM_ELEMENTS = 50;
constexpr int64_t MAX_DEGREE = 8;
constexpr int64_t EF_CONSTRUCTION = 30;
constexpr int64_t EF_SEARCH = 20;
constexpr int64_t THREAD_COUNT = 4;

vsag::IndexPtr
CreateHGraphIndex() {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    vsag::Options::Instance().set_block_size_limit(1024 * 1024 * 2);

    nlohmann::json index_param;
    index_param["base_quantization_type"] = "fp32";
    index_param["max_degree"] = MAX_DEGREE;
    index_param["ef_construction"] = EF_CONSTRUCTION;
    index_param["build_thread_count"] = 0;
    index_param["use_reverse_edges"] = true;

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

}  // namespace

TEST_CASE("HGraph Sequential Add and ForceRemove", "[ft][hgraph]") {
    fixtures::logger::LoggerReplacer _;

    auto index = CreateHGraphIndex();

    std::vector<int64_t> ids(NUM_ELEMENTS);
    std::vector<float> vectors(DIM * NUM_ELEMENTS);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib(0.1, 0.9);
    for (int64_t i = 0; i < NUM_ELEMENTS; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < DIM * NUM_ELEMENTS; ++i) {
        vectors[i] = distrib(rng);
    }

    std::string search_param = nlohmann::json{{"hgraph", {{"ef_search", EF_SEARCH}}}}.dump();

    for (int64_t i = 0; i < NUM_ELEMENTS; ++i) {
        auto dataset = vsag::Dataset::Make();
        dataset->Dim(DIM)
            ->NumElements(1)
            ->Ids(&ids[i])
            ->Float32Vectors(&vectors[i * DIM])
            ->Owner(false);
        auto result = index->Add(dataset);
        REQUIRE(result.has_value());
    }

    REQUIRE(index->GetNumElements() == NUM_ELEMENTS);

    for (int64_t i = 0; i < NUM_ELEMENTS / 2; ++i) {
        auto result = index->Remove(ids[i], vsag::RemoveMode::ForceRemove);
        REQUIRE(result.has_value());
    }

    REQUIRE(index->GetNumElements() == NUM_ELEMENTS / 2);

    for (int64_t i = NUM_ELEMENTS / 2; i < NUM_ELEMENTS; ++i) {
        auto query = vsag::Dataset::Make();
        query->Dim(DIM)->NumElements(1)->Float32Vectors(&vectors[i * DIM])->Owner(false);
        auto result = index->KnnSearch(query, 10, search_param);
        REQUIRE(result.has_value());
    }
}

TEST_CASE("HGraph Concurrent Add and ForceRemove", "[ft][hgraph][concurrent]") {
    fixtures::logger::LoggerReplacer _;

    auto index = CreateHGraphIndex();

    std::vector<int64_t> ids(NUM_ELEMENTS);
    std::vector<float> vectors(DIM * NUM_ELEMENTS);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib(0.1, 0.9);
    for (int64_t i = 0; i < NUM_ELEMENTS; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < DIM * NUM_ELEMENTS; ++i) {
        vectors[i] = distrib(rng);
    }

    fixtures::ThreadPool pool(THREAD_COUNT);
    std::vector<std::future<bool>> futures;

    std::atomic<int64_t> add_success{0};
    std::atomic<int64_t> remove_success{0};

    auto add_func = [&](int64_t i) -> bool {
        auto dataset = vsag::Dataset::Make();
        dataset->Dim(DIM)
            ->NumElements(1)
            ->Ids(&ids[i])
            ->Float32Vectors(&vectors[i * DIM])
            ->Owner(false);

        auto add_result = index->Add(dataset);
        if (add_result.has_value()) {
            add_success++;
            return true;
        }
        return false;
    };

    auto remove_func = [&](int64_t i) -> bool {
        auto remove_result = index->Remove(ids[i], vsag::RemoveMode::ForceRemove);
        if (remove_result.has_value()) {
            remove_success++;
            return true;
        }
        return false;
    };

    for (int64_t i = 0; i < NUM_ELEMENTS; ++i) {
        futures.emplace_back(pool.enqueue(add_func, i));
    }

    for (auto& future : futures) {
        REQUIRE(future.get());
    }

    REQUIRE(index->GetNumElements() == NUM_ELEMENTS);
    futures.clear();

    for (int64_t i = 0; i < NUM_ELEMENTS / 2; ++i) {
        futures.emplace_back(pool.enqueue(remove_func, i));
    }

    for (auto& future : futures) {
        REQUIRE(future.get());
    }

    REQUIRE(index->GetNumElements() == NUM_ELEMENTS / 2);
    REQUIRE(add_success == NUM_ELEMENTS);
    REQUIRE(remove_success == NUM_ELEMENTS / 2);
}

TEST_CASE("HGraph Sequential Add Remove ReAdd", "[ft][hgraph]") {
    fixtures::logger::LoggerReplacer _;

    auto index = CreateHGraphIndex();

    std::vector<int64_t> ids(NUM_ELEMENTS);
    std::vector<float> vectors(DIM * NUM_ELEMENTS);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib(0.1, 0.9);
    for (int64_t i = 0; i < NUM_ELEMENTS; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < DIM * NUM_ELEMENTS; ++i) {
        vectors[i] = distrib(rng);
    }

    std::string search_param = nlohmann::json{{"hgraph", {{"ef_search", EF_SEARCH}}}}.dump();

    for (int64_t i = 0; i < NUM_ELEMENTS; ++i) {
        auto dataset = vsag::Dataset::Make();
        dataset->Dim(DIM)
            ->NumElements(1)
            ->Ids(&ids[i])
            ->Float32Vectors(&vectors[i * DIM])
            ->Owner(false);
        auto result = index->Add(dataset);
        REQUIRE(result.has_value());
    }

    REQUIRE(index->GetNumElements() == NUM_ELEMENTS);

    for (int64_t i = 0; i < NUM_ELEMENTS / 2; ++i) {
        auto result = index->Remove(ids[i], vsag::RemoveMode::ForceRemove);
        REQUIRE(result.has_value());
    }

    REQUIRE(index->GetNumElements() == NUM_ELEMENTS / 2);

    for (int64_t i = 0; i < NUM_ELEMENTS / 2; ++i) {
        auto dataset = vsag::Dataset::Make();
        dataset->Dim(DIM)
            ->NumElements(1)
            ->Ids(&ids[i])
            ->Float32Vectors(&vectors[i * DIM])
            ->Owner(false);
        auto result = index->Add(dataset);
        REQUIRE(result.has_value());
    }

    REQUIRE(index->GetNumElements() == NUM_ELEMENTS);

    for (int64_t i = 0; i < NUM_ELEMENTS; ++i) {
        auto query = vsag::Dataset::Make();
        query->Dim(DIM)->NumElements(1)->Float32Vectors(&vectors[i * DIM])->Owner(false);
        auto result = index->KnnSearch(query, 10, search_param);
        REQUIRE(result.has_value());
    }
}

TEST_CASE("HGraph ForceRemove All Elements", "[ft][hgraph]") {
    fixtures::logger::LoggerReplacer _;

    auto index = CreateHGraphIndex();

    std::vector<int64_t> ids(NUM_ELEMENTS);
    std::vector<float> vectors(DIM * NUM_ELEMENTS);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib(0.1, 0.9);
    for (int64_t i = 0; i < NUM_ELEMENTS; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < DIM * NUM_ELEMENTS; ++i) {
        vectors[i] = distrib(rng);
    }

    std::string search_param = nlohmann::json{{"hgraph", {{"ef_search", EF_SEARCH}}}}.dump();

    auto base_dataset = vsag::Dataset::Make();
    base_dataset->Dim(DIM)
        ->NumElements(NUM_ELEMENTS)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto build_result = index->Build(base_dataset);
    REQUIRE(build_result.has_value());
    REQUIRE(index->GetNumElements() == NUM_ELEMENTS);

    for (int64_t i = 0; i < NUM_ELEMENTS; ++i) {
        auto query = vsag::Dataset::Make();
        query->Dim(DIM)->NumElements(1)->Float32Vectors(&vectors[i * DIM])->Owner(false);
        auto result = index->KnnSearch(query, 10, search_param);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetDim() > 0);
    }

    for (int64_t i = 0; i < NUM_ELEMENTS; ++i) {
        auto remove_result = index->Remove(ids[i], vsag::RemoveMode::ForceRemove);
        REQUIRE(remove_result.has_value());
        REQUIRE(remove_result.value() > 0);
    }

    REQUIRE(index->GetNumElements() == 0);

    auto empty_query = vsag::Dataset::Make();
    empty_query->Dim(DIM)->NumElements(1)->Float32Vectors(vectors.data())->Owner(false);
    auto empty_result = index->KnnSearch(empty_query, 10, search_param);
    REQUIRE(empty_result.has_value());
    REQUIRE(empty_result.value()->GetDim() == 0);
}

TEST_CASE("HGraph Batch ForceRemove supports legacy remove alias", "[ft][hgraph]") {
    fixtures::logger::LoggerReplacer _;

    auto index = CreateHGraphIndex();

    std::vector<int64_t> ids(NUM_ELEMENTS);
    std::vector<float> vectors(DIM * NUM_ELEMENTS);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib(0.1, 0.9);
    for (int64_t i = 0; i < NUM_ELEMENTS; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < DIM * NUM_ELEMENTS; ++i) {
        vectors[i] = distrib(rng);
    }

    auto base_dataset = vsag::Dataset::Make();
    base_dataset->Dim(DIM)
        ->NumElements(NUM_ELEMENTS)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto build_result = index->Build(base_dataset);
    REQUIRE(build_result.has_value());

    std::vector<int64_t> remove_ids(ids.begin(), ids.begin() + 5);
    auto remove_result = index->Remove(remove_ids, vsag::RemoveMode::REMOVE_AND_REPAIR);
    REQUIRE(remove_result.has_value());
    REQUIRE(remove_result.value() == remove_ids.size());
    REQUIRE(index->GetNumElements() == NUM_ELEMENTS - remove_ids.size());
}
