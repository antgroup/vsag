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

#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

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

TEST_CASE("HGraph connects finite additions after a non-finite entry point",
          "[ut][hgraph][build][nonfinite]") {
    const auto metric = GENERATE("l2", "cosine");
    const auto quantizer = GENERATE("fp32", "fp16");
    INFO("metric=" << metric << ", quantizer=" << quantizer);
    const auto params = fmt::format(R"({{
        "dtype":"float32", "metric_type":"{}", "dim":64,
        "index_param":{{"base_quantization_type":"{}","max_degree":96,
        "ef_construction":100,"build_thread_count":1}}
    }})",
                                    metric,
                                    quantizer);
    auto index = vsag::Factory::CreateIndex("hgraph", params).value();
    std::vector<int64_t> ids{0};
    std::vector<float> dirty(64, 1.0F);
    dirty[0] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(index->Build(MakeBuildDataset(ids, dirty, 1, 64)).has_value());
    ids[0] = 1;
    std::vector<float> finite(64, 1.0F);
    REQUIRE(index->Add(MakeBuildDataset(ids, finite, 1, 64)).has_value());
    auto result = index->KnnSearch(
        MakeBuildDataset(ids, finite, 1, 64), 1, R"({"hgraph":{"ef_search":100}})");
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == 1);
    REQUIRE(result.value()->GetIds()[0] == 1);
}

TEST_CASE("HGraph mixed NaN batch preserves finite vector connectivity",
          "[ut][hgraph][build][nonfinite]") {
    const auto threads = GENERATE(1, 5);
    INFO("threads=" << threads);
    const auto params = fmt::format(R"({{
        "dtype":"float32", "metric_type":"cosine", "dim":64,
        "index_param":{{"base_quantization_type":"fp16","max_degree":128,
        "ef_construction":128,"build_thread_count":{}}}
    }})",
                                    threads);
    constexpr int64_t count = 128;
    constexpr int64_t dim = 64;
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> values(0.1F, 0.9F);
    std::vector<float> vectors(count * dim);
    for (auto& value : vectors) {
        value = values(rng);
    }
    for (const auto id : {7, 31, 63}) {
        vectors[id * dim] = std::numeric_limits<float>::quiet_NaN();
    }
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    auto index = vsag::Factory::CreateIndex("hgraph", params).value();
    REQUIRE(index->Build(MakeBuildDataset(ids, vectors, count, dim)).has_value());
    REQUIRE(index->GetNumElements() == count);
    auto result = index->KnnSearch(
        MakeBuildDataset(ids, vectors, 1, dim), count - 3, R"({"hgraph":{"ef_search":128}})");
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == count - 3);
    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
        const auto id = result.value()->GetIds()[i];
        REQUIRE(id != 7);
        REQUIRE(id != 31);
        REQUIRE(id != 63);
    }
}

TEST_CASE("HGraph unbounded KNN preserves infinite FP16 distances like FP32",
          "[ut][hgraph][build][nonfinite]") {
    const auto metric = GENERATE("ip", "l2");
    const auto quantizer = GENERATE("fp32", "fp16");
    INFO("metric=" << metric << ", quantizer=" << quantizer);
    const auto params = fmt::format(R"({{
        "dtype":"float32", "metric_type":"{}", "dim":1,
        "index_param":{{"base_quantization_type":"{}","max_degree":8,
        "ef_construction":32,"build_thread_count":1}}
    }})",
                                    metric,
                                    quantizer);
    auto index = vsag::Factory::CreateIndex("hgraph", params).value();
    const auto infinity = std::numeric_limits<float>::infinity();
    std::vector<int64_t> ids{0, 1, 2};
    std::vector<float> vectors{infinity, 1.0F, -infinity};
    REQUIRE(index->Build(MakeBuildDataset(ids, vectors, 3, 1)).has_value());
    std::vector<float> query{1.0F};
    auto result =
        index->KnnSearch(MakeBuildDataset(ids, query, 1, 1), 3, R"({"hgraph":{"ef_search":32}})");
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == 3);
    const auto* result_ids = result.value()->GetIds();
    const auto* distances = result.value()->GetDistances();
    if (std::string(metric) == "ip") {
        REQUIRE(result_ids[0] == 0);
        REQUIRE(std::isinf(distances[0]));
        REQUIRE(std::signbit(distances[0]));
        REQUIRE(result_ids[1] == 1);
        REQUIRE(distances[1] == 0.0F);
        REQUIRE(result_ids[2] == 2);
        REQUIRE(std::isinf(distances[2]));
        REQUIRE_FALSE(std::signbit(distances[2]));
    } else {
        REQUIRE(result_ids[0] == 1);
        REQUIRE(distances[0] == 0.0F);
        REQUIRE(result_ids[1] != result_ids[2]);
        for (int64_t i = 1; i < 3; ++i) {
            REQUIRE((result_ids[i] == 0 or result_ids[i] == 2));
            REQUIRE(std::isinf(distances[i]));
            REQUIRE_FALSE(std::signbit(distances[i]));
        }
    }
}

TEST_CASE("HGraph recovers finite connectivity after an all-NaN batch",
          "[ut][hgraph][build][nonfinite]") {
    const auto threads = GENERATE(1, 5);
    INFO("threads=" << threads);
    const auto params = fmt::format(R"({{
        "dtype":"float32", "metric_type":"cosine", "dim":64,
        "index_param":{{"base_quantization_type":"fp16","max_degree":96,
        "ef_construction":256,"build_thread_count":{}}}
    }})",
                                    threads);
    constexpr int64_t dirty_count = 128;
    constexpr int64_t finite_count = 64;
    constexpr int64_t dim = 64;
    // Keep reverse lists below capacity: recovering every finite node must test the dirty-entry
    // bridge independently of approximate edge pruning, which can also lose a node in clean data.
    std::vector<float> dirty(dirty_count * dim, std::numeric_limits<float>::quiet_NaN());
    std::vector<int64_t> dirty_ids(dirty_count);
    std::iota(dirty_ids.begin(), dirty_ids.end(), 0);
    auto index = vsag::Factory::CreateIndex("hgraph", params).value();
    REQUIRE(index->Build(MakeBuildDataset(dirty_ids, dirty, dirty_count, dim)).has_value());
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> values(0.1F, 0.9F);
    std::vector<float> finite(finite_count * dim);
    for (auto& value : finite) {
        value = values(rng);
    }
    std::vector<int64_t> ids(finite_count);
    std::iota(ids.begin(), ids.end(), dirty_count);
    REQUIRE(index->Add(MakeBuildDataset(ids, finite, finite_count, dim)).has_value());
    auto result = index->KnnSearch(
        MakeBuildDataset(ids, finite, 1, dim), finite_count, R"({"hgraph":{"ef_search":256}})");
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == finite_count);
    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
        REQUIRE(result.value()->GetIds()[i] >= dirty_count);
    }
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
