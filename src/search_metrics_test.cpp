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

#include <future>
#include <limits>
#include <string>
#include <vector>

#include "query_context.h"
#include "search_metrics_internal.h"
#include "search_metrics_names.h"
#include "unittest.h"
#include "vsag/vsag.h"

TEST_CASE("Search metrics snapshots an external collector without losing extensions",
          "[ut][search_metrics]") {
    vsag::SearchStatistics statistics;
    statistics.dist_cmp.store(17);
    statistics.hops.store(9);
    statistics.query_computer_count.store(1);
    statistics.reorder_distance_count.store(23);
    statistics.AddDistance(
        vsag::DistanceEvaluationPhase::ROUTING, vsag::DistanceEvaluationBackend::RABITQ, 41);
    statistics.AddDistance(
        vsag::DistanceEvaluationPhase::RERANK, vsag::DistanceEvaluationBackend::RABITQ, 23);
    vsag::SearchMetrics metrics;
    metrics.base.dist_cmp.store(999);
    REQUIRE(metrics.AddExtension("mci_hybrid_route", std::string("hgraph")));
    REQUIRE(metrics.AddExtension("custom_flag", true));
    auto snapshot = std::move(metrics).Snapshot(statistics);
    auto expected = statistics.ToJson();
    expected["mci_hybrid_route"].SetString("hgraph");
    expected["custom_flag"].SetBool(true);
    CHECK(snapshot.Dump() == expected.Dump());
    CHECK(snapshot.dist_cmp == 17);
    CHECK(snapshot.hops == 9);
    CHECK(snapshot.reorder_distance_count == 23);
    CHECK(snapshot.DistanceEvaluations(vsag::DistanceEvaluationPhase::ROUTING) == 41);
    CHECK(snapshot.DistanceEvaluations(vsag::DistanceEvaluationPhase::RERANK) == 23);
    CHECK(statistics.dist_cmp.load() == 17);
}

TEST_CASE("HGraph single-query metrics retain lazy JSON compatibility", "[ut][search_metrics]") {
    constexpr int64_t count = 200;
    constexpr int64_t dim = 16;
    std::vector<float> base(count * dim);
    std::vector<int64_t> ids(count);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = i;
        for (int64_t d = 0; d < dim; ++d) {
            base[i * dim + d] = static_cast<float>((i * 17 + d * 13) % 101) / 101.0F;
        }
    }
    auto created = vsag::Factory::CreateIndex("hgraph", R"({"dtype":"float32",
        "metric_type":"l2","dim":16,"index_param":{"graph_type":"pipnn",
        "max_degree":16,"ef_construction":64,"base_quantization_type":"fp32"}})");
    REQUIRE(created.has_value());
    auto index = created.value();
    auto data = vsag::Dataset::Make()
                    ->NumElements(count)
                    ->Dim(dim)
                    ->Owner(false)
                    ->Ids(ids.data())
                    ->Float32Vectors(base.data());
    REQUIRE(index->Build(data).has_value());
    auto query =
        vsag::Dataset::Make()->NumElements(1)->Dim(dim)->Owner(false)->Float32Vectors(base.data());
    auto found = index->KnnSearch(query, 5, R"({"hgraph":{"ef_search":30}})");
    REQUIRE(found.has_value());
    auto snapshot = found.value()->GetSearchMetrics();
    REQUIRE(snapshot.has_value());
    const auto first = found.value()->GetStatistics();
    CHECK(first == snapshot->Dump());
    CHECK(first == found.value()->GetStatistics());
    const auto json = vsag::JsonType::Parse(first);
    CHECK(json["mci_hybrid_route"].GetString() == "disabled");
    CHECK(json["query_computer_count"].GetUint64() == 1);
    CHECK(json["distance_evaluations"].GetUint64() == snapshot->distance_evaluations);
    CHECK(snapshot->Extensions().size() == 17);
    found.value()->Statistics(R"({"replacement":true})");
    CHECK_FALSE(found.value()->GetSearchMetrics().has_value());
    CHECK(found.value()->GetStatistics() == R"({"replacement":true})");
}

TEST_CASE("Search metrics snapshot preserves counters", "[ut][search_metrics]") {
    vsag::SearchMetrics metrics;
    metrics.base.AddDistance(
        vsag::DistanceEvaluationPhase::APPROXIMATE, vsag::DistanceEvaluationBackend::FP32, 7);
    auto snapshot = std::move(metrics).Snapshot();

    CHECK(snapshot.distance_evaluations == 7);
    CHECK(snapshot.DistanceEvaluations(vsag::DistanceEvaluationPhase::APPROXIMATE) == 7);
    CHECK(snapshot.DistanceEvaluations(vsag::DistanceEvaluationBackend::FP32) == 7);
    CHECK(vsag::MetricsSchema::Base().Contains("distance_evaluations"));
    CHECK(vsag::MetricsSchema::Base().Fields().size() == 19);
    CHECK(std::string(vsag::DistanceEvaluationPhaseName(
              static_cast<vsag::DistanceEvaluationPhase>(255))) == "unknown");
    CHECK(std::string(vsag::DistanceEvaluationBackendName(
              vsag::DistanceEvaluationBackend::COUNT)) == "unknown");
}

TEST_CASE("Search metric extensions preserve baseline ownership", "[ut][search_metrics]") {
    vsag::SearchMetrics metrics;
    metrics.base.distance_evaluations.store(3, std::memory_order_relaxed);
    CHECK(metrics.AddExtension("custom_uint", uint64_t{7}));
    CHECK(metrics.AddExtension("custom_uint32_max", std::numeric_limits<uint32_t>::max()));
    CHECK(metrics.AddExtension("custom_bool", true));
    CHECK(metrics.AddExtension("custom_string", std::string("value")));
    CHECK_FALSE(metrics.AddExtension("custom_uint", uint64_t{999}));
    CHECK_FALSE(metrics.AddExtension("distance_evaluations", uint64_t{999}));

    auto snapshot = std::move(metrics).Snapshot();
    auto json = vsag::JsonType::Parse(snapshot.Dump());

    CHECK(std::get<uint64_t>(snapshot.Extensions().at("custom_uint")) == 7);
    CHECK(snapshot.Extensions().count("distance_evaluations") == 0);
    CHECK(json["custom_uint"].GetUint64() == 7);
    CHECK(json["custom_uint32_max"].GetUint64() == std::numeric_limits<uint32_t>::max());
    CHECK(json["custom_bool"].GetBool());
    CHECK(json["custom_string"].GetString() == "value");
    CHECK(json["distance_evaluations"].GetUint64() == 3);
}

TEST_CASE("Legacy search statistics keeps integer accessor compatibility", "[ut][search_metrics]") {
    vsag::SearchStatistics statistics;
    statistics.dist_cmp.store(12, std::memory_order_relaxed);
    statistics.hops.store(std::numeric_limits<uint32_t>::max(), std::memory_order_relaxed);

    auto json = statistics.ToJson();
    CHECK(json["dist_cmp"].GetInt() == 12);
    CHECK(json["hops"].GetUint64() == std::numeric_limits<uint32_t>::max());
}

TEST_CASE("Search metrics serializes uint32 fields without signed overflow",
          "[ut][search_metrics]") {
    vsag::SearchResultMetrics metrics;
    metrics.dist_cmp = std::numeric_limits<uint32_t>::max();
    metrics.hops = std::numeric_limits<uint32_t>::max();

    auto json = vsag::JsonType::Parse(metrics.Dump());
    CHECK(json["dist_cmp"].GetUint64() == std::numeric_limits<uint32_t>::max());
    CHECK(json["hops"].GetUint64() == std::numeric_limits<uint32_t>::max());
}

TEST_CASE("Dataset search statistics has one authoritative view", "[ut][search_metrics]") {
    auto dataset = vsag::Dataset::Make();
    CHECK_FALSE(dataset->GetSearchMetrics().has_value());
    CHECK(dataset->GetStatistics() == "{}");

    vsag::SearchResultMetrics metrics;
    metrics.distance_evaluations = 9;
    dataset->Statistics(R"({"legacy":1})");
    CHECK(vsag::AttachSearchMetrics(dataset, metrics));

    REQUIRE(dataset->GetSearchMetrics().has_value());
    auto selected = dataset->GetStatistics({"distance_evaluations", "complete"});
    REQUIRE(selected.size() == 2);
    CHECK(selected[0] == "9");
    CHECK(selected[1] == "true");

    dataset->Statistics(R"({"legacy":1})");
    CHECK_FALSE(dataset->GetSearchMetrics().has_value());
    CHECK(dataset->GetStatistics() == R"({"legacy":1})");
}

TEST_CASE("Dataset lazy statistics cache supports concurrent readers", "[ut][search_metrics]") {
    vsag::SearchResultMetrics metrics;
    metrics.distance_evaluations = 11;
    auto dataset = vsag::Dataset::Make();
    CHECK(vsag::AttachSearchMetrics(dataset, metrics));

    std::vector<std::future<std::string>> futures;
    for (uint64_t i = 0; i < 16; ++i) {
        futures.emplace_back(
            std::async(std::launch::async, [dataset]() { return dataset->GetStatistics(); }));
    }
    for (auto& future : futures) {
        auto json = vsag::JsonType::Parse(future.get());
        CHECK(json["distance_evaluations"].GetUint64() == 11);
    }
}
