// Copyright 2026-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <filesystem>
#include <future>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <sstream>

#include "algorithm/pyramid/pyramid_zparameters.h"
#include "vsag/vsag.h"

using Json = nlohmann::json;

TEST_CASE("Pyramid parses flat tail search parameters", "[ut][tail_knn]") {
    using vsag::PyramidSearchParameters;
    Json search = {{"pyramid", {{"ef_search", 64}}}};
    auto parsed = PyramidSearchParameters::FromJson(search.dump());
    REQUIRE_FALSE(parsed.tail_search.Enabled());
    REQUIRE(parsed.tail_search.start_after == 20);
    REQUIRE(parsed.tail_search.batch_size == 1);
    REQUIRE_FALSE(parsed.tail_search.relative_gap.has_value());
    search["pyramid"]["tail_start_after"] = 0;
    search["pyramid"]["tail_batch_size"] = 16;
    search["pyramid"]["tail_relative_gap"] = 0.1;
    parsed = PyramidSearchParameters::FromJson(search.dump());
    REQUIRE(parsed.tail_search.Enabled());
    REQUIRE(parsed.tail_search.start_after == 0);
    REQUIRE(parsed.tail_search.batch_size == 16);
    REQUIRE(*parsed.tail_search.relative_gap > 0.09F);
    search["pyramid"].erase("tail_batch_size");
    REQUIRE(PyramidSearchParameters::FromJson(search.dump()).tail_search.Enabled());
    search["pyramid"]["tail_relative_gap"] = 0;
    REQUIRE(PyramidSearchParameters::FromJson(search.dump()).tail_search.Enabled());
    for (const auto& key : {"tail_start_after", "tail_batch_size", "tail_relative_gap"}) {
        for (const Json bad : {Json(-1), Json("16"), Json(nullptr), Json(true), Json(1e100)}) {
            INFO(key << "=" << bad.dump());
            Json invalid = {{"pyramid", {{"ef_search", 64}, {key, bad}}}};
            REQUIRE_THROWS(PyramidSearchParameters::FromJson(invalid.dump()));
        }
    }
    for (const auto& key : {"tail_start_after", "tail_batch_size"}) {
        for (const Json bad : {Json(1.5), Json(4294967296ULL)}) {
            Json invalid = {{"pyramid", {{"ef_search", 64}, {key, bad}}}};
            REQUIRE_THROWS(PyramidSearchParameters::FromJson(invalid.dump()));
        }
    }
    REQUIRE_THROWS(
        PyramidSearchParameters::FromJson(R"({"pyramid":{"ef_search":64,"tail_batch_size":0}})"));
}

TEST_CASE("Pyramid KnnSearch exposes batch expansion and gap stopping", "[ut][tail_knn]") {
    constexpr uint64_t count = 512, dim = 64, query_count = 8;
    std::mt19937 rng(817);
    std::normal_distribution<float> normal;
    std::vector<float> data((count + query_count) * dim);
    for (auto& value : data) {
        value = normal(rng);
    }
    std::vector<int64_t> labels(count);
    std::vector<std::string> paths(count, "");
    std::iota(labels.begin(), labels.end(), 0);
    const auto base = vsag::Dataset::Make()
                          ->NumElements(count)
                          ->Dim(dim)
                          ->Float32Vectors(data.data())
                          ->Ids(labels.data())
                          ->Paths(paths.data())
                          ->Owner(false);
    const auto query_at = [&](uint64_t i) {
        return vsag::Dataset::Make()
            ->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(data.data() + (count + i) * dim)
            ->Owner(false);
    };
    const auto equal = [](const vsag::DatasetPtr& a, const vsag::DatasetPtr& b) {
        REQUIRE(a->GetDim() == b->GetDim());
        for (int64_t j = 0; j < a->GetDim(); ++j) {
            REQUIRE(a->GetIds()[j] == b->GetIds()[j]);
            REQUIRE(a->GetDistances()[j] == b->GetDistances()[j]);
        }
        REQUIRE(a->GetStatistics({"io_cnt",
                                  "dist_cmp",
                                  "reorder_candidate_count",
                                  "reorder_distance_count",
                                  "rabitq_filter_count"}) ==
                b->GetStatistics({"io_cnt",
                                  "dist_cmp",
                                  "reorder_candidate_count",
                                  "reorder_distance_count",
                                  "rabitq_filter_count"}));
    };
    for (const bool routing : {false, true}) {
        auto made = vsag::Factory::CreateIndex(
            "pyramid",
            Json{{"dtype", "float32"},
                 {"metric_type", "l2"},
                 {"dim", dim},
                 {"index_param",
                  {{"graph_type", "pipnn"},
                   {"max_degree", 32},
                   {"pipnn_reservoir_size", 64},
                   {"build_thread_count", 2},
                   {"root_graph_type", routing ? "multi_layer" : "single_layer"},
                   {"base_quantization_type", "rabitq"},
                   {"precise_quantization_type", "rabitq"},
                   {"rabitq_bits_per_dim_base", 3},
                   {"rabitq_bits_per_dim_precise", 5},
                   {"rabitq_bits_per_dim_query", 32},
                   {"use_reorder", true}}}}
                .dump());
        REQUIRE(made.has_value());
        const auto index = made.value();
        REQUIRE(index->Build(base).has_value());
        const auto query = query_at(0);
        const std::string enabled = R"({"pyramid":{"ef_search":64,"tail_batch_size":16}})";
        REQUIRE_FALSE(index->KnnSearch(query, 7, enabled).has_value());
        REQUIRE(index->SetImmutable().has_value());
        std::stringstream before;
        REQUIRE(index->Serialize(before).has_value());
        const std::string native = R"({"pyramid":{"ef_search":64}})";
        const std::string disabled =
            R"({"pyramid":{"ef_search":64,"tail_start_after":0,"tail_batch_size":1}})";
        for (uint64_t i = 0; i < query_count; ++i) {
            equal(index->KnnSearch(query_at(i), 7, native).value(),
                  index->KnnSearch(query_at(i), 7, disabled).value());
        }
        const std::string exhaustive = R"({"pyramid":{"ef_search":512}})";
        const std::string exhaustive_b16 = R"({"pyramid":{"ef_search":512,"tail_batch_size":16}})";
        for (uint64_t i = 0; i < query_count; ++i) {
            equal(index->KnnSearch(query_at(i), 7, exhaustive).value(),
                  index->KnnSearch(query_at(i), 7, exhaustive_b16).value());
            const auto batched = index->KnnSearch(query_at(i), 7, enabled).value();
            const auto gap =
                index
                    ->KnnSearch(
                        query_at(i),
                        7,
                        R"({"pyramid":{"ef_search":64,"tail_batch_size":16,"tail_relative_gap":0}})")
                    .value();
            REQUIRE(gap->GetDim() == 7);
            const auto gap_stats = Json::parse(gap->GetStatistics());
            const auto batch_stats = Json::parse(batched->GetStatistics());
            REQUIRE(gap_stats["hops"].get<uint64_t>() >= 20);
            REQUIRE(gap_stats["dist_cmp"].get<uint64_t>() <=
                    batch_stats["dist_cmp"].get<uint64_t>());
            REQUIRE(gap_stats["reorder_distance_count"].get<uint64_t>() ==
                    gap_stats["rabitq_full_count"].get<uint64_t>());
            REQUIRE(gap_stats["rabitq_filter_fallback_full_count"].get<uint64_t>() == 0);
        }
        // Large warmup forces single-node expansion through the new kernel.
        equal(
            index->KnnSearch(query, 7, native).value(),
            index
                ->KnnSearch(
                    query,
                    7,
                    R"({"pyramid":{"ef_search":64,"tail_start_after":4294967295,"tail_batch_size":16}})")
                .value());
        for (
            const auto& options :
            {R"({"pyramid":{"ef_search":64,"tail_start_after":0,"tail_batch_size":16,"tail_relative_gap":0}})",
             R"({"pyramid":{"ef_search":64,"tail_relative_gap":0.1}})"}) {
            const auto actual = index->KnnSearch(query, 7, options);
            REQUIRE(actual.has_value());
            REQUIRE(actual.value()->GetDim() == 7);
        }
        std::vector<std::future<vsag::DatasetPtr>> concurrent;
        for (uint64_t i = 0; i < query_count; ++i) {
            concurrent.emplace_back(std::async(std::launch::async, [&, i]() {
                return index->KnnSearch(query_at(i), 7, i % 2 ? enabled : native).value();
            }));
        }
        for (uint64_t i = 0; i < query_count; ++i) {
            equal(concurrent[i].get(),
                  index->KnnSearch(query_at(i), 7, i % 2 ? enabled : native).value());
        }
        for (const auto& extra : {Json{{"parallelism", 2}},
                                  Json{{"rabitq_one_bit_search", false}},
                                  Json{{"hierarchies", Json::array({""})}}}) {
            auto options = Json::parse(enabled);
            options["pyramid"].update(extra);
            REQUIRE_FALSE(index->KnnSearch(query, 7, options.dump()).has_value());
        }
        REQUIRE_FALSE(index->RangeSearch(query, 100.0F, enabled).has_value());
        std::function<bool(int64_t)> filter = [](int64_t) { return false; };
        REQUIRE_FALSE(index->KnnSearch(query, 7, enabled, filter).has_value());
        std::string path = "";
        query->Paths(&path);
        REQUIRE_FALSE(index->KnnSearch(query, 7, enabled).has_value());
        std::stringstream after;
        REQUIRE(index->Serialize(after).has_value());
        REQUIRE(before.str() == after.str());
    }
}
