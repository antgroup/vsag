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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <string>
#include <vector>

#include "argparse/argparse.hpp"
#include "case/search_eval_case.h"
#include "eval_config.h"
#include "eval_dataset.h"
#include "eval_job.h"
#include "monitor/recall_monitor.h"

namespace vsag::eval {
namespace {

EvalConfig
LoadCliConfig(const std::vector<std::string>& extra_arguments) {
    std::vector<std::string> arguments{"eval_performance",
                                       "--datapath",
                                       "/tmp/eval.hdf5",
                                       "--type",
                                       "search",
                                       "--index_name",
                                       "hgraph",
                                       "--create_params",
                                       "{}"};
    arguments.insert(arguments.end(), extra_arguments.begin(), extra_arguments.end());

    argparse::ArgumentParser parser("eval_performance");
    EvalConfig::AddCliArguments(parser);
    parser.parse_args(arguments);
    return EvalConfig::Load(parser);
}

EvalConfig
LoadYamlConfig(const std::string& extra_yaml = "") {
    auto yaml = YAML::Load(
        "datapath: /tmp/eval.hdf5\n"
        "type: search\n"
        "index_name: hgraph\n"
        "create_params: '{}'\n"
        "search_params: '{}'\n" +
        extra_yaml);
    EvalConfig::CheckKeyAndType(yaml);
    EvalJob global_options;
    return EvalConfig::Load(yaml, global_options);
}

void
ConstructSearchCase(const EvalConfig& config) {
    std::vector<float> vectors{0.0F};
    std::vector<int64_t> ids{0};
    auto queries =
        vsag::Dataset::Make()->NumElements(1)->Dim(1)->Float32Vectors(vectors.data())->Owner(false);
    auto truth = vsag::Dataset::Make()->NumElements(1)->Dim(1)->Ids(ids.data())->Owner(false);
    auto dataset = EvalDataset::FromSearchDatasets(queries, truth);
    SearchEvalCase eval_case("", "", nullptr, config, dataset);
}

void
RequireCoverage(const Monitor::JsonType& result,
                double target,
                uint64_t reached_queries,
                uint64_t query_count,
                double rate) {
    const auto& coverage = result.at("query_coverage");
    REQUIRE(coverage.at("recall_target").get<double>() == Catch::Approx(target));
    REQUIRE(coverage.at("reached_queries").get<uint64_t>() == reached_queries);
    REQUIRE(coverage.at("query_count").get<uint64_t>() == query_count);
    REQUIRE(coverage.at("rate").get<double>() == Catch::Approx(rate));
}

}  // namespace

TEST_CASE("Search evaluation parses recall targets and the query count",
          "[ut][eval][query_coverage]") {
    const auto cli = LoadCliConfig({"--recall_target", "0.9", "--search-query-count", "1"});
    REQUIRE(cli.recall_target.value() == Catch::Approx(0.9));
    REQUIRE(cli.search_query_count == 1);

    REQUIRE_FALSE(LoadYamlConfig().recall_target.has_value());
    REQUIRE(LoadYamlConfig("recall_target: 0.9\n").recall_target.value() == Catch::Approx(0.9));
}

TEST_CASE("Search evaluation rejects invalid recall targets", "[ut][eval][query_coverage]") {
    REQUIRE_NOTHROW(ConstructSearchCase(LoadYamlConfig("recall_target: 0\n")));
    REQUIRE_NOTHROW(ConstructSearchCase(LoadYamlConfig("recall_target: 1\n")));
    REQUIRE_NOTHROW(
        ConstructSearchCase(LoadYamlConfig("search_mode: knn_filter\nrecall_target: 0.9\n")));

    for (const auto* value : {"1.01", "-0.01", ".nan"}) {
        CAPTURE(value);
        REQUIRE_THROWS_WITH(
            ConstructSearchCase(LoadYamlConfig(std::string("recall_target: ") + value + "\n")),
            "recall_target must be finite and in [0, 1]");
    }
    for (const auto* mode : {"range", "range_filter"}) {
        CAPTURE(mode);
        REQUIRE_THROWS_WITH(ConstructSearchCase(LoadYamlConfig(std::string("search_mode: ") + mode +
                                                               "\nrecall_target: 0.9\n")),
                            "recall_target is supported only for knn and knn_filter search modes");
    }
}

TEST_CASE("Recall coverage reports zero queries before recording", "[ut][eval][query_coverage]") {
    RecallMonitor monitor(0, true, 0.9);
    RequireCoverage(monitor.GetResult(), 0.9, 0, 0, 0.0);
}

TEST_CASE("Recall coverage uses the discrete per-query boundary", "[ut][eval][query_coverage]") {
    std::vector<float> query_vector{0.0F};
    auto queries = vsag::Dataset::Make()
                       ->NumElements(1)
                       ->Dim(1)
                       ->Float32Vectors(query_vector.data())
                       ->Owner(false);
    std::vector<int64_t> ground_truth_ids{10, 11, 12, 13, 14};
    auto ground_truth =
        vsag::Dataset::Make()->NumElements(1)->Dim(5)->Ids(ground_truth_ids.data())->Owner(false);
    auto dataset = EvalDataset::FromSearchDatasets(queries, ground_truth);

    int64_t perfect_result[]{10, 11, 12, 13, 14};
    int64_t four_match_result[]{10, 11, 12, 13, 99};
    SearchRecord perfect_record{
        perfect_result, dataset->GetNeighbors(0), dataset.get(), dataset->GetOneTest(0), 5, 5};
    SearchRecord four_match_record{
        four_match_result, dataset->GetNeighbors(0), dataset.get(), dataset->GetOneTest(0), 5, 5};

    RecallMonitor monitor(2, true, 0.9);
    monitor.Record(&perfect_record);
    monitor.Record(&four_match_record);
    RequireCoverage(monitor.GetResult(), 0.9, 1, 2, 0.5);

    // 4 of 5 reaches 0.8 exactly.
    RecallMonitor exact_boundary_monitor(1, true, 0.8);
    exact_boundary_monitor.Record(&four_match_record);
    RequireCoverage(exact_boundary_monitor.GetResult(), 0.8, 1, 1, 1.0);
}

TEST_CASE("Recall coverage preserves distance-based and ID-based semantics",
          "[ut][eval][query_coverage]") {
    constexpr int64_t dim = 2;
    std::vector<float> base_vectors{0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F};
    std::vector<int64_t> base_ids{10, 20, 30};
    auto base = vsag::Dataset::Make()
                    ->NumElements(3)
                    ->Dim(dim)
                    ->Ids(base_ids.data())
                    ->Float32Vectors(base_vectors.data())
                    ->Owner(false);
    std::vector<float> query_vectors{0.0F, 0.0F};
    auto queries = vsag::Dataset::Make()
                       ->NumElements(1)
                       ->Dim(dim)
                       ->Float32Vectors(query_vectors.data())
                       ->Owner(false);
    std::vector<int64_t> ground_truth_ids{10, 20};
    auto ground_truth =
        vsag::Dataset::Make()->NumElements(1)->Dim(2)->Ids(ground_truth_ids.data())->Owner(false);
    auto dataset = EvalDataset::FromDatasets(base, queries, ground_truth, "l2");

    // 30 is not a true neighbor by ID but ties 20 by distance: recall 1.0 by distance, 0.5 by ID.
    int64_t result_ids[]{10, 30};
    SearchRecord record{
        result_ids, dataset->GetNeighbors(0), dataset.get(), dataset->GetOneTest(0), 2, 2};

    RecallMonitor distance_monitor(1, false, 0.75);
    distance_monitor.Record(&record);
    RequireCoverage(distance_monitor.GetResult(), 0.75, 1, 1, 1.0);

    RecallMonitor id_monitor(1, true, 0.75);
    id_monitor.Record(&record);
    RequireCoverage(id_monitor.GetResult(), 0.75, 0, 1, 0.0);

    RecallMonitor unconfigured_monitor(1, true);
    unconfigured_monitor.SetMetrics("avg_recall");
    unconfigured_monitor.Record(&record);
    REQUIRE_FALSE(unconfigured_monitor.GetResult().contains("query_coverage"));
}

}  // namespace vsag::eval
