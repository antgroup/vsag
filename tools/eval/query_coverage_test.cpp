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
#include <catch2/matchers/catch_matchers_string.hpp>
#include <string>
#include <vector>

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
                                       "{}",
                                       "--search_params",
                                       "{}",
                                       "--index_path",
                                       "/tmp/eval.index",
                                       "--search_mode",
                                       "knn",
                                       "--search-query-count",
                                       "1"};
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

TEST_CASE("EvalConfig parses optional recall targets from CLI and YAML",
          "[ut][eval][query_coverage]") {
    REQUIRE_FALSE(LoadCliConfig({}).recall_target.has_value());
    REQUIRE(LoadCliConfig({"--recall_target", "0.9"}).recall_target.value() == Catch::Approx(0.9));
    REQUIRE_THROWS_WITH(LoadCliConfig({"--recall_target", "1.01"}),
                        "recall_target must be finite and in [0, 1]");

    REQUIRE_FALSE(LoadYamlConfig().recall_target.has_value());
    REQUIRE(LoadYamlConfig("recall_target: 0\n").recall_target.value() == Catch::Approx(0.0));
    REQUIRE(LoadYamlConfig("recall_target: 1\n").recall_target.value() == Catch::Approx(1.0));
    REQUIRE_THROWS_WITH(LoadYamlConfig("recall_target: -0.01\n"),
                        "recall_target must be finite and in [0, 1]");
    REQUIRE_THROWS_WITH(LoadYamlConfig("recall_target: .nan\n"),
                        "recall_target must be finite and in [0, 1]");
}

TEST_CASE("EvalConfig rejects recall targets for modes without recall at k",
          "[ut][eval][query_coverage]") {
    const auto knn_filter = LoadYamlConfig("search_mode: knn_filter\nrecall_target: 0.9\n");
    REQUIRE(knn_filter.search_mode == "knn_filter");
    REQUIRE(knn_filter.recall_target.value() == Catch::Approx(0.9));

    REQUIRE_THROWS_WITH(LoadYamlConfig("search_mode: range\nrecall_target: 0.9\n"),
                        "recall_target is supported only for knn and knn_filter search modes");
    REQUIRE_THROWS_WITH(LoadYamlConfig("search_mode: range_filter\nrecall_target: 0.9\n"),
                        "recall_target is supported only for knn and knn_filter search modes");

    REQUIRE(LoadYamlConfig("search_mode: range\n").search_mode == "range");
}

TEST_CASE("Recall coverage uses the discrete per-query boundary", "[ut][eval][query_coverage]") {
    constexpr int64_t dim = 2;
    std::vector<float> query_vectors{0.0F, 0.0F, 1.0F, 1.0F};
    auto queries = vsag::Dataset::Make()
                       ->NumElements(2)
                       ->Dim(dim)
                       ->Float32Vectors(query_vectors.data())
                       ->Owner(false);
    std::vector<int64_t> ground_truth_ids{10, 11, 12, 13, 14, 20, 21, 22, 23, 24};
    auto ground_truth =
        vsag::Dataset::Make()->NumElements(2)->Dim(5)->Ids(ground_truth_ids.data())->Owner(false);
    auto dataset = EvalDataset::FromSearchDatasets(queries, ground_truth);

    int64_t perfect_result[]{10, 11, 12, 13, 14};
    int64_t four_match_result[]{20, 21, 22, 23, 99};
    SearchRecord perfect_record{
        perfect_result, dataset->GetNeighbors(0), dataset.get(), dataset->GetOneTest(0), 5, 5};
    SearchRecord four_match_record{
        four_match_result, dataset->GetNeighbors(1), dataset.get(), dataset->GetOneTest(1), 5, 5};

    RecallMonitor monitor(2, true, 0.9);
    monitor.SetMetrics("avg_recall");
    monitor.Record(&perfect_record);
    monitor.Record(&four_match_record);
    const auto result = monitor.GetResult();

    REQUIRE(result.at("recall_avg").get<double>() == Catch::Approx(0.9));
    RequireCoverage(result, 0.9, 1, 2, 0.5);

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

    int64_t result_ids[]{10, 30};
    SearchRecord record{
        result_ids, dataset->GetNeighbors(0), dataset.get(), dataset->GetOneTest(0), 2, 2};

    RecallMonitor distance_monitor(1, false, 0.75);
    distance_monitor.SetMetrics("avg_recall");
    distance_monitor.Record(&record);
    const auto distance_result = distance_monitor.GetResult();
    REQUIRE(distance_result.at("recall_avg").get<double>() == Catch::Approx(1.0));
    RequireCoverage(distance_result, 0.75, 1, 1, 1.0);

    RecallMonitor id_monitor(1, true, 0.75);
    id_monitor.SetMetrics("avg_recall");
    id_monitor.Record(&record);
    const auto id_result = id_monitor.GetResult();
    REQUIRE(id_result.at("recall_avg").get<double>() == Catch::Approx(0.5));
    RequireCoverage(id_result, 0.75, 0, 1, 0.0);

    RecallMonitor unconfigured_monitor(1, true);
    unconfigured_monitor.SetMetrics("avg_recall");
    unconfigured_monitor.Record(&record);
    REQUIRE_FALSE(unconfigured_monitor.GetResult().contains("query_coverage"));
}

}  // namespace vsag::eval
