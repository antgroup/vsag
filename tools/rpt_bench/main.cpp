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

#include <argparse/argparse.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "eval_dataset.h"
#include "metrics.h"
#include "partitioners.h"
#include "vsag/vsag.h"

namespace {

using vsag::rpt_bench::LocalityMetrics;
using vsag::rpt_bench::PartitionRequest;
using vsag::rpt_bench::UniformityMetrics;

std::vector<std::string>
SplitList(const std::string& text) {
    std::vector<std::string> items;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (not item.empty()) {
            items.push_back(item);
        }
    }
    return items;
}

std::vector<uint64_t>
ParseBucketSizes(const std::string& text) {
    std::vector<uint64_t> sizes;
    for (const auto& item : SplitList(text)) {
        sizes.push_back(std::stoull(item));
    }
    return sizes;
}

void
ParseArgs(argparse::ArgumentParser& parser, int argc, char** argv) {
    parser.add_argument<std::string>("--dataset", "-d")
        .required()
        .help("Path to an ann-benchmarks style hdf5 dataset");

    parser.add_argument<std::string>("--bucket_sizes", "-L")
        .default_value(std::string("100,1000,10000"))
        .help("Comma separated target bucket sizes");

    parser.add_argument<std::string>("--strategies", "-s")
        .default_value(std::string("rpt,random,single_dim"))
        .help("Comma separated strategies: rpt, random, single_dim, kmeans");

    parser.add_argument("--seed")
        .default_value(0)
        .help("Random seed passed to the strategies")
        .scan<'i', int>();

    parser.add_argument("--topk", "-k")
        .default_value(10)
        .help("Number of ground-truth neighbours used for the locality metrics")
        .scan<'i', int>();

    parser.add_argument("--max_queries")
        .default_value(0)
        .help("Limit the queries used for the locality metrics; 0 means all")
        .scan<'i', int>();

    parser.add_argument("--max_base")
        .default_value(0)
        .help("Use only the first N base vectors; 0 means all")
        .scan<'i', int>();

    parser.add_argument<std::string>("--output", "-o")
        .default_value(std::string(""))
        .help("Optional path to write the results as JSON");

    try {
        parser.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        std::exit(1);
    }
}

nlohmann::json
ToJson(const UniformityMetrics& metrics) {
    nlohmann::json json;
    json["num_partitions"] = metrics.num_partitions;
    json["expected_partitions"] = metrics.expected_partitions;
    json["max_size"] = metrics.max_size;
    json["min_size"] = metrics.min_size;
    json["max_min_ratio"] = metrics.max_min_ratio;
    json["mean_size"] = metrics.mean_size;
    json["stddev_size"] = metrics.stddev_size;
    json["gini"] = metrics.gini;
    json["satisfies_bounds"] = metrics.satisfies_bounds;
    return json;
}

nlohmann::json
ToJson(const LocalityMetrics& metrics) {
    nlohmann::json json;
    json["queries_evaluated"] = metrics.queries_evaluated;
    json["k"] = metrics.k;
    json["mean_partitions_touched"] = metrics.mean_partitions_touched;
    json["top1_same_partition_rate"] = metrics.top1_same_partition_rate;
    json["mean_partitions_for_90_recall"] = metrics.mean_partitions_for_90_recall;
    return json;
}

void
PrintHeader() {
    std::cout << std::left << std::setw(12) << "strategy" << std::right << std::setw(8) << "L"
              << std::setw(10) << "parts" << std::setw(8) << "max" << std::setw(8) << "min"
              << std::setw(9) << "max/min" << std::setw(9) << "gini" << std::setw(8) << "bounds"
              << std::setw(12) << "build(ms)" << std::setw(11) << "peak(MiB)" << std::setw(10)
              << "touched" << std::setw(10) << "same1" << std::setw(11) << "scan@90" << std::endl;
}

void
PrintRow(const std::string& strategy,
         uint64_t bucket_size,
         const UniformityMetrics& uniformity,
         const LocalityMetrics& locality,
         double build_time_ms,
         int64_t peak_rss_delta_kb) {
    std::cout << std::left << std::setw(12) << strategy << std::right << std::setw(8) << bucket_size
              << std::setw(10) << uniformity.num_partitions << std::setw(8) << uniformity.max_size
              << std::setw(8) << uniformity.min_size << std::fixed << std::setprecision(2)
              << std::setw(9) << uniformity.max_min_ratio << std::setprecision(4) << std::setw(9)
              << uniformity.gini << std::setw(8) << (uniformity.satisfies_bounds ? "yes" : "no")
              << std::setprecision(1) << std::setw(12) << build_time_ms << std::setprecision(1)
              << std::setw(11) << static_cast<double>(peak_rss_delta_kb) / 1024.0
              << std::setprecision(3) << std::setw(10) << locality.mean_partitions_touched
              << std::setw(10) << locality.top1_same_partition_rate << std::setprecision(2)
              << std::setw(11) << locality.mean_partitions_for_90_recall << std::endl;
}

}  // namespace

int
main(int argc, char** argv) {
    argparse::ArgumentParser parser("rpt_bench");
    ParseArgs(parser, argc, argv);

    std::string dataset_path = parser.get<std::string>("--dataset");
    auto bucket_sizes = ParseBucketSizes(parser.get<std::string>("--bucket_sizes"));
    auto strategies = SplitList(parser.get<std::string>("--strategies"));
    auto seed = static_cast<uint64_t>(parser.get<int>("--seed"));
    auto topk = static_cast<uint64_t>(parser.get<int>("--topk"));
    auto max_queries = static_cast<uint64_t>(parser.get<int>("--max_queries"));
    auto max_base = static_cast<uint64_t>(parser.get<int>("--max_base"));
    std::string output_path = parser.get<std::string>("--output");

    vsag::init();

    auto dataset = vsag::eval::EvalDataset::Load(dataset_path);
    if (dataset == nullptr) {
        std::cerr << "failed to load dataset: " << dataset_path << std::endl;
        return 1;
    }
    if (dataset->GetTrainDataType() != "float32") {
        std::cerr << "rpt_bench only supports float32 datasets, got " << dataset->GetTrainDataType()
                  << std::endl;
        return 1;
    }

    auto dim = static_cast<uint64_t>(dataset->GetDim());
    auto count = static_cast<uint64_t>(dataset->GetNumberOfBase());
    if (max_base > 0 && max_base < count) {
        // Ground-truth ids beyond the truncated base are skipped by the locality metrics.
        count = max_base;
    }
    const auto* base = static_cast<const float*>(dataset->GetTrain());
    auto num_queries = static_cast<uint64_t>(dataset->GetNumberOfQuery());
    uint64_t ground_truth_k = dataset->GetGroundTruthK();
    const int64_t* neighbors = num_queries > 0 ? dataset->GetNeighbors(0) : nullptr;

    std::cout << "dataset: " << dataset_path << "  dim=" << dim << "  base=" << count
              << "  queries=" << num_queries << "  gt_k=" << ground_truth_k << std::endl
              << std::endl;
    PrintHeader();

    nlohmann::json report;
    report["dataset"] = dataset_path;
    report["dim"] = dim;
    report["base_count"] = count;
    report["seed"] = seed;
    report["runs"] = nlohmann::json::array();

    for (uint64_t bucket_size : bucket_sizes) {
        for (const auto& strategy : strategies) {
            PartitionRequest request;
            request.dim = dim;
            request.count = count;
            request.data = base;
            request.bucket_size = bucket_size;
            request.seed = seed;

            auto result = vsag::rpt_bench::RunStrategy(strategy, request);
            if (not result.error.empty()) {
                std::cerr << strategy << " (L=" << bucket_size << ") failed: " << result.error
                          << std::endl;
                continue;
            }

            auto uniformity =
                vsag::rpt_bench::ComputeUniformity(result.partitions, count, bucket_size);
            auto locality = vsag::rpt_bench::ComputeLocality(result.partitions,
                                                             count,
                                                             neighbors,
                                                             num_queries,
                                                             ground_truth_k,
                                                             topk,
                                                             max_queries);
            PrintRow(strategy,
                     bucket_size,
                     uniformity,
                     locality,
                     result.build_time_ms,
                     result.peak_rss_delta_kb);

            nlohmann::json run;
            run["strategy"] = strategy;
            run["bucket_size"] = bucket_size;
            run["build_time_ms"] = result.build_time_ms;
            run["peak_rss_delta_kb"] = result.peak_rss_delta_kb;
            run["uniformity"] = ToJson(uniformity);
            run["locality"] = ToJson(locality);
            report["runs"].push_back(run);
        }
    }

    if (not output_path.empty()) {
        std::ofstream out(output_path);
        if (not out.is_open()) {
            std::cerr << "failed to open output file: " << output_path << std::endl;
            return 1;
        }
        out << report.dump(2) << std::endl;
        std::cout << std::endl << "results written to " << output_path << std::endl;
    }

    return 0;
}
