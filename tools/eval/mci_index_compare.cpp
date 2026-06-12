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
#include <omp.h>
#include <vsag/vsag.h>

#include <algorithm>
#include <argparse/argparse.hpp>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "eval_dataset.h"
#include "json_types.h"
#include "typing.h"

namespace {

using vsag::JsonType;
using vsag::eval::EvalDataset;
using vsag::eval::EvalDatasetPtr;
using Clock = std::chrono::steady_clock;

class LabelFilter : public vsag::Filter {
public:
    LabelFilter(const int64_t* train_labels,
                int64_t test_label,
                float valid_ratio,
                const std::vector<int64_t>* valid_ids)
        : train_labels_(train_labels),
          test_label_(test_label),
          valid_ratio_(valid_ratio),
          valid_ids_(valid_ids) {
    }

    bool
    CheckValid(int64_t id) const override {
        return train_labels_[id] == test_label_;
    }

    float
    ValidRatio() const override {
        return valid_ratio_;
    }

    void
    GetValidIds(const int64_t** valid_ids, int64_t& count) const override {
        if (valid_ids_ == nullptr or valid_ids_->empty()) {
            *valid_ids = nullptr;
            count = 0;
            return;
        }
        *valid_ids = valid_ids_->data();
        count = static_cast<int64_t>(valid_ids_->size());
    }

private:
    const int64_t* train_labels_;
    int64_t test_label_;
    float valid_ratio_;
    const std::vector<int64_t>* valid_ids_{nullptr};
};

struct IndexSpec {
    std::string name;
    std::string path;
    std::string create_params;
};

struct IndexReport {
    std::string name;
    std::string path;
    uint64_t file_size{0};
    int64_t num_elements{0};
    int64_t memory_usage{0};
    uint64_t query_count{0};
    double recall_avg{0.0};
    double qps{0.0};
    double total_search_seconds{0.0};
    uint64_t hgraph_route_count{0};
    uint64_t mci_route_count{0};
    uint64_t empty_stats_count{0};
    double avg_dist_cmp{0.0};
    double avg_hops{0.0};
    double avg_total_clique_count{0.0};
    double avg_base_clique_count{0.0};
    double avg_delta_clique_count{0.0};
    double avg_valid_ratio{0.0};
};

void
check(bool condition, const std::string& message) {
    if (not condition) {
        throw std::runtime_error(message);
    }
}

double
seconds_since(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double
recall_for_query(const int64_t* neighbors,
                 uint64_t result_count,
                 const int64_t* gt_neighbors,
                 EvalDataset* dataset,
                 const void* query_vector,
                 uint64_t topk) {
    const auto dim = static_cast<size_t>(dataset->GetDim());
    const auto base_count = dataset->GetNumberOfBase();
    auto distance_func = dataset->GetDistanceFunc();

    std::vector<float> gt_distances;
    gt_distances.reserve(topk);
    for (uint64_t i = 0; i < topk; ++i) {
        if (gt_neighbors[i] < 0 or gt_neighbors[i] >= base_count) {
            break;
        }
        gt_distances.push_back(
            distance_func(query_vector, dataset->GetOneTrain(gt_neighbors[i]), &dim));
    }
    if (gt_distances.empty()) {
        return 0.0;
    }
    std::sort(gt_distances.begin(), gt_distances.end());
    const float threshold = gt_distances.back() + 2e-6F;

    uint64_t count = 0;
    if (neighbors != nullptr) {
        for (uint64_t i = 0; i < result_count; ++i) {
            if (neighbors[i] < 0 or neighbors[i] >= base_count) {
                continue;
            }
            auto distance = distance_func(query_vector, dataset->GetOneTrain(neighbors[i]), &dim);
            if (distance <= threshold) {
                ++count;
            }
        }
    }
    return static_cast<double>(count) / static_cast<double>(gt_distances.size());
}

vsag::IndexPtr
load_index(const IndexSpec& spec) {
    auto created = vsag::Factory::CreateIndex("mci", spec.create_params);
    check(created.has_value(),
          fmt::format("[{}] create failed: {}", spec.name, created.error().message));
    auto index = created.value();
    std::ifstream input(spec.path, std::ios::binary);
    check(input.is_open(), fmt::format("[{}] failed to open {}", spec.name, spec.path));
    auto result = index->Deserialize(input);
    check(result.has_value(),
          fmt::format("[{}] deserialize failed: {}", spec.name, result.error().message));
    return index;
}

std::unordered_map<int64_t, std::vector<int64_t>>
build_valid_ids(const EvalDatasetPtr& dataset) {
    const auto* train_labels = dataset->GetTrainLabels().get();
    check(train_labels != nullptr, "dataset must have train_labels");
    std::unordered_map<int64_t, std::vector<int64_t>> valid_ids;
    const auto total_base = dataset->GetNumberOfBase();
    for (int64_t id = 0; id < total_base; ++id) {
        valid_ids[train_labels[id]].push_back(id);
    }
    return valid_ids;
}

int64_t
json_int_or_zero(const JsonType& stats, const char* key) {
    if (not stats.Contains(key)) {
        return 0;
    }
    return stats[key].GetInt();
}

double
json_float_or_zero(const JsonType& stats, const char* key) {
    if (not stats.Contains(key)) {
        return 0.0;
    }
    return stats[key].GetFloat();
}

void
accumulate_stats(IndexReport& report, const std::string& stats_string) {
    if (stats_string.empty()) {
        ++report.empty_stats_count;
        return;
    }
    auto stats = JsonType::Parse(stats_string);
    if (stats.Contains("mci_hybrid_route")) {
        const std::string route = stats["mci_hybrid_route"].GetString();
        if (route == "hgraph") {
            ++report.hgraph_route_count;
        } else if (route == "mci") {
            ++report.mci_route_count;
        }
    }
    report.avg_dist_cmp += static_cast<double>(json_int_or_zero(stats, "dist_cmp"));
    report.avg_hops += static_cast<double>(json_int_or_zero(stats, "hops"));
    report.avg_total_clique_count +=
        static_cast<double>(json_int_or_zero(stats, "total_clique_count"));
    report.avg_base_clique_count +=
        static_cast<double>(json_int_or_zero(stats, "base_clique_count"));
    report.avg_delta_clique_count +=
        static_cast<double>(json_int_or_zero(stats, "delta_clique_count"));
    report.avg_valid_ratio += json_float_or_zero(stats, "mci_hybrid_valid_ratio");
}

IndexReport
run_index(const IndexSpec& spec,
          const EvalDatasetPtr& dataset,
          const std::unordered_map<int64_t, std::vector<int64_t>>& valid_ids,
          const std::string& search_params,
          uint64_t topk,
          uint64_t query_count_cap,
          int num_threads) {
    IndexReport report;
    report.name = spec.name;
    report.path = spec.path;
    report.file_size = std::filesystem::file_size(spec.path);

    auto load_start = Clock::now();
    auto index = load_index(spec);
    std::cout << fmt::format("[{}] loaded in {:.2f}s elements={} file_size={} memory={}\n",
                             spec.name,
                             seconds_since(load_start),
                             index->GetNumElements(),
                             report.file_size,
                             index->GetMemoryUsage());
    report.num_elements = index->GetNumElements();
    report.memory_usage = index->GetMemoryUsage();

    const auto query_count = static_cast<uint64_t>(dataset->GetNumberOfQuery());
    const uint64_t total_queries =
        query_count_cap == 0 ? query_count : std::min<uint64_t>(query_count_cap, query_count);
    report.query_count = total_queries;

    const auto* train_labels = dataset->GetTrainLabels().get();
    const auto* test_labels = dataset->GetTestLabels().get();
    check(train_labels != nullptr and test_labels != nullptr,
          "dataset must contain train_labels and test_labels for filtered search");

    std::vector<double> recalls(total_queries, 0.0);
    std::vector<std::string> stats_strings(total_queries);
    auto search_start = Clock::now();
    omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic)
    for (int64_t qi = 0; qi < static_cast<int64_t>(total_queries); ++qi) {
        const auto i = qi % static_cast<int64_t>(query_count);
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dataset->GetDim())
            ->Float32Vectors(static_cast<const float*>(dataset->GetOneTest(i)))
            ->Owner(false);
        const auto test_label = test_labels[i];
        const std::vector<int64_t>* vids = nullptr;
        auto it = valid_ids.find(test_label);
        if (it != valid_ids.end()) {
            vids = &it->second;
        }
        auto filter = std::make_shared<LabelFilter>(
            train_labels, test_label, dataset->GetValidRatio(test_label), vids);
        auto result = index->KnnSearch(query, topk, search_params, filter);
        if (not result.has_value()) {
            continue;
        }
        recalls[qi] = recall_for_query(result.value()->GetIds(),
                                       static_cast<uint64_t>(result.value()->GetDim()),
                                       dataset->GetNeighbors(i),
                                       dataset.get(),
                                       dataset->GetOneTest(i),
                                       topk);
        stats_strings[qi] = result.value()->GetStatistics();
    }
    report.total_search_seconds = seconds_since(search_start);
    report.qps = static_cast<double>(total_queries) / std::max(report.total_search_seconds, 1e-9);

    for (uint64_t qi = 0; qi < total_queries; ++qi) {
        report.recall_avg += recalls[qi];
        accumulate_stats(report, stats_strings[qi]);
    }
    const auto denom = static_cast<double>(std::max<uint64_t>(total_queries, 1));
    report.recall_avg /= denom;
    report.avg_dist_cmp /= denom;
    report.avg_hops /= denom;
    report.avg_total_clique_count /= denom;
    report.avg_base_clique_count /= denom;
    report.avg_delta_clique_count /= denom;
    report.avg_valid_ratio /= denom;
    return report;
}

void
print_report(const IndexReport& report) {
    std::cout << fmt::format("[{}] recall_avg={:.4f} qps={:.1f} queries={} search_time={:.3f}s\n",
                             report.name,
                             report.recall_avg,
                             report.qps,
                             report.query_count,
                             report.total_search_seconds);
    std::cout << fmt::format("[{}] routes hgraph={} mci={} empty_stats={} avg_valid_ratio={:.4f}\n",
                             report.name,
                             report.hgraph_route_count,
                             report.mci_route_count,
                             report.empty_stats_count,
                             report.avg_valid_ratio);
    std::cout << fmt::format(
        "[{}] avg_dist_cmp={:.1f} avg_hops={:.1f} avg_cliques total={:.1f} base={:.1f} "
        "delta={:.1f}\n",
        report.name,
        report.avg_dist_cmp,
        report.avg_hops,
        report.avg_total_clique_count,
        report.avg_base_clique_count,
        report.avg_delta_clique_count);
}

void
parse_args(argparse::ArgumentParser& parser, int argc, char** argv) {
    parser.add_argument("--datapath", "-d").required().help("HDF5 dataset path.");
    parser.add_argument("--search_params", "-s").required().help("MCI search params JSON.");
    parser.add_argument("--topk", "-k").default_value(20).scan<'i', int>();
    parser.add_argument("--query_count")
        .default_value(static_cast<uint64_t>(0))
        .scan<'u', uint64_t>();
    parser.add_argument("--num_threads", "-t").default_value(16).scan<'i', int>();
    parser.add_argument("--left_name").required();
    parser.add_argument("--left_index").required();
    parser.add_argument("--left_create_params").required();
    parser.add_argument("--right_name").required();
    parser.add_argument("--right_index").required();
    parser.add_argument("--right_create_params").required();
    parser.parse_args(argc, argv);
}

}  // namespace

int
main(int argc, char** argv) {
    vsag::init();
    argparse::ArgumentParser parser("mci_index_compare");
    parse_args(parser, argc, argv);

    const auto datapath = parser.get<std::string>("--datapath");
    const auto search_params = parser.get<std::string>("--search_params");
    const auto topk = static_cast<uint64_t>(parser.get<int>("--topk"));
    const auto query_count = parser.get<uint64_t>("--query_count");
    const auto num_threads = parser.get<int>("--num_threads");

    IndexSpec left{parser.get<std::string>("--left_name"),
                   parser.get<std::string>("--left_index"),
                   parser.get<std::string>("--left_create_params")};
    IndexSpec right{parser.get<std::string>("--right_name"),
                    parser.get<std::string>("--right_index"),
                    parser.get<std::string>("--right_create_params")};

    std::cout << "[compare] loading dataset: " << datapath << std::endl;
    auto dataset = EvalDataset::Load(datapath);
    check(dataset->GetVectorType() == vsag::eval::DENSE_VECTORS, "only dense datasets supported");
    check(dataset->GetTrainDataType() == vsag::DATATYPE_FLOAT32, "only float32 train supported");
    std::cout << fmt::format("[compare] dim={} base={} queries={}\n",
                             dataset->GetDim(),
                             dataset->GetNumberOfBase(),
                             dataset->GetNumberOfQuery());
    auto valid_ids = build_valid_ids(dataset);

    auto left_report =
        run_index(left, dataset, valid_ids, search_params, topk, query_count, num_threads);
    print_report(left_report);
    auto right_report =
        run_index(right, dataset, valid_ids, search_params, topk, query_count, num_threads);
    print_report(right_report);

    std::cout << "\n==================== COMPARE SUMMARY ====================\n";
    std::cout << fmt::format("{:<14} {:>10} {:>12} {:>12} {:>10} {:>10} {:>12}\n",
                             "index",
                             "recall",
                             "qps",
                             "file_bytes",
                             "hgraph",
                             "mci",
                             "avg_delta");
    for (const auto& report : {left_report, right_report}) {
        std::cout << fmt::format("{:<14} {:>10.4f} {:>12.1f} {:>12} {:>10} {:>10} {:>12.1f}\n",
                                 report.name,
                                 report.recall_avg,
                                 report.qps,
                                 report.file_size,
                                 report.hgraph_route_count,
                                 report.mci_route_count,
                                 report.avg_delta_clique_count);
    }
    std::cout << fmt::format("{:<14} {:>+10.4f} {:>+12.1f}\n",
                             "right-left",
                             right_report.recall_avg - left_report.recall_avg,
                             right_report.qps - left_report.qps);
    std::cout << "=========================================================\n";
    return 0;
}
