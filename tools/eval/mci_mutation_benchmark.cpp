// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <omp.h>

#include <algorithm>
#include <argparse/argparse.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "eval_dataset.h"
#include "vsag/options.h"
#include "vsag/vsag.h"

namespace {

using Clock = std::chrono::steady_clock;

struct benchmark_config {
    std::string dataset_path{"/root/data/codefilter-10k-384-angular-f32.hdf5"};
    std::string output_path{"/tmp/vsag_mci_mutation/codefilter_mci_mutation.csv"};
    std::string save_initial_index;
    std::string ef_search_values{"40,80,160"};
    uint64_t max_base{0};
    uint64_t stress_rounds{0};
    uint64_t stress_initial_count{800000};
    uint64_t stress_step_count{0};
    std::string stress_mode{"toggle"};
    uint64_t query_count{200};
    uint64_t search_count{10000};
    uint64_t warmup_count{64};
    uint64_t mutation_batch_size{10};
    uint64_t topk{10};
    uint64_t build_threads{std::min(16U, std::max(1U, std::thread::hardware_concurrency()))};
    uint64_t search_threads{std::min(16U, std::max(1U, std::thread::hardware_concurrency()))};
    uint64_t max_degree{32};
    uint64_t ef_construction{200};
    uint64_t mci_mcs{50};
    uint64_t mci_clique_max{50};
    uint64_t mci_incremental_clique_max{0};
    uint64_t mci_incremental_added_mct{3};
    uint64_t mci_delete_clique_size_threshold{3};
    uint64_t mci_delete_node_mct_threshold{3};
    uint64_t random_seed{20260907};
    float delete_fraction{0.1F};
    float mci_alpha{1.2F};
    float mci_incremental_join_ratio_threshold{0.6F};
    float mci_seed_ratio{0.1F};
    float route_threshold{1.0F};
    bool flush_after_mutation{false};
    bool force_remove{false};
};

struct stage_stats {
    uint64_t mci_total_nodes{0};
    uint64_t mci_covered_nodes{0};
    uint64_t mci_total_clique_count{0};
    uint64_t mci_delta_clique_count{0};
    uint64_t mci_retired_clique_count{0};
    uint64_t mci_inactive_node_count{0};
    uint64_t mci_memory_usage{0};
    uint64_t total_memberships{0};
    uint64_t delta_extra_memberships{0};
    double avg_clique_size{0.0};
    double avg_memberships{0.0};
};

struct curve_point {
    double qps{0.0};
    double recall{0.0};
    double mci_route_ratio{0.0};
    double wall_seconds{0.0};
    double raw_float_ratio{0.0};
    double avg_dist_cmp{0.0};
    double avg_hops{0.0};
    double avg_seeds{0.0};
};

struct query_case {
    vsag::DatasetPtr query;
    vsag::FilterPtr filter;
    std::vector<int64_t> ground_truth;
};

class LabelFilter final : public vsag::Filter {
public:
    LabelFilter(const int64_t* train_labels,
                const uint8_t* active_ids,
                uint64_t base_count,
                int64_t target_label,
                float valid_ratio)
        : train_labels_(train_labels),
          active_ids_(active_ids),
          base_count_(base_count),
          target_label_(target_label),
          valid_ratio_(valid_ratio) {
        valid_ids_.reserve(static_cast<uint64_t>(valid_ratio * static_cast<float>(base_count)) + 1);
        for (uint64_t id = 0; id < base_count; ++id) {
            if (active_ids_[id] != 0 and train_labels_[id] == target_label_) {
                valid_ids_.push_back(static_cast<int64_t>(id));
            }
        }
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        return id >= 0 and static_cast<uint64_t>(id) < base_count_ and active_ids_[id] != 0 and
               train_labels_[id] == target_label_;
    }

    [[nodiscard]] float
    ValidRatio() const override {
        return valid_ratio_;
    }

    void
    GetValidIds(const int64_t** valid_ids, int64_t& count) const override {
        *valid_ids = valid_ids_.empty() ? nullptr : valid_ids_.data();
        count = static_cast<int64_t>(valid_ids_.size());
    }

private:
    const int64_t* train_labels_{nullptr};
    const uint8_t* active_ids_{nullptr};
    uint64_t base_count_{0};
    int64_t target_label_{0};
    float valid_ratio_{0.0F};
    std::vector<int64_t> valid_ids_;
};

template <typename T>
T
take_expected(tl::expected<T, vsag::Error>&& result, const std::string& operation) {
    if (not result.has_value()) {
        throw std::runtime_error(operation + " failed: " + result.error().message);
    }
    if constexpr (not std::is_void_v<T>) {
        return std::move(result.value());
    }
}

std::vector<uint64_t>
parse_ef_search_values(const std::string& text) {
    std::vector<uint64_t> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            continue;
        }
        const auto value = std::stoull(token);
        if (value == 0) {
            throw std::invalid_argument("ef_search values must be positive");
        }
        values.push_back(value);
    }
    if (values.empty()) {
        throw std::invalid_argument("at least one ef_search value is required");
    }
    return values;
}

benchmark_config
parse_arguments(int argc, char** argv) {
    benchmark_config config;
    argparse::ArgumentParser parser("mci_mutation_benchmark");
    parser.add_argument("--dataset").default_value(config.dataset_path);
    parser.add_argument("--output").default_value(config.output_path);
    parser.add_argument("--save-initial-index").default_value(config.save_initial_index);
    parser.add_argument("--flush-after-mutation").default_value(false).implicit_value(true);
    parser.add_argument("--force-remove").default_value(false).implicit_value(true);
    parser.add_argument("--ef-search-values").default_value(config.ef_search_values);
    parser.add_argument("--max-base").default_value(config.max_base).scan<'i', uint64_t>();
    parser.add_argument("--stress-rounds")
        .default_value(config.stress_rounds)
        .scan<'i', uint64_t>();
    parser.add_argument("--stress-initial-count")
        .default_value(config.stress_initial_count)
        .scan<'i', uint64_t>();
    parser.add_argument("--stress-step-count")
        .default_value(config.stress_step_count)
        .scan<'i', uint64_t>();
    parser.add_argument("--stress-mode").default_value(config.stress_mode);
    parser.add_argument("--query-count").default_value(config.query_count).scan<'i', uint64_t>();
    parser.add_argument("--search-count").default_value(config.search_count).scan<'i', uint64_t>();
    parser.add_argument("--warmup-count").default_value(config.warmup_count).scan<'i', uint64_t>();
    parser.add_argument("--mutation-batch-size")
        .default_value(config.mutation_batch_size)
        .scan<'i', uint64_t>();
    parser.add_argument("--topk").default_value(config.topk).scan<'i', uint64_t>();
    parser.add_argument("--build-threads")
        .default_value(config.build_threads)
        .scan<'i', uint64_t>();
    parser.add_argument("--search-threads")
        .default_value(config.search_threads)
        .scan<'i', uint64_t>();
    parser.add_argument("--max-degree").default_value(config.max_degree).scan<'i', uint64_t>();
    parser.add_argument("--ef-construction")
        .default_value(config.ef_construction)
        .scan<'i', uint64_t>();
    parser.add_argument("--mci-mcs").default_value(config.mci_mcs).scan<'i', uint64_t>();
    parser.add_argument("--mci-clique-max")
        .default_value(config.mci_clique_max)
        .scan<'i', uint64_t>();
    parser.add_argument("--mci-incremental-clique-max")
        .default_value(config.mci_incremental_clique_max)
        .scan<'i', uint64_t>();
    parser.add_argument("--mci-incremental-added-mct")
        .default_value(config.mci_incremental_added_mct)
        .scan<'i', uint64_t>();
    parser.add_argument("--mci-incremental-join-ratio-threshold")
        .default_value(config.mci_incremental_join_ratio_threshold)
        .scan<'f', float>();
    parser.add_argument("--mci-delete-clique-size-threshold")
        .default_value(config.mci_delete_clique_size_threshold)
        .scan<'i', uint64_t>();
    parser.add_argument("--mci-delete-node-mct-threshold")
        .default_value(config.mci_delete_node_mct_threshold)
        .scan<'i', uint64_t>();
    parser.add_argument("--random-seed").default_value(config.random_seed).scan<'i', uint64_t>();
    parser.add_argument("--delete-fraction")
        .default_value(config.delete_fraction)
        .scan<'f', float>();
    parser.add_argument("--mci-alpha").default_value(config.mci_alpha).scan<'f', float>();
    parser.add_argument("--mci-seed-ratio").default_value(config.mci_seed_ratio).scan<'f', float>();
    parser.add_argument("--route-threshold")
        .default_value(config.route_threshold)
        .scan<'f', float>();

    try {
        parser.parse_args(argc, argv);
    } catch (const std::runtime_error& error) {
        std::cerr << error.what() << '\n' << parser;
        throw;
    }
    config.dataset_path = parser.get<std::string>("--dataset");
    config.output_path = parser.get<std::string>("--output");
    config.save_initial_index = parser.get<std::string>("--save-initial-index");
    config.flush_after_mutation = parser.get<bool>("--flush-after-mutation");
    config.force_remove = parser.get<bool>("--force-remove");
    config.ef_search_values = parser.get<std::string>("--ef-search-values");
    config.max_base = parser.get<uint64_t>("--max-base");
    config.stress_rounds = parser.get<uint64_t>("--stress-rounds");
    config.stress_initial_count = parser.get<uint64_t>("--stress-initial-count");
    config.stress_step_count = parser.get<uint64_t>("--stress-step-count");
    config.stress_mode = parser.get<std::string>("--stress-mode");
    if (config.stress_mode != "toggle" and config.stress_mode != "alternate") {
        throw std::invalid_argument("stress-mode must be toggle or alternate");
    }
    config.query_count = parser.get<uint64_t>("--query-count");
    config.search_count = parser.get<uint64_t>("--search-count");
    config.warmup_count = parser.get<uint64_t>("--warmup-count");
    config.mutation_batch_size = parser.get<uint64_t>("--mutation-batch-size");
    config.topk = parser.get<uint64_t>("--topk");
    config.build_threads = parser.get<uint64_t>("--build-threads");
    config.search_threads = parser.get<uint64_t>("--search-threads");
    config.max_degree = parser.get<uint64_t>("--max-degree");
    config.ef_construction = parser.get<uint64_t>("--ef-construction");
    config.mci_mcs = parser.get<uint64_t>("--mci-mcs");
    config.mci_clique_max = parser.get<uint64_t>("--mci-clique-max");
    config.mci_incremental_clique_max = parser.get<uint64_t>("--mci-incremental-clique-max");
    if (config.mci_incremental_clique_max == 0) {
        config.mci_incremental_clique_max = config.mci_clique_max;
    }
    config.mci_incremental_added_mct = parser.get<uint64_t>("--mci-incremental-added-mct");
    config.mci_incremental_join_ratio_threshold =
        parser.get<float>("--mci-incremental-join-ratio-threshold");
    config.mci_delete_clique_size_threshold =
        parser.get<uint64_t>("--mci-delete-clique-size-threshold");
    config.mci_delete_node_mct_threshold = parser.get<uint64_t>("--mci-delete-node-mct-threshold");
    config.random_seed = parser.get<uint64_t>("--random-seed");
    config.delete_fraction = parser.get<float>("--delete-fraction");
    config.mci_alpha = parser.get<float>("--mci-alpha");
    config.mci_seed_ratio = parser.get<float>("--mci-seed-ratio");
    config.route_threshold = parser.get<float>("--route-threshold");

    if (config.topk == 0 or config.search_count == 0 or config.mutation_batch_size == 0 or
        config.build_threads == 0 or config.search_threads == 0 or
        config.mci_delete_clique_size_threshold == 0 or config.mci_delete_node_mct_threshold == 0) {
        throw std::invalid_argument("topk, counts, batch size, and thread counts must be positive");
    }
    if (not std::isfinite(config.delete_fraction) or config.delete_fraction <= 0.0F or
        config.delete_fraction >= 0.5F) {
        throw std::invalid_argument("delete-fraction must be finite and in (0, 0.5)");
    }
    if (config.mci_incremental_clique_max < 2 or config.mci_incremental_added_mct == 0 or
        not std::isfinite(config.mci_incremental_join_ratio_threshold) or
        config.mci_incremental_join_ratio_threshold < 0.0F or
        config.mci_incremental_join_ratio_threshold > 1.0F) {
        throw std::invalid_argument(
            "invalid MCI incremental clique cap, membership, or join ratio");
    }
    if (config.stress_rounds != 0 and not config.save_initial_index.empty()) {
        throw std::invalid_argument("initial-index saving requires five-stage mode");
    }
    return config;
}

std::string
make_build_parameters(const benchmark_config& config, uint64_t dim) {
    nlohmann::json parameters;
    parameters["dtype"] = "float32";
    parameters["metric_type"] = "cosine";
    parameters["dim"] = dim;
    auto& index = parameters["index_param"];
    index["base_quantization_type"] = "fp32";
    index["base_io_type"] = "memory_io";
    index["graph_type"] = "nsw";
    index["max_degree"] = config.max_degree;
    index["ef_construction"] = config.ef_construction;
    index["build_thread_count"] = config.build_threads;
    index["use_mci"] = true;
    index["support_force_remove"] = config.force_remove;
    index["mci_mcs"] = config.mci_mcs;
    index["mci_clique_max"] = config.mci_clique_max;
    index["mci_incremental_clique_max"] = config.mci_incremental_clique_max;
    index["mci_incremental_added_mct"] = config.mci_incremental_added_mct;
    index["mci_incremental_join_ratio_threshold"] = config.mci_incremental_join_ratio_threshold;
    index["mci_delete_clique_size_threshold"] = config.mci_delete_clique_size_threshold;
    index["mci_delete_node_mct_threshold"] = config.mci_delete_node_mct_threshold;
    index["mci_alpha"] = config.mci_alpha;
    index["mci_knng_source"] = "hgraph";
    return parameters.dump();
}

std::string
make_search_parameters(const benchmark_config& config, uint64_t ef_search) {
    nlohmann::json parameters;
    auto& hgraph = parameters["hgraph"];
    hgraph["ef_search"] = ef_search;
    hgraph["use_mci"] = true;
    hgraph["mci_seed_ratio"] = config.mci_seed_ratio;
    hgraph["hgraph_valid_ratio_threshold"] = config.route_threshold;
    return parameters.dump();
}

std::vector<std::vector<int64_t>>
prepare_ground_truth(const vsag::eval::EvalDatasetPtr& dataset,
                     uint64_t base_count,
                     uint64_t requested_query_count,
                     uint64_t topk,
                     std::vector<uint64_t>& query_ids) {
    const auto available_queries = static_cast<uint64_t>(dataset->GetNumberOfQuery());
    const auto target_queries = requested_query_count == 0
                                    ? available_queries
                                    : std::min<uint64_t>(requested_query_count, available_queries);
    const auto full_base_count = static_cast<uint64_t>(dataset->GetNumberOfBase());
    std::vector<std::vector<int64_t>> ground_truth;
    ground_truth.reserve(target_queries);
    query_ids.reserve(target_queries);
    const auto train_labels = dataset->GetTrainLabels();
    const auto test_labels = dataset->GetTestLabels();
    if (train_labels == nullptr or test_labels == nullptr) {
        throw std::invalid_argument("the mutation benchmark requires train/test labels");
    }
    const auto* train_label_values = train_labels.get();
    const auto* test_label_values = test_labels.get();
    bool use_provided = base_count == full_base_count and
                        topk <= static_cast<uint64_t>(dataset->GetGroundTruthCount());
    for (uint64_t query_id = 0; use_provided and query_id < target_queries; ++query_id) {
        const auto* provided = dataset->GetNeighbors(static_cast<int64_t>(query_id));
        for (uint64_t rank = 0; rank < topk; ++rank) {
            const auto id = provided[rank];
            if (id < 0 or static_cast<uint64_t>(id) >= base_count or
                train_label_values[id] != test_label_values[query_id]) {
                use_provided = false;
                break;
            }
        }
    }
    if (not use_provided) {
        std::cout << "[ground-truth] provided neighbors do not match the filtered workload; "
                     "recomputing exact neighbors"
                  << std::endl;
    }

    for (uint64_t query_id = 0; query_id < available_queries and query_ids.size() < target_queries;
         ++query_id) {
        std::vector<int64_t> neighbors;
        neighbors.reserve(topk);
        if (use_provided) {
            auto* provided = dataset->GetNeighbors(static_cast<int64_t>(query_id));
            for (uint64_t rank = 0; rank < topk; ++rank) {
                neighbors.push_back(provided[rank]);
            }
        } else {
            using Neighbor = std::pair<float, int64_t>;
            std::priority_queue<Neighbor> nearest;
            auto distance = dataset->GetDistanceFunc();
            const auto* query = dataset->GetOneTest(static_cast<int64_t>(query_id));
            auto dim = static_cast<size_t>(dataset->GetDim());
            for (uint64_t base_id = 0; base_id < base_count; ++base_id) {
                if (train_label_values[base_id] != test_label_values[query_id]) {
                    continue;
                }
                Neighbor candidate{
                    distance(query, dataset->GetOneTrain(static_cast<int64_t>(base_id)), &dim),
                    static_cast<int64_t>(base_id)};
                if (nearest.size() < topk) {
                    nearest.push(candidate);
                } else if (candidate < nearest.top()) {
                    nearest.pop();
                    nearest.push(candidate);
                }
            }
            if (nearest.size() < topk) {
                continue;
            }
            neighbors.resize(topk);
            for (uint64_t rank = topk; rank > 0; --rank) {
                neighbors[rank - 1] = nearest.top().second;
                nearest.pop();
            }
        }
        query_ids.push_back(query_id);
        ground_truth.push_back(std::move(neighbors));
    }
    if (query_ids.empty()) {
        throw std::invalid_argument("no query has enough filtered ground-truth neighbors");
    }
    std::cout << "[ground-truth] mode=" << (use_provided ? "provided" : "exact-filtered")
              << " queries=" << query_ids.size() << " topk=" << topk << std::endl;
    return ground_truth;
}

std::vector<int64_t>
make_delete_order(uint64_t base_count,
                  const std::vector<std::vector<int64_t>>& ground_truth,
                  uint64_t delete_count,
                  uint64_t seed) {
    std::vector<uint8_t> protected_ids(base_count, 0);
    for (const auto& neighbors : ground_truth) {
        for (auto id : neighbors) {
            protected_ids[static_cast<uint64_t>(id)] = 1;
        }
    }
    std::vector<int64_t> candidates;
    candidates.reserve(base_count);
    for (uint64_t id = 0; id < base_count; ++id) {
        if (protected_ids[id] == 0) {
            candidates.push_back(static_cast<int64_t>(id));
        }
    }
    if (delete_count * 2 > candidates.size()) {
        throw std::invalid_argument("not enough non-ground-truth vectors for two delete stages");
    }
    std::mt19937_64 random(seed);
    std::shuffle(candidates.begin(), candidates.end(), random);
    candidates.resize(delete_count * 2);
    return candidates;
}

std::vector<uint64_t>
make_active_label_counts(const int64_t* train_labels, uint64_t base_count) {
    int64_t max_label = 0;
    for (uint64_t id = 0; id < base_count; ++id) {
        max_label = std::max(max_label, train_labels[id]);
    }
    if (max_label < 0) {
        throw std::invalid_argument("negative train labels are unsupported");
    }
    std::vector<uint64_t> counts(static_cast<uint64_t>(max_label) + 1, 0);
    for (uint64_t id = 0; id < base_count; ++id) {
        ++counts[static_cast<uint64_t>(train_labels[id])];
    }
    return counts;
}

std::vector<query_case>
make_query_cases(const vsag::eval::EvalDatasetPtr& dataset,
                 uint64_t base_count,
                 const std::vector<uint64_t>& query_ids,
                 const std::vector<std::vector<int64_t>>& ground_truth,
                 const std::vector<uint64_t>& active_label_counts,
                 const std::vector<uint8_t>& active_ids,
                 uint64_t active_count) {
    std::vector<query_case> cases;
    cases.reserve(query_ids.size());
    const auto train_labels = dataset->GetTrainLabels();
    const auto test_labels = dataset->GetTestLabels();
    const auto* test_label_values = test_labels.get();
    for (uint64_t i = 0; i < query_ids.size(); ++i) {
        const auto query_id = query_ids[i];
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dataset->GetDim())
            ->Float32Vectors(
                static_cast<const float*>(dataset->GetOneTest(static_cast<int64_t>(query_id))))
            ->Owner(false);
        const auto label = test_label_values[query_id];
        const auto label_index = static_cast<uint64_t>(label);
        const auto valid_count =
            label_index < active_label_counts.size() ? active_label_counts[label_index] : 0;
        const auto ratio = active_count == 0
                               ? 0.0F
                               : static_cast<float>(valid_count) / static_cast<float>(active_count);
        cases.push_back(
            query_case{std::move(query),
                       std::make_shared<LabelFilter>(
                           train_labels.get(), active_ids.data(), base_count, label, ratio),
                       ground_truth[i]});
    }
    return cases;
}

double
recall_at_k(const vsag::DatasetPtr& result, const std::vector<int64_t>& ground_truth) {
    std::unordered_set<int64_t> truth(ground_truth.begin(), ground_truth.end());
    uint64_t hits = 0;
    const auto result_count = static_cast<uint64_t>(result->GetDim());
    for (uint64_t rank = 0; rank < std::min<uint64_t>(ground_truth.size(), result_count); ++rank) {
        if (truth.count(result->GetIds()[rank]) != 0) {
            ++hits;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(ground_truth.size());
}

curve_point
measure_curve_point(const vsag::IndexPtr& index,
                    const std::vector<query_case>& queries,
                    const std::string& search_parameters,
                    const benchmark_config& config) {
    uint64_t mci_routes = 0;
    uint64_t raw_float_routes = 0;
    double recall_sum = 0.0;
    double dist_cmp_sum = 0.0;
    double hops_sum = 0.0;
    double seeds_sum = 0.0;
    for (const auto& query : queries) {
        auto result = take_expected(
            index->KnnSearch(
                query.query, static_cast<int64_t>(config.topk), search_parameters, query.filter),
            "recall search");
        recall_sum += recall_at_k(result, query.ground_truth);
        const auto route = result->GetStatistics({"mci_hybrid_route"});
        if (not route.empty() and route.front() == R"("mci")") {
            ++mci_routes;
        }
        const auto raw = result->GetStatistics({"mci_raw_float_csr"});
        if (not raw.empty() and raw.front() == "true") {
            ++raw_float_routes;
        }
        const auto diagnostics = result->GetStatistics({"dist_cmp", "hops", "mci_seed_count"});
        if (diagnostics.size() != 3) {
            throw std::runtime_error("search diagnostics are missing");
        }
        dist_cmp_sum += std::stod(diagnostics[0]);
        hops_sum += std::stod(diagnostics[1]);
        seeds_sum += std::stod(diagnostics[2]);
    }

    for (uint64_t i = 0; i < config.warmup_count; ++i) {
        const auto& query = queries[i % queries.size()];
        take_expected(
            index->KnnSearch(
                query.query, static_cast<int64_t>(config.topk), search_parameters, query.filter),
            "warmup search");
    }

    std::atomic<bool> failed{false};
    std::mutex error_mutex;
    std::string error_message;
    omp_set_num_threads(static_cast<int>(config.search_threads));
    const auto start = Clock::now();
#pragma omp parallel for schedule(dynamic) default(none) \
    shared(config, error_message, error_mutex, failed, index, queries, search_parameters)
    for (int64_t search_id = 0; search_id < static_cast<int64_t>(config.search_count);
         ++search_id) {
        if (failed.load(std::memory_order_relaxed)) {
            continue;
        }
        const auto& query = queries[static_cast<uint64_t>(search_id) % queries.size()];
        auto result = index->KnnSearch(
            query.query, static_cast<int64_t>(config.topk), search_parameters, query.filter);
        if (not result.has_value()) {
            failed.store(true, std::memory_order_relaxed);
            std::scoped_lock lock(error_mutex);
            error_message = result.error().message;
        }
    }
    const auto end = Clock::now();
    if (failed.load(std::memory_order_relaxed)) {
        throw std::runtime_error("timed search failed: " + error_message);
    }
    const auto wall_seconds = std::chrono::duration<double>(end - start).count();
    return curve_point{static_cast<double>(config.search_count) / wall_seconds,
                       recall_sum / static_cast<double>(queries.size()),
                       static_cast<double>(mci_routes) / static_cast<double>(queries.size()),
                       wall_seconds,
                       static_cast<double>(raw_float_routes) / static_cast<double>(queries.size()),
                       dist_cmp_sum / static_cast<double>(queries.size()),
                       hops_sum / static_cast<double>(queries.size()),
                       seeds_sum / static_cast<double>(queries.size())};
}

stage_stats
read_stage_stats(const vsag::IndexPtr& index) {
    const auto stats = nlohmann::json::parse(index->GetStats());
    auto get = [&](const char* key) -> uint64_t {
        return stats.contains(key) ? stats[key].get<uint64_t>() : 0;
    };
    return stage_stats{get("mci_total_nodes"),
                       get("mci_covered_nodes"),
                       get("mci_total_clique_count"),
                       get("mci_delta_clique_count"),
                       get("mci_retired_clique_count"),
                       get("mci_inactive_node_count"),
                       get("mci_memory_usage"),
                       get("mci_total_membership_count"),
                       get("mci_delta_extra_membership_count"),
                       stats.value("mci_avg_clique_size", 0.0),
                       stats.value("mci_avg_membership_per_node", 0.0)};
}

void
write_csv_header(std::ofstream& output) {
    output << "stage,active_vectors,index_elements,ef_search,qps,recall_at_k,mci_route_ratio,"
              "query_count,search_count,search_threads,build_seconds,mutation_seconds,"
              "index_memory_bytes,mci_memory_bytes,mci_total_nodes,mci_covered_nodes,"
              "mci_total_cliques,mci_delta_cliques,mci_retired_cliques,inactive_nodes,"
              "mci_raw_float_ratio,flush_seconds,remove_mode,vector_memory_bytes,"
              "graph_memory_bytes,mci_total_memberships,mci_delta_extra_memberships,"
              "mci_avg_clique_size,mci_avg_memberships_per_node,avg_dist_cmp,avg_hops,"
              "avg_seed_count,incremental_join_ratio,incremental_added_mct,"
              "incremental_clique_max,delete_clique_size_threshold,delete_node_mct_threshold\n";
}

void
benchmark_stage(const std::string& stage,
                const vsag::IndexPtr& index,
                const vsag::eval::EvalDatasetPtr& dataset,
                const std::vector<uint64_t>& query_ids,
                const std::vector<std::vector<int64_t>>& ground_truth,
                const std::vector<uint64_t>& active_label_counts,
                const std::vector<uint8_t>& active_ids,
                uint64_t active_count,
                const std::vector<uint64_t>& ef_search_values,
                const benchmark_config& config,
                double build_seconds,
                double mutation_seconds,
                std::ofstream& output) {
    double flush_seconds = 0.0;
    if (config.flush_after_mutation and stage != "initial_100pct") {
        const auto begin = Clock::now();
        const auto flushed = index->Flush();
        if (not flushed.has_value()) {
            throw std::runtime_error("flush failed: " + flushed.error().message);
        }
        flush_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    }
    const auto full_base_count = static_cast<uint64_t>(dataset->GetNumberOfBase());
    const auto filter_base_count = config.max_base == 0
                                       ? full_base_count
                                       : std::min<uint64_t>(config.max_base, full_base_count);
    const auto queries = make_query_cases(dataset,
                                          filter_base_count,
                                          query_ids,
                                          ground_truth,
                                          active_label_counts,
                                          active_ids,
                                          active_count);
    const auto stats = read_stage_stats(index);
    const auto index_elements = static_cast<uint64_t>(index->GetNumElements());
    const auto index_memory = static_cast<uint64_t>(index->GetMemoryUsage());
    const auto memory_detail = index->GetMemoryUsageDetail();
    std::cout << "[stage] " << stage << " active=" << active_count
              << " index_elements=" << index_elements
              << " inactive_nodes=" << stats.mci_inactive_node_count
              << " retired_cliques=" << stats.mci_retired_clique_count << std::endl;
    for (auto ef_search : ef_search_values) {
        const auto curve =
            measure_curve_point(index, queries, make_search_parameters(config, ef_search), config);
        std::cout << "  ef=" << std::setw(4) << ef_search << " recall=" << std::fixed
                  << std::setprecision(4) << curve.recall << " qps=" << std::setprecision(2)
                  << curve.qps << " mci_route=" << std::setprecision(4) << curve.mci_route_ratio
                  << " raw_float=" << curve.raw_float_ratio << std::endl;
        output << stage << ',' << active_count << ',' << index_elements << ',' << ef_search << ','
               << std::setprecision(10) << curve.qps << ',' << curve.recall << ','
               << curve.mci_route_ratio << ',' << queries.size() << ',' << config.search_count
               << ',' << config.search_threads << ',' << build_seconds << ',' << mutation_seconds
               << ',' << index_memory << ',' << stats.mci_memory_usage << ','
               << stats.mci_total_nodes << ',' << stats.mci_covered_nodes << ','
               << stats.mci_total_clique_count << ',' << stats.mci_delta_clique_count << ','
               << stats.mci_retired_clique_count << ',' << stats.mci_inactive_node_count << ','
               << curve.raw_float_ratio << ',' << flush_seconds << ','
               << (config.force_remove ? "force" : "mark") << ','
               << memory_detail.at("basic_flatten_codes") << ','
               << memory_detail.at("bottom_graph") + memory_detail.at("route_graph") << ','
               << stats.total_memberships << ',' << stats.delta_extra_memberships << ','
               << stats.avg_clique_size << ',' << stats.avg_memberships << ',' << curve.avg_dist_cmp
               << ',' << curve.avg_hops << ',' << curve.avg_seeds << ','
               << config.mci_incremental_join_ratio_threshold << ','
               << config.mci_incremental_added_mct << ',' << config.mci_incremental_clique_max
               << ',' << config.mci_delete_clique_size_threshold << ','
               << config.mci_delete_node_mct_threshold << '\n';
        output.flush();
    }
}

double
remove_vectors(const vsag::IndexPtr& index,
               const std::vector<int64_t>& ids,
               uint64_t begin,
               uint64_t count,
               uint64_t batch_size,
               bool force_remove) {
    const auto start = Clock::now();
    uint64_t processed = 0;
    while (processed < count) {
        const auto current = std::min<uint64_t>(batch_size, count - processed);
        std::vector<int64_t> batch(ids.begin() + static_cast<int64_t>(begin + processed),
                                   ids.begin() + static_cast<int64_t>(begin + processed + current));
        const auto mode =
            force_remove ? vsag::RemoveMode::FORCE_REMOVE : vsag::RemoveMode::MARK_REMOVE;
        const auto removed = take_expected(index->Remove(batch, mode), "remove");
        if (removed != current) {
            throw std::runtime_error("mark remove count mismatch");
        }
        processed += current;
        std::cout << "\r[remove] " << processed << '/' << count << std::flush;
    }
    std::cout << std::endl;
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double
add_vectors(const vsag::IndexPtr& index,
            const vsag::eval::EvalDatasetPtr& dataset,
            const std::vector<int64_t>& ids,
            uint64_t begin,
            uint64_t count,
            uint64_t batch_size) {
    const auto start = Clock::now();
    const auto dim = static_cast<uint64_t>(dataset->GetDim());
    const auto* train = static_cast<const float*>(dataset->GetTrain());
    uint64_t processed = 0;
    while (processed < count) {
        const auto current = std::min<uint64_t>(batch_size, count - processed);
        std::vector<int64_t> batch_ids(
            ids.begin() + static_cast<int64_t>(begin + processed),
            ids.begin() + static_cast<int64_t>(begin + processed + current));
        std::vector<float> batch_vectors(current * dim);
        for (uint64_t row = 0; row < current; ++row) {
            const auto source_id = static_cast<uint64_t>(batch_ids[row]);
            std::memcpy(
                batch_vectors.data() + row * dim, train + source_id * dim, dim * sizeof(float));
        }
        auto base = vsag::Dataset::Make();
        base->NumElements(static_cast<int64_t>(current))
            ->Dim(static_cast<int64_t>(dim))
            ->Ids(batch_ids.data())
            ->Float32Vectors(batch_vectors.data())
            ->Owner(false);
        const auto failed_ids = take_expected(index->Add(base), "add back");
        if (not failed_ids.empty()) {
            throw std::runtime_error("add back returned failed ids");
        }
        processed += current;
        std::cout << "\r[add] " << processed << '/' << count << std::flush;
    }
    std::cout << std::endl;
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void
update_active_state(std::vector<uint64_t>& counts,
                    std::vector<uint8_t>& active_ids,
                    const int64_t* train_labels,
                    const std::vector<int64_t>& ids,
                    uint64_t begin,
                    uint64_t count,
                    bool add) {
    for (uint64_t offset = 0; offset < count; ++offset) {
        const auto id = static_cast<uint64_t>(ids[begin + offset]);
        const auto label = static_cast<uint64_t>(train_labels[id]);
        if (add) {
            ++counts[label];
            active_ids[id] = 1;
        } else {
            --counts[label];
            active_ids[id] = 0;
        }
    }
}

// Cache complete exact filtered rankings once. Selecting their first live IDs at each
// checkpoint is exact for that checkpoint, including newly added and deleted true neighbors.
std::vector<std::vector<int64_t>>
stress_rankings(const vsag::eval::EvalDatasetPtr& dataset,
                uint64_t pool_count,
                uint64_t query_count,
                uint64_t threads) {
    const auto train_labels = dataset->GetTrainLabels();
    const auto test_labels = dataset->GetTestLabels();
    if (train_labels == nullptr or test_labels == nullptr) {
        throw std::invalid_argument("stress requires train/test labels");
    }
    std::vector<std::vector<int64_t>> rankings(query_count);
    omp_set_num_threads(static_cast<int>(threads));
#pragma omp parallel for schedule(dynamic)
    for (int64_t q = 0; q < static_cast<int64_t>(query_count); ++q) {
        std::vector<std::pair<float, int64_t>> candidates;
        auto distance = dataset->GetDistanceFunc();
        auto dim = static_cast<uint64_t>(dataset->GetDim());
        for (uint64_t id = 0; id < pool_count; ++id) {
            if (train_labels[id] == test_labels[q]) {
                candidates.emplace_back(distance(dataset->GetOneTest(q),
                                                 dataset->GetOneTrain(static_cast<int64_t>(id)),
                                                 &dim),
                                        static_cast<int64_t>(id));
            }
        }
        std::sort(candidates.begin(), candidates.end());
        rankings[q].reserve(candidates.size());
        for (const auto& candidate : candidates) {
            rankings[q].push_back(candidate.second);
        }
    }
    return rankings;
}

void
run_stress(const benchmark_config& config,
           const vsag::eval::EvalDatasetPtr& dataset,
           uint64_t pool_count,
           const std::vector<uint64_t>& efs) {
    const auto initial_count = config.stress_initial_count;
    const auto step_count = config.stress_step_count == 0
                                ? static_cast<uint64_t>(std::llround(pool_count / 14.0))
                                : config.stress_step_count;
    if (initial_count == 0 or initial_count > pool_count or step_count == 0 or
        step_count > pool_count) {
        throw std::invalid_argument("stress initial/step count is outside the vector pool");
    }
    if (config.stress_mode == "alternate" and step_count > pool_count - initial_count) {
        throw std::invalid_argument("not enough inactive vectors for alternate ADD");
    }
    const auto train_labels = dataset->GetTrainLabels();
    if (train_labels == nullptr) {
        throw std::invalid_argument("stress requires train labels");
    }
    for (uint64_t id = 0; id < pool_count; ++id) {
        if (train_labels[id] < 0) {
            throw std::invalid_argument("stress requires nonnegative train labels");
        }
    }
    const auto query_count =
        config.query_count == 0
            ? static_cast<uint64_t>(dataset->GetNumberOfQuery())
            : std::min<uint64_t>(config.query_count, dataset->GetNumberOfQuery());
    if (query_count == 0) {
        throw std::invalid_argument("stress requires at least one query");
    }
    std::vector<uint64_t> query_ids(query_count);
    std::iota(query_ids.begin(), query_ids.end(), 0);
    std::cout << "[stress] pool=" << pool_count << " initial=" << initial_count
              << " step=" << step_count << " rounds=" << config.stress_rounds
              << " mode=" << config.stress_mode << " seed=" << config.random_seed << std::endl;
    const auto truth_start = Clock::now();
    const auto rankings = stress_rankings(dataset, pool_count, query_count, config.build_threads);
    const auto truth_seconds = std::chrono::duration<double>(Clock::now() - truth_start).count();
    std::cout << "[ground-truth] exact full-pool rankings ready in " << truth_seconds << " seconds"
              << std::endl;

    const auto parent = std::filesystem::path(config.output_path).parent_path();
    if (not parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream output(config.output_path);
    std::ofstream events(config.output_path + ".events.csv");
    std::ofstream audit(config.output_path + ".ids.csv");
    std::ofstream truth_audit(config.output_path + ".truth.csv");
    if (not output or not events or not audit or not truth_audit) {
        throw std::runtime_error("cannot open stress output files");
    }
    write_csv_header(output);
    events << "stage,round,operation,count,active_vectors,mutation_seconds,truth_select_seconds,"
              "truth_prepare_seconds\n";
    audit << "stage,operation,id\n";
    truth_audit << "stage,query_id,rank,id\n";
    std::mt19937_64 random(config.random_seed);
    std::vector<int64_t> population(pool_count);
    std::iota(population.begin(), population.end(), 0);
    std::shuffle(population.begin(), population.end(), random);
    std::vector<int64_t> initial_ids(population.begin(), population.begin() + initial_count);
    std::vector<uint8_t> active_ids(pool_count, 0);
    auto counts = make_active_label_counts(train_labels.get(), pool_count);
    std::fill(counts.begin(), counts.end(), 0);
    update_active_state(
        counts, active_ids, train_labels.get(), initial_ids, 0, initial_count, true);
    const auto dim = static_cast<uint64_t>(dataset->GetDim());
    auto index =
        take_expected(vsag::Factory::CreateIndex("hgraph", make_build_parameters(config, dim)),
                      "create stress index");
    const auto build_start = Clock::now();
    {
        std::vector<float> vectors(initial_count * dim);
        for (uint64_t row = 0; row < initial_count; ++row) {
            std::memcpy(vectors.data() + row * dim,
                        dataset->GetOneTrain(initial_ids[row]),
                        dim * sizeof(float));
            audit << "initial_100pct,build," << initial_ids[row] << '\n';
        }
        auto base = vsag::Dataset::Make();
        base->NumElements(initial_count)
            ->Dim(dim)
            ->Ids(initial_ids.data())
            ->Float32Vectors(vectors.data())
            ->Owner(false);
        std::cout << "[build] random initial subset: " << initial_count << std::endl;
        if (not take_expected(index->Build(base), "stress build").empty()) {
            throw std::runtime_error("stress build returned failed IDs");
        }
    }
    const auto build_seconds = std::chrono::duration<double>(Clock::now() - build_start).count();
    uint64_t live = initial_count;
    auto checkpoint = [&](const std::string& stage,
                          uint64_t round,
                          const std::string& operation,
                          uint64_t count,
                          double seconds) {
        const auto begin = Clock::now();
        std::vector<std::vector<int64_t>> truth(query_count);
        for (uint64_t q = 0; q < query_count; ++q) {
            for (const auto id : rankings[q]) {
                if (active_ids[id] != 0) {
                    truth[q].push_back(id);
                    if (truth[q].size() == config.topk) {
                        break;
                    }
                }
            }
            // Never silently change the query cohort or divide by an empty truth set.
            if (truth[q].size() != config.topk) {
                throw std::runtime_error("a fixed stress query has fewer than topk live matches");
            }
            for (uint64_t rank = 0; rank < truth[q].size(); ++rank) {
                truth_audit << stage << ',' << query_ids[q] << ',' << rank << ',' << truth[q][rank]
                            << '\n';
            }
        }
        const auto select_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
        benchmark_stage(stage,
                        index,
                        dataset,
                        query_ids,
                        truth,
                        counts,
                        active_ids,
                        live,
                        efs,
                        config,
                        build_seconds,
                        seconds,
                        output);
        const auto stats = read_stage_stats(index);
        if (index->GetNumElements() != static_cast<int64_t>(live) or
            stats.mci_covered_nodes != live or
            (config.force_remove and
             (stats.mci_total_nodes != live or stats.mci_inactive_node_count != 0))) {
            throw std::runtime_error("stress live count, physical count, or MCI coverage mismatch");
        }
        events << stage << ',' << round << ',' << operation << ',' << count << ',' << live << ','
               << std::setprecision(10) << seconds << ',' << select_seconds << ',' << truth_seconds
               << '\n';
        events.flush();
        audit.flush();
        truth_audit.flush();
    };
    checkpoint("initial_100pct", 0, "build", initial_count, 0.0);
    for (uint64_t round = 1; round <= config.stress_rounds; ++round) {
        std::vector<int64_t> candidates;
        for (uint64_t id = 0; id < pool_count; ++id) {
            if (config.stress_mode == "toggle" or (active_ids[id] != 0) == (round % 2 == 0)) {
                candidates.push_back(static_cast<int64_t>(id));
            }
        }
        if (candidates.size() < step_count) {
            throw std::runtime_error("stress candidate pool is smaller than step count");
        }
        std::shuffle(candidates.begin(), candidates.end(), random);
        std::vector<int64_t> adds;
        std::vector<int64_t> removes;
        for (uint64_t i = 0; i < step_count; ++i) {
            (active_ids[candidates[i]] != 0 ? removes : adds).push_back(candidates[i]);
        }
        auto mutate = [&](const std::vector<int64_t>& ids, bool add) {
            if (ids.empty()) {
                return;
            }
            const auto operation = add ? "add" : "delete";
            const auto stage = "round_" + std::to_string(round) + "_" + operation;
            for (const auto id : ids) {
                audit << stage << ',' << operation << ',' << id << '\n';
            }
            audit.flush();
            const auto seconds =
                add ? add_vectors(index, dataset, ids, 0, ids.size(), config.mutation_batch_size)
                    : remove_vectors(index,
                                     ids,
                                     0,
                                     ids.size(),
                                     config.mutation_batch_size,
                                     config.force_remove);
            update_active_state(counts, active_ids, train_labels.get(), ids, 0, ids.size(), add);
            live = add ? live + ids.size() : live - ids.size();
            checkpoint(stage, round, operation, ids.size(), seconds);
        };
        // A round samples without replacement, then deletes selected live IDs and adds
        // selected absent IDs. Checkpoint both operations to expose memory troughs.
        mutate(removes, false);
        mutate(adds, true);
    }
    std::cout << "[done] stress CSV=" << config.output_path << std::endl;
}

}  // namespace

int
main(int argc, char** argv) {
    try {
        const auto config = parse_arguments(argc, argv);
        const auto ef_search_values = parse_ef_search_values(config.ef_search_values);
        vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
        vsag::Options::Instance().set_num_threads_building(
            static_cast<int32_t>(config.build_threads));

        std::cout << "[load] " << config.dataset_path << std::endl;
        const auto dataset = vsag::eval::EvalDataset::Load(config.dataset_path);
        if (dataset->GetVectorType() != vsag::eval::DENSE_VECTORS or
            dataset->GetTrainDataType() != vsag::DATATYPE_FLOAT32 or
            dataset->GetTestDataType() != vsag::DATATYPE_FLOAT32) {
            throw std::invalid_argument("benchmark requires a dense float32 dataset");
        }
        const auto full_base_count = static_cast<uint64_t>(dataset->GetNumberOfBase());
        const auto base_count = config.max_base == 0
                                    ? full_base_count
                                    : std::min<uint64_t>(config.max_base, full_base_count);
        const auto dim = static_cast<uint64_t>(dataset->GetDim());
        if (base_count == 0) {
            throw std::invalid_argument("dataset contains no base vectors");
        }
        std::cout << "[load] base=" << base_count << '/' << full_base_count
                  << " queries=" << dataset->GetNumberOfQuery() << " dim=" << dim << std::endl;

        if (config.stress_rounds != 0) {
            run_stress(config, dataset, base_count, ef_search_values);
            return 0;
        }

        std::vector<uint64_t> query_ids;
        const auto ground_truth =
            prepare_ground_truth(dataset, base_count, config.query_count, config.topk, query_ids);
        const auto delete_count = static_cast<uint64_t>(
            std::llround(static_cast<double>(base_count) * config.delete_fraction));
        if (delete_count == 0) {
            throw std::invalid_argument("delete fraction selects zero vectors");
        }
        const auto delete_order =
            make_delete_order(base_count, ground_truth, delete_count, config.random_seed);
        const auto train_labels = dataset->GetTrainLabels();
        auto active_label_counts = make_active_label_counts(train_labels.get(), base_count);
        std::vector<uint8_t> active_ids(base_count, 1);

        std::vector<int64_t> build_ids(base_count);
        std::iota(build_ids.begin(), build_ids.end(), 0);
        auto base = vsag::Dataset::Make();
        base->NumElements(static_cast<int64_t>(base_count))
            ->Dim(static_cast<int64_t>(dim))
            ->Ids(build_ids.data())
            ->Float32Vectors(static_cast<const float*>(dataset->GetTrain()))
            ->Owner(false);
        const auto build_parameters = make_build_parameters(config, dim);
        auto index = take_expected(vsag::Factory::CreateIndex("hgraph", build_parameters),
                                   "create HGraph+MCI");
        const nlohmann::json identity{
            {"schema", 1},
            {"format", "vsag_ostream"},
            {"state", "initial_100pct"},
            {"dataset", std::filesystem::canonical(config.dataset_path).string()},
            {"dataset_bytes", std::filesystem::file_size(config.dataset_path)},
            {"dataset_mtime",
             std::filesystem::last_write_time(config.dataset_path).time_since_epoch().count()},
            {"base_count", base_count},
            {"build_parameters", nlohmann::json::parse(build_parameters)}};
        if (not config.save_initial_index.empty()) {
            for (const auto& suffix : {"", ".partial", ".json"}) {
                if (std::filesystem::exists(config.save_initial_index + suffix)) {
                    throw std::runtime_error("refusing to overwrite initial index: " +
                                             config.save_initial_index + suffix);
                }
            }
        }
        std::cout << "[build] HGraph/NSW+MCI parameters=" << build_parameters << std::endl;
        const auto build_start = Clock::now();
        const auto failed_ids = take_expected(index->Build(base), "full build");
        if (not failed_ids.empty()) {
            throw std::runtime_error("full build returned failed ids");
        }
        const auto build_seconds =
            std::chrono::duration<double>(Clock::now() - build_start).count();
        std::cout << "[build] completed in " << build_seconds << " seconds" << std::endl;
        if (not config.save_initial_index.empty()) {
            const auto path = std::filesystem::path(config.save_initial_index);
            if (not path.parent_path().empty()) {
                std::filesystem::create_directories(path.parent_path());
            }
            std::ofstream saved(config.save_initial_index + ".partial", std::ios::binary);
            take_expected(index->Serialize(saved), "save initial index");
            saved.close();
            if (not saved) {
                throw std::runtime_error("failed to write initial index");
            }
            std::filesystem::rename(config.save_initial_index + ".partial", path);
            std::ofstream metadata(config.save_initial_index + ".json");
            metadata << identity.dump(2) << '\n';
            metadata.close();
            if (not metadata) {
                throw std::runtime_error("failed to write initial index metadata");
            }
            std::cout << "[index-save] " << config.save_initial_index << std::endl;
        }

        const auto output_parent = std::filesystem::path(config.output_path).parent_path();
        if (not output_parent.empty()) {
            std::filesystem::create_directories(output_parent);
        }
        std::ofstream output(config.output_path, std::ios::trunc);
        if (not output) {
            throw std::runtime_error("cannot open output CSV: " + config.output_path);
        }
        write_csv_header(output);

        uint64_t active_count = base_count;
        benchmark_stage("initial_100pct",
                        index,
                        dataset,
                        query_ids,
                        ground_truth,
                        active_label_counts,
                        active_ids,
                        active_count,
                        ef_search_values,
                        config,
                        build_seconds,
                        0.0,
                        output);

        auto mutation_seconds = remove_vectors(
            index, delete_order, 0, delete_count, config.mutation_batch_size, config.force_remove);
        update_active_state(active_label_counts,
                            active_ids,
                            train_labels.get(),
                            delete_order,
                            0,
                            delete_count,
                            false);
        active_count -= delete_count;
        benchmark_stage("delete_10pct",
                        index,
                        dataset,
                        query_ids,
                        ground_truth,
                        active_label_counts,
                        active_ids,
                        active_count,
                        ef_search_values,
                        config,
                        build_seconds,
                        mutation_seconds,
                        output);

        mutation_seconds = remove_vectors(index,
                                          delete_order,
                                          delete_count,
                                          delete_count,
                                          config.mutation_batch_size,
                                          config.force_remove);
        update_active_state(active_label_counts,
                            active_ids,
                            train_labels.get(),
                            delete_order,
                            delete_count,
                            delete_count,
                            false);
        active_count -= delete_count;
        benchmark_stage("delete_20pct",
                        index,
                        dataset,
                        query_ids,
                        ground_truth,
                        active_label_counts,
                        active_ids,
                        active_count,
                        ef_search_values,
                        config,
                        build_seconds,
                        mutation_seconds,
                        output);

        mutation_seconds =
            add_vectors(index, dataset, delete_order, 0, delete_count, config.mutation_batch_size);
        update_active_state(active_label_counts,
                            active_ids,
                            train_labels.get(),
                            delete_order,
                            0,
                            delete_count,
                            true);
        active_count += delete_count;
        benchmark_stage("add_back_10pct",
                        index,
                        dataset,
                        query_ids,
                        ground_truth,
                        active_label_counts,
                        active_ids,
                        active_count,
                        ef_search_values,
                        config,
                        build_seconds,
                        mutation_seconds,
                        output);

        mutation_seconds = add_vectors(
            index, dataset, delete_order, delete_count, delete_count, config.mutation_batch_size);
        update_active_state(active_label_counts,
                            active_ids,
                            train_labels.get(),
                            delete_order,
                            delete_count,
                            delete_count,
                            true);
        active_count += delete_count;
        benchmark_stage("add_back_20pct",
                        index,
                        dataset,
                        query_ids,
                        ground_truth,
                        active_label_counts,
                        active_ids,
                        active_count,
                        ef_search_values,
                        config,
                        build_seconds,
                        mutation_seconds,
                        output);
        std::cout << "[done] CSV=" << config.output_path << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mci_mutation_benchmark: " << error.what() << std::endl;
        return 1;
    }
}
