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

// Standalone experiment: compare a full-build MCI index against one that is
// built on a fraction of the base set and then grown with incremental Add().
// Both are evaluated with the same filtered KNN search and recall is computed
// against the dataset ground truth (matching tools/eval recall methodology).

#include <fmt/format.h>
#include <omp.h>
#include <vsag/vsag.h>

#include <algorithm>
#include <argparse/argparse.hpp>
#include <atomic>
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
#include "typing.h"

namespace {

using vsag::eval::EvalDataset;
using vsag::eval::EvalDatasetPtr;
using Clock = std::chrono::steady_clock;

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

// Filter that mirrors tools/eval/case/search_eval_case.cpp FilterObj: a base id
// is valid iff its train label equals the query's test label.
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

struct SearchOutcome {
    double recall_avg{0.0};
    double qps{0.0};
    double total_search_seconds{0.0};
    uint64_t query_count{0};
};

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

    std::vector<float> distances;
    distances.reserve(result_count);
    if (neighbors != nullptr) {
        for (uint64_t i = 0; i < result_count; ++i) {
            if (neighbors[i] < 0 or neighbors[i] >= base_count) {
                continue;
            }
            distances.push_back(
                distance_func(query_vector, dataset->GetOneTrain(neighbors[i]), &dim));
        }
    }
    size_t count = 0;
    for (float d : distances) {
        if (d <= threshold) {
            ++count;
        }
    }
    return static_cast<double>(count) / static_cast<double>(gt_distances.size());
}

vsag::DatasetPtr
make_slice(const EvalDatasetPtr& dataset,
           const std::vector<int64_t>& ids,
           int64_t begin,
           int64_t count) {
    const auto dim = dataset->GetDim();
    const auto* train = static_cast<const float*>(dataset->GetTrain());
    auto slice = vsag::Dataset::Make();
    slice->NumElements(count)
        ->Dim(dim)
        ->Ids(ids.data() + begin)
        ->Float32Vectors(train + begin * dim)
        ->Owner(false);
    return slice;
}

vsag::IndexPtr
create_index(const std::string& create_params) {
    auto created = vsag::Factory::CreateIndex("mci", create_params);
    check(created.has_value(),
          fmt::format("failed to create mci index: {}", created.error().message));
    return created.value();
}

void
save_index_to_file(const vsag::IndexPtr& index, const std::string& path) {
    std::filesystem::path dir_path = std::filesystem::path(path).parent_path();
    if (not dir_path.empty() and not std::filesystem::exists(dir_path)) {
        std::filesystem::create_directories(dir_path);
    }
    std::ofstream outfile(path, std::ios::binary);
    check(outfile.is_open(), fmt::format("failed to open file for writing: {}", path));
    auto result = index->Serialize(outfile);
    check(result.has_value(), fmt::format("failed to serialize index: {}", result.error().message));
    outfile.close();
    std::cout << fmt::format("[save] index saved to {} ({} bytes)\n",
                             path,
                             std::filesystem::file_size(path));
}

vsag::IndexPtr
load_index_from_file(const std::string& create_params, const std::string& path) {
    check(std::filesystem::exists(path), fmt::format("index file not found: {}", path));
    auto index = create_index(create_params);
    std::ifstream infile(path, std::ios::binary);
    check(infile.is_open(), fmt::format("failed to open file for reading: {}", path));
    auto result = index->Deserialize(infile);
    check(result.has_value(),
          fmt::format("failed to deserialize index: {}", result.error().message));
    infile.close();
    std::cout << fmt::format(
        "[load] index loaded from {} (num_elements={})\n", path, index->GetNumElements());
    return index;
}

// Build an index either fully or as build(prefix) + incremental Add(rest),
// reporting timing along the way (so a prohibitively slow Add path is visible
// early instead of after the whole run).
vsag::IndexPtr
build_index(const EvalDatasetPtr& dataset,
            const std::vector<int64_t>& ids,
            const std::string& create_params,
            int64_t total_base,
            double build_ratio,
            int64_t add_batch,
            uint64_t add_limit,
            const std::string& tag) {
    auto index = create_index(create_params);

    int64_t prefix = static_cast<int64_t>(static_cast<double>(total_base) * build_ratio);
    prefix = std::max<int64_t>(1, std::min<int64_t>(prefix, total_base));

    std::cout << fmt::format(
        "[{}] build prefix={} (ratio={:.2f}) of total={}\n", tag, prefix, build_ratio, total_base);
    auto build_start = Clock::now();
    auto build_result = index->Build(make_slice(dataset, ids, 0, prefix));
    check(build_result.has_value(),
          fmt::format("[{}] build failed: {}", tag, build_result.error().message));
    std::cout << fmt::format("[{}] build done in {:.2f}s, num_elements={}\n",
                             tag,
                             seconds_since(build_start),
                             index->GetNumElements());

    int64_t add_end = total_base;
    if (add_limit > 0) {
        add_end = std::min<int64_t>(total_base, prefix + static_cast<int64_t>(add_limit));
    }
    const int64_t to_add = add_end - prefix;
    if (to_add <= 0) {
        return index;
    }

    std::cout << fmt::format(
        "[{}] incremental add of {} vectors in batches of {}\n", tag, to_add, add_batch);
    auto add_start = Clock::now();
    int64_t added = 0;
    auto last_report = Clock::now();
    for (int64_t begin = prefix; begin < add_end; begin += add_batch) {
        const int64_t count = std::min<int64_t>(add_batch, add_end - begin);
        auto add_result = index->Add(make_slice(dataset, ids, begin, count));
        check(add_result.has_value(),
              fmt::format("[{}] add failed at {}: {}", tag, begin, add_result.error().message));
        added += count;
        if (seconds_since(last_report) >= 5.0) {
            const double elapsed = seconds_since(add_start);
            const double rate = added / std::max(elapsed, 1e-9);
            const double eta = (to_add - added) / std::max(rate, 1e-9);
            std::cout << fmt::format(
                "[{}] added {}/{} ({:.1f}/s, elapsed={:.1f}s, eta={:.1f}s, num_elements={})\n",
                tag,
                added,
                to_add,
                rate,
                elapsed,
                eta,
                index->GetNumElements());
            std::cout.flush();
            last_report = Clock::now();
        }
    }
    std::cout << fmt::format("[{}] add done in {:.2f}s, final num_elements={}\n",
                             tag,
                             seconds_since(add_start),
                             index->GetNumElements());
    return index;
}

SearchOutcome
run_filter_search(const vsag::IndexPtr& index,
                  const EvalDatasetPtr& dataset,
                  const std::string& search_params,
                  uint64_t topk,
                  uint64_t query_count_cap,
                  int num_threads,
                  const std::string& tag) {
    const auto query_count = static_cast<uint64_t>(dataset->GetNumberOfQuery());
    const uint64_t total_queries =
        query_count_cap == 0 ? query_count : std::min<uint64_t>(query_count_cap, query_count);
    const auto* train_labels = dataset->GetTrainLabels().get();
    const auto* test_labels = dataset->GetTestLabels().get();
    check(train_labels != nullptr and test_labels != nullptr,
          "dataset must contain train_labels and test_labels for filtered search");

    // Precompute valid id lists per label (like SearchEvalCase::init_filter_valid_ids).
    std::unordered_map<int64_t, std::vector<int64_t>> valid_ids;
    const auto total_base = dataset->GetNumberOfBase();
    for (int64_t id = 0; id < total_base; ++id) {
        valid_ids[train_labels[id]].push_back(id);
    }

    std::vector<double> recalls(total_queries, 0.0);
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
            recalls[qi] = 0.0;
            continue;
        }
        recalls[qi] = recall_for_query(result.value()->GetIds(),
                                       static_cast<uint64_t>(result.value()->GetDim()),
                                       dataset->GetNeighbors(i),
                                       dataset.get(),
                                       dataset->GetOneTest(i),
                                       topk);
    }
    const double elapsed = seconds_since(search_start);

    SearchOutcome outcome;
    outcome.query_count = total_queries;
    outcome.total_search_seconds = elapsed;
    outcome.qps = static_cast<double>(total_queries) / std::max(elapsed, 1e-9);
    double sum = 0.0;
    for (double r : recalls) {
        sum += r;
    }
    outcome.recall_avg = sum / static_cast<double>(total_queries);
    return outcome;
}

void
parse_args(argparse::ArgumentParser& parser, int argc, char** argv) {
    parser.add_argument("--datapath", "-d").required().help("HDF5 dataset path.");
    parser.add_argument("--create_params", "-c").required().help("MCI create_params JSON string.");
    parser.add_argument("--search_params", "-s").required().help("MCI search_params JSON string.");
    parser.add_argument("--topk", "-k").default_value(20).scan<'i', int>();
    parser.add_argument("--build_ratio")
        .default_value(0.5)
        .scan<'g', double>()
        .help("Fraction of base used for the initial Build() in the incremental variant.");
    parser.add_argument("--add_batch")
        .default_value(1)
        .scan<'i', int>()
        .help("Number of vectors per Add() call.");
    parser.add_argument("--add_limit")
        .default_value(static_cast<uint64_t>(0))
        .scan<'u', uint64_t>()
        .help(
            "Cap on how many vectors to incrementally add (0 = add the whole remaining tail). "
            "Use a small value for a timing probe.");
    parser.add_argument("--base_limit")
        .default_value(static_cast<uint64_t>(0))
        .scan<'u', uint64_t>()
        .help("Cap on total base vectors loaded from /train (0 = use all).");
    parser.add_argument("--query_count")
        .default_value(static_cast<uint64_t>(0))
        .scan<'u', uint64_t>()
        .help("Cap on number of queries (0 = all queries in the dataset).");
    parser.add_argument("--num_threads", "-t").default_value(16).scan<'i', int>();
    parser.add_argument("--save_index")
        .default_value(std::string(""))
        .help("Path to save the built index after build+add completes.");
    parser.add_argument("--load_index")
        .default_value(std::string(""))
        .help("Path to load a previously saved index (skips build+add).");
    parser.add_argument("--run_full")
        .default_value(true)
        .implicit_value(true)
        .help("Run the full-build baseline variant.");
    parser.add_argument("--skip_full")
        .default_value(false)
        .implicit_value(true)
        .help("Skip the full-build baseline (only run the incremental variant).");

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << "\n" << parser;
        throw;
    }
}

}  // namespace

int
main(int argc, char** argv) {
    vsag::init();
    argparse::ArgumentParser parser("mci_incremental_exp");
    parse_args(parser, argc, argv);

    const auto datapath = parser.get<std::string>("--datapath");
    const auto create_params = parser.get<std::string>("--create_params");
    const auto search_params = parser.get<std::string>("--search_params");
    const auto topk = static_cast<uint64_t>(parser.get<int>("--topk"));
    const auto build_ratio = parser.get<double>("--build_ratio");
    const auto add_batch = std::max<int64_t>(1, parser.get<int>("--add_batch"));
    const auto add_limit = parser.get<uint64_t>("--add_limit");
    const auto base_limit = parser.get<uint64_t>("--base_limit");
    const auto query_count_cap = parser.get<uint64_t>("--query_count");
    const auto num_threads = parser.get<int>("--num_threads");
    const auto save_index_path = parser.get<std::string>("--save_index");
    const auto load_index_path = parser.get<std::string>("--load_index");
    const auto skip_full = parser.get<bool>("--skip_full");

    std::cout << "[exp] loading dataset: " << datapath << std::endl;
    auto dataset = EvalDataset::Load(datapath);
    check(dataset->GetVectorType() == vsag::eval::DENSE_VECTORS, "only dense datasets supported");
    check(dataset->GetTrainDataType() == vsag::DATATYPE_FLOAT32, "only float32 train supported");

    int64_t total_base = dataset->GetNumberOfBase();
    if (base_limit > 0) {
        total_base = std::min<int64_t>(total_base, static_cast<int64_t>(base_limit));
    }
    std::vector<int64_t> ids(total_base);
    for (int64_t i = 0; i < total_base; ++i) {
        ids[i] = i;
    }
    std::cout << fmt::format("[exp] dim={} total_base={} (dataset base={}) query_count={}\n",
                             dataset->GetDim(),
                             total_base,
                             dataset->GetNumberOfBase(),
                             dataset->GetNumberOfQuery());

    // Variant B: incremental (build a prefix, then Add the rest).
    vsag::IndexPtr incr_index;
    if (not load_index_path.empty()) {
        incr_index = load_index_from_file(create_params, load_index_path);
    } else {
        incr_index = build_index(
            dataset, ids, create_params, total_base, build_ratio, add_batch, add_limit, "incremental");
        if (not save_index_path.empty()) {
            save_index_to_file(incr_index, save_index_path);
        }
    }
    auto incr_outcome = run_filter_search(
        incr_index, dataset, search_params, topk, query_count_cap, num_threads, "incremental");
    std::cout << fmt::format(
        "[incremental] recall_avg={:.4f} qps={:.1f} queries={} search_time={:.2f}s\n",
        incr_outcome.recall_avg,
        incr_outcome.qps,
        incr_outcome.query_count,
        incr_outcome.total_search_seconds);

    // Variant A: full build (baseline). Built on the same number of base
    // vectors that the incremental variant ended up containing, so recall is
    // comparable. When add_limit caps the tail, match that subset.
    SearchOutcome full_outcome;
    bool ran_full = false;
    if (not skip_full) {
        int64_t prefix = static_cast<int64_t>(static_cast<double>(total_base) * build_ratio);
        prefix = std::max<int64_t>(1, std::min<int64_t>(prefix, total_base));
        int64_t full_count = total_base;
        if (add_limit > 0) {
            full_count = std::min<int64_t>(total_base, prefix + static_cast<int64_t>(add_limit));
        }
        auto full_index = create_index(create_params);
        std::cout << fmt::format("[full] build all {} vectors at once\n", full_count);
        auto build_start = Clock::now();
        auto build_result = full_index->Build(make_slice(dataset, ids, 0, full_count));
        check(build_result.has_value(),
              fmt::format("[full] build failed: {}", build_result.error().message));
        std::cout << fmt::format("[full] build done in {:.2f}s, num_elements={}\n",
                                 seconds_since(build_start),
                                 full_index->GetNumElements());
        full_outcome = run_filter_search(
            full_index, dataset, search_params, topk, query_count_cap, num_threads, "full");
        ran_full = true;
        std::cout << fmt::format(
            "[full] recall_avg={:.4f} qps={:.1f} queries={} search_time={:.2f}s\n",
            full_outcome.recall_avg,
            full_outcome.qps,
            full_outcome.query_count,
            full_outcome.total_search_seconds);
    }

    std::cout << "\n==================== SUMMARY ====================\n";
    std::cout << fmt::format("dataset={} total_base={} topk={} build_ratio={:.2f} add_batch={}\n",
                             datapath,
                             total_base,
                             topk,
                             build_ratio,
                             add_batch);
    std::cout << fmt::format("{:<14} {:>12} {:>12}\n", "variant", "recall_avg", "qps");
    std::cout << fmt::format(
        "{:<14} {:>12.4f} {:>12.1f}\n", "incremental", incr_outcome.recall_avg, incr_outcome.qps);
    if (ran_full) {
        std::cout << fmt::format(
            "{:<14} {:>12.4f} {:>12.1f}\n", "full", full_outcome.recall_avg, full_outcome.qps);
        std::cout << fmt::format("{:<14} {:>+12.4f} {:>+12.1f}\n",
                                 "delta(incr-full)",
                                 incr_outcome.recall_avg - full_outcome.recall_avg,
                                 incr_outcome.qps - full_outcome.qps);
    }
    std::cout << "================================================\n";
    return 0;
}
