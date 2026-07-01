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

#include <H5Cpp.h>
#include <vsag/filter.h>
#include <vsag/vsag.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

struct Options {
    std::string data_path{"/root/data/wufufilter-5m-128-euclidean.h5"};
    std::string index_path{};
    std::string metric{};
    std::string quantization{"sq8"};
    std::string base_io_type{"block_memory_io"};
    uint64_t limit{0};
    uint64_t query_count{0};
    uint64_t topk{10};
    uint64_t ef_search{100};
    uint64_t thread_count{16};
    uint64_t max_degree{32};
    uint64_t ef_construction{400};
    uint64_t build_thread_count{100};
    uint64_t graph_iter_turn{30};
    uint64_t mci_mcs{200};
    uint64_t mci_clique_max{50};
    float mci_alpha{1.2F};
    uint64_t mci_seed_count{32};
    uint64_t search_mci_seed_count{0};
    float search_mci_seed_ratio{0.0F};
    uint64_t exact_filtered_gt_limit{200000};
    float hgraph_valid_ratio_threshold{1.0F};
    float search_hgraph_valid_ratio_threshold{-1.0F};
    std::string mci_knng_path{};
    bool use_filter{true};
    bool store_raw_vector{true};
    bool search_use_mci{true};
};

class LabelFilter : public vsag::Filter {
public:
    LabelFilter(const std::vector<int64_t>& train_labels,
                int64_t test_label,
                float valid_ratio,
                const std::vector<int64_t>* valid_ids)
        : train_labels_(train_labels),
          test_label_(test_label),
          valid_ratio_(valid_ratio),
          valid_ids_(valid_ids) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        return id >= 0 and static_cast<uint64_t>(id) < train_labels_.size() and
               train_labels_[id] == test_label_;
    }

    [[nodiscard]] float
    ValidRatio() const override {
        return valid_ratio_;
    }

    void
    GetValidIds(const int64_t** valid_ids, int64_t& count) const override {
        if (valid_ids_ == nullptr) {
            *valid_ids = nullptr;
            count = 0;
            return;
        }
        *valid_ids = valid_ids_->data();
        count = static_cast<int64_t>(valid_ids_->size());
    }

private:
    const std::vector<int64_t>& train_labels_;
    int64_t test_label_{0};
    float valid_ratio_{1.0F};
    const std::vector<int64_t>* valid_ids_{nullptr};
};

void
check(bool condition, const std::string& message) {
    if (not condition) {
        throw std::runtime_error(message);
    }
}

template <typename T, typename E>
void
check_expected(const tl::expected<T, E>& result, const std::string& prefix) {
    if (not result.has_value()) {
        throw std::runtime_error(prefix + result.error().message);
    }
}

uint64_t
parse_u64(const std::string& value) {
    return static_cast<uint64_t>(std::stoull(value));
}

float
parse_float(const std::string& value) {
    return std::stof(value);
}

Options
parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need_value = [&](const std::string& name) {
            check(i + 1 < argc, "missing value for " + name);
            return std::string(argv[++i]);
        };
        if (key == "--data") {
            options.data_path = need_value(key);
        } else if (key == "--index") {
            options.index_path = need_value(key);
        } else if (key == "--metric") {
            options.metric = need_value(key);
        } else if (key == "--quantization") {
            options.quantization = need_value(key);
        } else if (key == "--base-io-type") {
            options.base_io_type = need_value(key);
        } else if (key == "--limit") {
            options.limit = parse_u64(need_value(key));
        } else if (key == "--query-count") {
            options.query_count = parse_u64(need_value(key));
        } else if (key == "--topk") {
            options.topk = parse_u64(need_value(key));
        } else if (key == "--ef-search") {
            options.ef_search = parse_u64(need_value(key));
        } else if (key == "--threads") {
            options.thread_count = parse_u64(need_value(key));
        } else if (key == "--max-degree") {
            options.max_degree = parse_u64(need_value(key));
        } else if (key == "--ef-construction") {
            options.ef_construction = parse_u64(need_value(key));
        } else if (key == "--build-thread-count") {
            options.build_thread_count = parse_u64(need_value(key));
        } else if (key == "--graph-iter-turn") {
            options.graph_iter_turn = parse_u64(need_value(key));
        } else if (key == "--mci-mcs") {
            options.mci_mcs = parse_u64(need_value(key));
        } else if (key == "--mci-clique-max") {
            options.mci_clique_max = parse_u64(need_value(key));
        } else if (key == "--mci-alpha") {
            options.mci_alpha = parse_float(need_value(key));
        } else if (key == "--mci-seed-count") {
            options.mci_seed_count = parse_u64(need_value(key));
        } else if (key == "--search-mci-seed-count") {
            options.search_mci_seed_count = parse_u64(need_value(key));
        } else if (key == "--search-mci-seed-ratio") {
            options.search_mci_seed_ratio = parse_float(need_value(key));
        } else if (key == "--exact-filtered-gt-limit") {
            options.exact_filtered_gt_limit = parse_u64(need_value(key));
        } else if (key == "--mci-knng-path") {
            options.mci_knng_path = need_value(key);
        } else if (key == "--hgraph-valid-ratio-threshold") {
            options.hgraph_valid_ratio_threshold = parse_float(need_value(key));
        } else if (key == "--search-hgraph-valid-ratio-threshold") {
            options.search_hgraph_valid_ratio_threshold = parse_float(need_value(key));
        } else if (key == "--no-filter") {
            options.use_filter = false;
        } else if (key == "--no-store-raw-vector") {
            options.store_raw_vector = false;
        } else if (key == "--search-no-mci") {
            options.search_use_mci = false;
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    return options;
}

bool
dataset_exists(const H5::H5File& file, const std::string& path) {
    return H5Lexists(file.getId(), path.c_str(), H5P_DEFAULT) > 0;
}

std::string
infer_metric(const std::string& path, const std::string& override_metric) {
    if (not override_metric.empty()) {
        return override_metric;
    }
    if (path.find("angular") != std::string::npos) {
        return "cosine";
    }
    return "l2";
}

std::string
default_index_path(const std::string& data_path) {
    auto slash = data_path.find_last_of('/');
    auto name = slash == std::string::npos ? data_path : data_path.substr(slash + 1);
    return "/tmp/" + name + ".hgraph_mci.index";
}

std::pair<uint64_t, uint64_t>
get_matrix_shape(const H5::H5File& file, const std::string& dataset_name) {
    auto dataset = file.openDataSet(dataset_name);
    auto dataspace = dataset.getSpace();
    check(dataspace.getSimpleExtentNdims() == 2, dataset_name + " must be a 2-D dataset");
    hsize_t dims[2];
    dataspace.getSimpleExtentDims(dims, nullptr);
    return {static_cast<uint64_t>(dims[0]), static_cast<uint64_t>(dims[1])};
}

std::vector<float>
read_float_matrix_prefix(const H5::H5File& file,
                         const std::string& dataset_name,
                         uint64_t rows,
                         uint64_t dim) {
    auto dataset = file.openDataSet(dataset_name);
    auto dataspace = dataset.getSpace();
    hsize_t offset[2] = {0, 0};
    hsize_t count[2] = {rows, dim};
    dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
    H5::DataSpace memspace(2, count);
    std::vector<float> values(rows * dim);
    dataset.read(values.data(), H5::PredType::NATIVE_FLOAT, memspace, dataspace);
    return values;
}

std::vector<int64_t>
read_i64_prefix(const H5::H5File& file, const std::string& dataset_name, uint64_t rows) {
    auto dataset = file.openDataSet(dataset_name);
    auto dataspace = dataset.getSpace();
    hsize_t offset[1] = {0};
    hsize_t count[1] = {rows};
    dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
    H5::DataSpace memspace(1, count);
    std::vector<int64_t> values(rows);
    dataset.read(values.data(), H5::PredType::NATIVE_INT64, memspace, dataspace);
    return values;
}

std::vector<int64_t>
read_i64_matrix_prefix(const H5::H5File& file,
                       const std::string& dataset_name,
                       uint64_t rows,
                       uint64_t cols) {
    auto dataset = file.openDataSet(dataset_name);
    auto dataspace = dataset.getSpace();
    hsize_t offset[2] = {0, 0};
    hsize_t count[2] = {rows, cols};
    dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
    H5::DataSpace memspace(2, count);
    std::vector<int64_t> values(rows * cols);
    dataset.read(values.data(), H5::PredType::NATIVE_INT64, memspace, dataspace);
    return values;
}

float
distance_for_metric(const float* lhs, const float* rhs, uint64_t dim, const std::string& metric) {
    float dot = 0.0F;
    float lhs_norm = 0.0F;
    float rhs_norm = 0.0F;
    for (uint64_t i = 0; i < dim; ++i) {
        const auto l = lhs[i];
        const auto r = rhs[i];
        dot += l * r;
        lhs_norm += l * l;
        rhs_norm += r * r;
    }
    if (metric == "cosine") {
        if (lhs_norm == 0.0F or rhs_norm == 0.0F) {
            return 1.0F;
        }
        return 1.0F - dot / std::sqrt(lhs_norm * rhs_norm);
    }
    return std::max(0.0F, lhs_norm + rhs_norm - 2.0F * dot);
}

std::vector<int64_t>
build_exact_filtered_ground_truth(const std::vector<float>& train,
                                  const std::vector<float>& queries,
                                  const std::vector<int64_t>& train_labels,
                                  const std::vector<int64_t>& test_labels,
                                  uint64_t query_rows,
                                  uint64_t base_rows,
                                  uint64_t dim,
                                  uint64_t topk,
                                  const std::string& metric) {
    std::vector<int64_t> gt(query_rows * topk, -1);
    for (uint64_t qi = 0; qi < query_rows; ++qi) {
        std::vector<std::pair<float, int64_t>> candidates;
        candidates.reserve(base_rows);
        const auto* query = queries.data() + qi * dim;
        for (uint64_t id = 0; id < base_rows; ++id) {
            if (train_labels[id] != test_labels[qi]) {
                continue;
            }
            candidates.emplace_back(
                distance_for_metric(query, train.data() + id * dim, dim, metric),
                static_cast<int64_t>(id));
        }
        const auto keep = std::min<uint64_t>(topk, candidates.size());
        if (keep == 0) {
            continue;
        }
        std::partial_sort(
            candidates.begin(), candidates.begin() + static_cast<int64_t>(keep), candidates.end());
        for (uint64_t k = 0; k < keep; ++k) {
            gt[qi * topk + k] = candidates[k].second;
        }
    }
    return gt;
}

std::vector<float>
read_valid_ratios(const H5::H5File& file) {
    if (not dataset_exists(file, "/valid_ratios")) {
        return {};
    }
    auto dataset = file.openDataSet("/valid_ratios");
    auto dataspace = dataset.getSpace();
    hsize_t dims[1];
    dataspace.getSimpleExtentDims(dims, nullptr);
    std::vector<float> values(static_cast<uint64_t>(dims[0]));
    dataset.read(values.data(), H5::PredType::NATIVE_FLOAT, dataspace);
    return values;
}

std::string
make_hgraph_mci_params(const Options& options, uint64_t dim, const std::string& metric) {
    std::ostringstream builder;
    builder << R"({
    "dtype": "float32",
    "metric_type": ")"
            << metric << R"(",
    "dim": )"
            << dim << R"(,
    "index_param": {
        "base_quantization_type": ")"
            << options.quantization << R"(",
        "base_io_type": ")"
            << options.base_io_type << R"(",
        "store_raw_vector": )"
            << (options.store_raw_vector ? "true" : "false") << R"(,
        "raw_vector_io_type": "memory_io",
        "graph_type": "odescent",
        "max_degree": )"
            << options.max_degree << R"(,
        "ef_construction": )"
            << options.ef_construction << R"(,
        "build_thread_count": )"
            << options.build_thread_count << R"(,
        "alpha": 1.2,
        "graph_iter_turn": )"
            << options.graph_iter_turn << R"(,
        "neighbor_sample_rate": 0.2,
        "mci": {
            "use_mci": true,
            "mcs": )"
            << options.mci_mcs << R"(,
            "clique_max": )"
            << options.mci_clique_max << R"(,
            "alpha": )"
            << options.mci_alpha << R"(,
            "seed_count": )"
            << options.mci_seed_count << R"(,
            "hgraph_valid_ratio_threshold": )"
            << options.hgraph_valid_ratio_threshold;
    if (not options.mci_knng_path.empty()) {
        builder << R"(,
            "knng_path": ")"
                << options.mci_knng_path << R"(")";
    }
    builder << R"(
        }
    }
})";
    return builder.str();
}

std::string
make_search_params(const Options& options) {
    std::ostringstream builder;
    builder << R"({
    "hgraph": {
        "ef_search": )"
            << options.ef_search << R"(,
        "use_mci": )"
            << (options.search_use_mci ? "true" : "false") << R"(,
        "seed_count": )"
            << (options.search_mci_seed_count == 0 ? options.mci_seed_count
                                                   : options.search_mci_seed_count);
    if (options.search_mci_seed_ratio > 0.0F) {
        builder << R"(,
        "seed_ratio": )"
                << options.search_mci_seed_ratio;
    }
    if (options.search_hgraph_valid_ratio_threshold >= 0.0F) {
        builder << R"(,
        "hgraph_valid_ratio_threshold": )"
                << options.search_hgraph_valid_ratio_threshold;
    }
    builder << R"(
    }
})";
    return builder.str();
}

std::unordered_map<int64_t, std::vector<int64_t>>
build_valid_id_map(const std::vector<int64_t>& train_labels) {
    std::unordered_map<int64_t, std::vector<int64_t>> valid_ids;
    for (uint64_t id = 0; id < train_labels.size(); ++id) {
        valid_ids[train_labels[id]].push_back(static_cast<int64_t>(id));
    }
    return valid_ids;
}

double
recall_for_query(const int64_t* result_ids,
                 uint64_t result_count,
                 const int64_t* gt_ids,
                 uint64_t gt_count) {
    if (result_ids == nullptr or gt_ids == nullptr or gt_count == 0) {
        return 0.0;
    }

    uint64_t valid_gt_count = 0;
    uint64_t hit_count = 0;
    for (uint64_t i = 0; i < gt_count; ++i) {
        if (gt_ids[i] < 0) {
            break;
        }
        ++valid_gt_count;
        for (uint64_t j = 0; j < result_count; ++j) {
            if (result_ids[j] == gt_ids[i]) {
                ++hit_count;
                break;
            }
        }
    }
    if (valid_gt_count == 0) {
        return 0.0;
    }
    return static_cast<double>(hit_count) / static_cast<double>(valid_gt_count);
}

std::string
extract_route(const std::string& stats) {
    const std::string key = "mci_hybrid_route";
    auto key_pos = stats.find(key);
    if (key_pos == std::string::npos) {
        return "";
    }
    auto colon_pos = stats.find(':', key_pos + key.size());
    auto first_quote = stats.find('"', colon_pos);
    if (colon_pos == std::string::npos or first_quote == std::string::npos) {
        return "";
    }
    auto second_quote = stats.find('"', first_quote + 1);
    if (second_quote == std::string::npos) {
        return "";
    }
    return stats.substr(first_quote + 1, second_quote - first_quote - 1);
}

bool
extract_bool_stat(const std::string& stats, const std::string& key) {
    auto key_pos = stats.find(key);
    if (key_pos == std::string::npos) {
        return false;
    }
    auto colon_pos = stats.find(':', key_pos + key.size());
    if (colon_pos == std::string::npos) {
        return false;
    }
    auto value_pos = stats.find_first_not_of(" \t\n\r", colon_pos + 1);
    return value_pos != std::string::npos and stats.compare(value_pos, 4, "true") == 0;
}

uint64_t
extract_u64_stat(const std::string& stats, const std::string& key) {
    auto key_pos = stats.find(key);
    if (key_pos == std::string::npos) {
        return 0;
    }
    auto colon_pos = stats.find(':', key_pos + key.size());
    if (colon_pos == std::string::npos) {
        return 0;
    }
    auto value_pos = stats.find_first_of("0123456789", colon_pos + 1);
    if (value_pos == std::string::npos) {
        return 0;
    }
    auto value_end = stats.find_first_not_of("0123456789", value_pos);
    return parse_u64(stats.substr(value_pos, value_end - value_pos));
}

}  // namespace

int
main(int argc, char** argv) {
    try {
        vsag::init();
        auto options = parse_args(argc, argv);
        if (options.index_path.empty()) {
            options.index_path = default_index_path(options.data_path);
        }

        H5::H5File file(options.data_path, H5F_ACC_RDONLY);
        auto [total_base, dim] = get_matrix_shape(file, "/train");
        auto [total_queries, query_dim] = get_matrix_shape(file, "/test");
        auto [gt_rows, gt_cols] = get_matrix_shape(file, "/neighbors");
        check(dim == query_dim, "train/test dim mismatch");
        check(gt_rows >= total_queries, "neighbors row count is smaller than test row count");
        check(options.topk > 0, "topk must be positive");
        check(options.thread_count > 0, "threads must be positive");
        auto base_rows =
            options.limit == 0 ? total_base : std::min<uint64_t>(options.limit, total_base);
        auto query_rows = options.query_count == 0
                              ? total_queries
                              : std::min<uint64_t>(options.query_count, total_queries);
        auto recall_topk = std::min<uint64_t>(options.topk, gt_cols);
        auto metric = infer_metric(options.data_path, options.metric);

        vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
        vsag::Engine engine(&resource);
        auto params = make_hgraph_mci_params(options, dim, metric);
        auto index_result = engine.CreateIndex("hgraph", params);
        check_expected(index_result, "create hgraph index failed: ");
        auto index = index_result.value();

        std::ifstream input(options.index_path, std::ios::binary);
        check(input.good(), "failed to open index input: " + options.index_path);
        auto deserialize_result = index->Deserialize(input);
        check_expected(deserialize_result, "deserialize index failed: ");
        std::cout << "loaded index: " << options.index_path
                  << ", vectors=" << index->GetNumElements() << std::endl;

        auto queries = read_float_matrix_prefix(file, "/test", query_rows, dim);
        auto ground_truth = read_i64_matrix_prefix(file, "/neighbors", query_rows, recall_topk);
        std::vector<int64_t> train_labels;
        std::vector<int64_t> test_labels;
        std::vector<float> valid_ratios = read_valid_ratios(file);
        std::unordered_map<int64_t, std::vector<int64_t>> valid_id_map;
        const bool has_labels = dataset_exists(file, "/train_labels") and
                                dataset_exists(file, "/test_labels") and options.use_filter;
        if (has_labels) {
            train_labels = read_i64_prefix(file, "/train_labels", base_rows);
            test_labels = read_i64_prefix(file, "/test_labels", query_rows);
            valid_id_map = build_valid_id_map(train_labels);
            if (base_rows <= options.exact_filtered_gt_limit) {
                auto train = read_float_matrix_prefix(file, "/train", base_rows, dim);
                ground_truth = build_exact_filtered_ground_truth(train,
                                                                 queries,
                                                                 train_labels,
                                                                 test_labels,
                                                                 query_rows,
                                                                 base_rows,
                                                                 dim,
                                                                 recall_topk,
                                                                 metric);
                std::cout << "using exact filtered ground truth for recall, base_rows=" << base_rows
                          << std::endl;
            } else {
                std::cout << "warning: using /neighbors for recall although filter is enabled; "
                          << "increase --exact-filtered-gt-limit for exact filtered recall"
                          << std::endl;
            }
        }

        auto search_params = make_search_params(options);
        std::vector<double> recalls(query_rows, 0.0);
        std::vector<std::string> routes(query_rows);
        std::vector<uint8_t> raw_float_csr_used(query_rows, 0);
        std::vector<uint64_t> dist_cmps(query_rows, 0);
        std::vector<uint64_t> hop_counts(query_rows, 0);
        std::atomic<uint64_t> next_query{0};
        std::atomic<uint64_t> failed_count{0};
        auto worker = [&]() {
            while (true) {
                const auto qi = next_query.fetch_add(1);
                if (qi >= query_rows) {
                    break;
                }

                auto query = vsag::Dataset::Make();
                query->NumElements(1)
                    ->Dim(static_cast<int64_t>(dim))
                    ->Float32Vectors(queries.data() + qi * dim)
                    ->Owner(false);

                std::shared_ptr<vsag::Filter> filter = nullptr;
                if (has_labels) {
                    auto test_label = test_labels[qi];
                    const auto iter = valid_id_map.find(test_label);
                    const std::vector<int64_t>* ids =
                        iter == valid_id_map.end() ? nullptr : &iter->second;
                    auto valid_count = ids == nullptr ? 0 : ids->size();
                    float valid_ratio = static_cast<float>(valid_count) /
                                        static_cast<float>(std::max<uint64_t>(1, base_rows));
                    if (test_label >= 0 and
                        static_cast<uint64_t>(test_label) < valid_ratios.size()) {
                        valid_ratio = valid_ratios[static_cast<uint64_t>(test_label)];
                    }
                    filter =
                        std::make_shared<LabelFilter>(train_labels, test_label, valid_ratio, ids);
                }

                auto result = index->KnnSearch(
                    query, static_cast<int64_t>(options.topk), search_params, filter);
                if (not result.has_value()) {
                    failed_count.fetch_add(1);
                    continue;
                }

                auto dataset = result.value();
                recalls[qi] = recall_for_query(dataset->GetIds(),
                                               static_cast<uint64_t>(dataset->GetDim()),
                                               ground_truth.data() + qi * recall_topk,
                                               recall_topk);
                const auto stats = dataset->GetStatistics();
                routes[qi] = extract_route(stats);
                raw_float_csr_used[qi] = extract_bool_stat(stats, "mci_raw_float_csr") ? 1 : 0;
                dist_cmps[qi] = extract_u64_stat(stats, "dist_cmp");
                hop_counts[qi] = extract_u64_stat(stats, "hops");
            }
        };

        std::cout << "searching queries=" << query_rows << ", topk=" << options.topk
                  << ", recall_topk=" << recall_topk << ", threads=" << options.thread_count
                  << std::endl;
        auto start = std::chrono::steady_clock::now();
        std::vector<std::thread> threads;
        threads.reserve(options.thread_count);
        for (uint64_t i = 0; i < options.thread_count; ++i) {
            threads.emplace_back(worker);
        }
        for (auto& thread : threads) {
            thread.join();
        }
        auto elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        double recall_sum = 0.0;
        uint64_t hgraph_route_count = 0;
        uint64_t mci_route_count = 0;
        uint64_t other_route_count = 0;
        uint64_t raw_float_csr_count = 0;
        uint64_t dist_cmp_sum = 0;
        uint64_t hop_sum = 0;
        for (uint64_t qi = 0; qi < query_rows; ++qi) {
            recall_sum += recalls[qi];
            dist_cmp_sum += dist_cmps[qi];
            hop_sum += hop_counts[qi];
            if (raw_float_csr_used[qi]) {
                ++raw_float_csr_count;
            }
            if (routes[qi] == "hgraph") {
                ++hgraph_route_count;
            } else if (routes[qi] == "mci") {
                ++mci_route_count;
            } else {
                ++other_route_count;
            }
        }
        auto success_count = query_rows - failed_count.load();
        auto recall = success_count == 0 ? 0.0 : recall_sum / static_cast<double>(success_count);
        auto qps = static_cast<double>(success_count) / std::max(elapsed, 1e-9);

        std::cout << std::fixed << std::setprecision(4) << "recall@" << recall_topk << "=" << recall
                  << std::setprecision(1) << " qps=" << qps << std::setprecision(3)
                  << " seconds=" << elapsed << " queries=" << success_count
                  << " failed=" << failed_count.load() << std::endl;
        std::cout << "routes hgraph=" << hgraph_route_count << " mci=" << mci_route_count
                  << " other=" << other_route_count << " raw_float_csr=" << raw_float_csr_count
                  << " avg_cmps="
                  << (success_count == 0
                          ? 0.0
                          : static_cast<double>(dist_cmp_sum) / static_cast<double>(success_count))
                  << " avg_hops="
                  << (success_count == 0
                          ? 0.0
                          : static_cast<double>(hop_sum) / static_cast<double>(success_count))
                  << std::endl;
        engine.Shutdown();
    } catch (const std::exception& err) {
        std::cerr << "error: " << err.what() << std::endl;
        return 1;
    }
    return 0;
}
