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

#include "dataset_profiler.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <limits>
#include <map>
#include <memory>

#include "quantization_simulation.h"
#include "sha256.h"
#include "statistics.h"
#include "vector_math.h"
#include "version.h"

namespace vsag::dataset_profile {

namespace {

constexpr double OOD_PERCENTILE = 0.95;

std::string
utc_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return buffer;
}

uint64_t
peak_rss_bytes() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.starts_with("VmHWM:")) {
            return std::stoull(line.substr(6)) * 1024ULL;
        }
    }
    return 0;
}

JsonType
issues_json(const std::vector<std::string>& issues) {
    JsonType array = JsonType::array();
    for (const auto& issue : issues) {
        array.push_back(issue);
    }
    return array;
}

std::vector<double>
sorted_doubles(const std::vector<double>& values) {
    std::vector<double> result = values;
    std::sort(result.begin(), result.end());
    return result;
}

/// Fraction of `values` above the given percentile of `reference`.
double
fraction_above_percentile(const std::vector<double>& values,
                          std::vector<double> reference,
                          double percentile,
                          double& threshold) {
    std::sort(reference.begin(), reference.end());
    threshold = quantile_sorted(reference, percentile);
    uint64_t above = 0;
    for (const double value : values) {
        above += value > threshold ? 1 : 0;
    }
    return safe_ratio(static_cast<double>(above), static_cast<double>(values.size()));
}

double
median_of(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return quantile_sorted(values, 0.5);
}

/// Sample description for analyses driven by the sampled test queries.
JsonType
query_sample_json(uint64_t population, uint64_t used) {
    return JsonType{{"method", "uniform without replacement (Floyd), seeded shuffle prefix"},
                    {"population", population},
                    {"used_rows", used},
                    {"subset", "first knn_queries valid sampled test rows"}};
}

/// Stored ground truth of the sampled queries. It is read before the base split is scanned, so
/// that pass 2 can recompute the distance of every stored neighbor id (`probe`).
struct GroundTruthCheck {
    /// Columns read per stored row.
    uint64_t k{0};
    /// Checked query rows and their positions in the cross query set.
    std::vector<uint64_t> rows;
    std::vector<uint64_t> positions;
    /// rows.size() x k stored ids and distances.
    std::vector<int64_t> neighbors;
    std::vector<float> distances;
    /// Pairs (cross query, stored id) sorted by id, and the matrix entry of each pair.
    RowDistanceProbe probe;
    std::vector<uint64_t> probe_entries;
};

GroundTruthCheck
prepare_ground_truth_check(const DatasetInfo& info,
                           const std::vector<uint64_t>& query_rows,
                           uint64_t knn_k) {
    GroundTruthCheck check;
    check.k = std::min<uint64_t>(info.ground_truth_k, knn_k);
    for (uint64_t i = 0; i < query_rows.size(); ++i) {
        if (query_rows[i] < info.ground_truth_rows) {
            check.rows.push_back(query_rows[i]);
            check.positions.push_back(i);
        }
    }
    if (check.rows.empty() || check.k == 0) {
        return check;
    }
    read_ground_truth_rows(info.path, check.rows, check.k, check.neighbors, check.distances);
    std::vector<std::pair<uint64_t, uint64_t>> pairs;
    for (uint64_t entry = 0; entry < check.neighbors.size(); ++entry) {
        if (check.neighbors[entry] >= 0) {
            pairs.emplace_back(static_cast<uint64_t>(check.neighbors[entry]), entry);
        }
    }
    std::sort(pairs.begin(), pairs.end());
    for (const auto& [row, entry] : pairs) {
        check.probe.rows.push_back(row);
        check.probe.queries.push_back(check.positions[entry / check.k]);
        check.probe_entries.push_back(entry);
    }
    check.probe.distances.assign(pairs.size(), std::numeric_limits<double>::quiet_NaN());
    return check;
}

/// Compares stored ground truth with the exact neighbors recomputed for sampled queries. The
/// distance of each stored id is recomputed too, so stored distances are never trusted to
/// belong to the stored ids.
JsonType
ground_truth_report(const DatasetInfo& info,
                    const GroundTruthCheck& check,
                    const std::vector<std::vector<Neighbor>>& computed,
                    Metric metric,
                    uint64_t train_rows) {
    const uint64_t k = std::min<uint64_t>(check.k, computed.empty() ? 0 : computed.front().size());
    JsonType report = JsonType{
        {"scope", "sample"},
        {"definition",
         "Stored /neighbors and /distances of sampled queries compared with exact neighbors "
         "recomputed under the profiled metric. The distance of every stored id is recomputed: "
         "distance_recall_at_k counts stored ids within the exact k-th distance (so ties are "
         "tolerated), and the convention errors compare stored distances with those recomputed "
         "distances."},
        {"stored_queries", info.ground_truth_rows},
        {"stored_k", info.ground_truth_k},
        {"checked_queries", check.rows.size()},
        {"compared_k", k}};
    report["sample"] = query_sample_json(info.ground_truth_rows, check.rows.size());
    if (check.rows.empty() || k == 0) {
        report["status"] = "not_checked";
        return report;
    }
    const std::vector<uint64_t>& rows = check.rows;
    const std::vector<int64_t>& neighbors = check.neighbors;
    const std::vector<float>& distances = check.distances;
    // Recomputed metric distance per stored entry; NaN for ids outside the split and for rows
    // excluded from distance metrics.
    std::vector<double> recomputed(neighbors.size(), std::numeric_limits<double>::quiet_NaN());
    for (uint64_t p = 0; p < check.probe_entries.size(); ++p) {
        recomputed[check.probe_entries[p]] = check.probe.distances[p];
    }

    // Candidate conventions for the stored distances, expressed through the VSAG distance m.
    struct Convention {
        const char* name;
        double (*from_metric)(double);
    };
    std::vector<Convention> conventions;
    if (metric == Metric::L2) {
        conventions = {
            {"euclidean (sqrt of l2sqr)", [](double m) { return std::sqrt(std::max(0.0, m)); }},
            {"squared euclidean", [](double m) { return m; }}};
    } else {
        conventions = {{"1 - similarity", [](double m) { return m; }},
                       {"similarity", [](double m) { return 1.0 - m; }},
                       {"negative similarity", [](double m) { return m - 1.0; }}};
    }
    const uint64_t stride = check.k;
    std::vector<double> convention_error(conventions.size(), 0.0);
    uint64_t recomputed_entries = 0;
    double id_recall = 0.0;
    double distance_recall = 0.0;
    uint64_t out_of_range = 0;
    uint64_t unsorted_rows = 0;
    for (uint64_t i = 0; i < rows.size(); ++i) {
        const auto& exact = computed[check.positions[i]];
        const auto first = static_cast<int64_t>(i * stride);
        std::vector<int64_t> stored(neighbors.begin() + first,
                                    neighbors.begin() + first + static_cast<int64_t>(k));
        for (const int64_t id : stored) {
            out_of_range += (id < 0 || static_cast<uint64_t>(id) >= train_rows) ? 1 : 0;
        }
        for (uint64_t j = 1; j < k; ++j) {
            if (distances[i * stride + j] < distances[i * stride + j - 1]) {
                ++unsorted_rows;
                break;
            }
        }
        std::vector<int64_t> exact_ids;
        exact_ids.reserve(k);
        for (uint64_t j = 0; j < k; ++j) {
            exact_ids.push_back(static_cast<int64_t>(exact[j].index));
        }
        std::sort(exact_ids.begin(), exact_ids.end());
        std::vector<int64_t> stored_sorted = stored;
        std::sort(stored_sorted.begin(), stored_sorted.end());
        std::vector<int64_t> common;
        std::set_intersection(exact_ids.begin(),
                              exact_ids.end(),
                              stored_sorted.begin(),
                              stored_sorted.end(),
                              std::back_inserter(common));
        id_recall += static_cast<double>(common.size()) / static_cast<double>(k);
        // A stored id is correct when its recomputed distance is within the exact k-th
        // distance, which absorbs the order of ties.
        const double kth = exact[k - 1].distance;
        const double tolerance = 1e-5 * std::max(1.0, std::abs(kth));
        uint64_t within = 0;
        for (uint64_t j = 0; j < k; ++j) {
            const double actual = recomputed[i * stride + j];
            if (!std::isfinite(actual)) {
                continue;
            }
            within += actual <= kth + tolerance ? 1 : 0;
            ++recomputed_entries;
            const double stored_distance = distances[i * stride + j];
            for (uint64_t c = 0; c < conventions.size(); ++c) {
                const double expected = conventions[c].from_metric(actual);
                convention_error[c] +=
                    std::abs(stored_distance - expected) /
                    std::max(1e-6, std::abs(expected) + std::abs(stored_distance));
            }
        }
        distance_recall += static_cast<double>(within) / static_cast<double>(k);
    }
    const auto checked = static_cast<double>(rows.size());
    report["status"] = "checked";
    report["id_recall_at_k"] = id_recall / checked;
    report["distance_recall_at_k"] = distance_recall / checked;
    report["recomputed_stored_ids"] = recomputed_entries;
    if (recomputed_entries > 0) {
        const uint64_t best = static_cast<uint64_t>(
            std::min_element(convention_error.begin(), convention_error.end()) -
            convention_error.begin());
        const auto entries = static_cast<double>(recomputed_entries);
        JsonType errors = JsonType::object();
        for (uint64_t c = 0; c < conventions.size(); ++c) {
            errors[conventions[c].name] = convention_error[c] / entries;
        }
        report["best_matching_distance_convention"] = conventions[best].name;
        report["best_convention_mean_relative_error"] = convention_error[best] / entries;
        report["convention_mean_relative_errors"] = errors;
    } else {
        report["best_matching_distance_convention"] = nullptr;
        report["best_convention_mean_relative_error"] = nullptr;
        report["convention_mean_relative_errors"] = nullptr;
    }
    report["out_of_range_ids"] = out_of_range;
    report["rows_with_unsorted_distances"] = unsorted_rows;
    return report;
}

JsonType
labels_report(const DatasetInfo& info, const std::vector<std::string>& splits) {
    JsonType report = JsonType{
        {"scope", "full"},
        {"definition",
         "Label frequencies, and for each query label the share of base rows with that label "
         "(filtered-search selectivity)."}};
    std::map<int64_t, uint64_t> train_counts;
    uint64_t train_total = 0;
    for (const auto& split : splits) {
        const std::string name = split + "_labels";
        std::map<int64_t, uint64_t> counts;
        uint64_t total = 0;
        for_each_int64_chunk(
            info.path, name, 1 << 20, [&](uint64_t, const std::vector<int64_t>& values) {
                for (const int64_t value : values) {
                    ++counts[value];
                    ++total;
                }
            });
        double entropy = 0.0;
        uint64_t largest = 0;
        for (const auto& [label, count] : counts) {
            const double share = safe_ratio(static_cast<double>(count), static_cast<double>(total));
            if (share > 0.0) {
                entropy -= share * std::log(share);
            }
            largest = std::max(largest, count);
        }
        report[split] = JsonType{
            {"rows", total},
            {"distinct_labels", counts.size()},
            {"min_label", counts.empty() ? JsonType(nullptr) : JsonType(counts.begin()->first)},
            {"max_label", counts.empty() ? JsonType(nullptr) : JsonType(counts.rbegin()->first)},
            {"largest_label_share",
             safe_ratio(static_cast<double>(largest), static_cast<double>(total))},
            {"normalized_entropy",
             counts.size() > 1 ? entropy / std::log(static_cast<double>(counts.size())) : 0.0}};
        if (split == "train") {
            train_counts = counts;
            train_total = total;
        }
    }
    if (std::find(splits.begin(), splits.end(), "test") != splits.end() && train_total > 0) {
        std::vector<double> selectivity;
        for_each_int64_chunk(
            info.path, "test_labels", 1 << 20, [&](uint64_t, const std::vector<int64_t>& values) {
                for (const int64_t value : values) {
                    const auto it = train_counts.find(value);
                    const uint64_t matches = it == train_counts.end() ? 0 : it->second;
                    selectivity.push_back(static_cast<double>(matches) /
                                          static_cast<double>(train_total));
                }
            });
        uint64_t empty = 0;
        for (const double value : selectivity) {
            empty += value == 0.0 ? 1 : 0;
        }
        std::vector<double> sorted = selectivity;
        report["query_selectivity"] = JsonType{{"summary", summarize_values(sorted)},
                                               {"histogram", histogram_of_sorted(sorted, 20)},
                                               {"queries_without_matching_base_rows", empty}};
    }
    if (info.has_valid_ratios) {
        std::vector<float> ratios = read_float_vector(info.path, "valid_ratios");
        std::vector<double> values(ratios.begin(), ratios.end());
        report["valid_ratios"] =
            JsonType{{"length", ratios.size()}, {"summary", summarize_values(values)}};
    }
    return report;
}

}  // namespace

const std::vector<std::string>&
non_deterministic_sections() {
    static const std::vector<std::string> sections = {"run", "ai_analysis"};
    return sections;
}

const JsonType*
find_path(const JsonType& root, const std::string& path) {
    const JsonType* node = &root;
    uint64_t begin = 0;
    while (begin <= path.size()) {
        const uint64_t end = std::min(path.find('.', begin), path.size());
        const std::string segment = path.substr(begin, end - begin);
        if (node->is_object()) {
            const auto it = node->find(segment);
            if (it == node->end()) {
                return nullptr;
            }
            node = &(*it);
        } else if (node->is_array()) {
            if (segment.empty() || segment.find_first_not_of("0123456789") != std::string::npos) {
                return nullptr;
            }
            const uint64_t index = std::stoull(segment);
            if (index >= node->size()) {
                return nullptr;
            }
            node = &(*node)[index];
        } else {
            return nullptr;
        }
        if (end == path.size()) {
            break;
        }
        begin = end + 1;
    }
    return node;
}

std::string
metrics_digest(const JsonType& report) {
    JsonType deterministic = report;
    for (const auto& section : non_deterministic_sections()) {
        deterministic.erase(section);
    }
    deterministic.erase("determinism");
    return sha256_hex(deterministic.dump());
}

JsonType
profile_dataset(const ProfileOptions& options, const ProgressFn& progress) {
    validate_options(options);
    const auto start = std::chrono::steady_clock::now();
    const std::string started_at = utc_timestamp();
    DatasetInfo info = inspect_dataset(options.data_path);

    Metric metric = info.kind == VectorKind::SPARSE ? Metric::IP : Metric::L2;
    std::string metric_source;
    Metric attribute_metric = Metric::L2;
    const bool attribute_known =
        metric_from_distance_attribute(info.distance_attribute, attribute_metric);
    std::vector<std::string> warnings;
    if (options.metric == "auto") {
        if (attribute_known) {
            metric = attribute_metric;
            metric_source = "hdf5 distance attribute '" + info.distance_attribute + "'";
        } else {
            metric_source = "default for " +
                            std::string(info.kind == VectorKind::SPARSE ? "sparse" : "dense") +
                            " data (distance attribute missing or unknown)";
            warnings.push_back("metric defaulted to " + metric_name(metric) +
                               "; pass --metric to choose explicitly");
        }
    } else {
        metric = parse_metric(options.metric);
        metric_source = "--metric";
        if (attribute_known && attribute_metric != metric) {
            warnings.push_back("--metric " + metric_name(metric) +
                               " differs from the HDF5 distance attribute '" +
                               info.distance_attribute + "'");
        }
    }
    if (info.kind == VectorKind::SPARSE && metric == Metric::L2) {
        throw ProfileError("sparse datasets support the ip and cosine metrics only");
    }

    std::vector<std::string> splits = options.splits;
    if (splits.empty()) {
        for (const char* name : {"train", "test"}) {
            if (info.Split(name).present) {
                splits.emplace_back(name);
            }
        }
    }
    for (const auto& split : splits) {
        if (!info.Split(split).present) {
            throw ProfileError("requested split '" + split + "' is not present in the file");
        }
    }
    const bool profile_train = std::find(splits.begin(), splits.end(), "train") != splits.end();
    const bool profile_test = std::find(splits.begin(), splits.end(), "test") != splits.end();
    const bool both = profile_train && profile_test;
    const bool knn = options.Enabled(SECTION_KNN);

    JsonType report;
    report["schema"] = REPORT_SCHEMA;
    report["schema_version"] = REPORT_SCHEMA_VERSION;
    report["generator"] = JsonType{{"tool", "profile_dataset"}, {"vsag_version", VSAG_VERSION}};
    JsonType datasets = JsonType::array();
    for (const auto& name : info.dataset_names) {
        datasets.push_back(name);
    }
    report["input"] = JsonType{
        {"file_name", info.file_name},
        {"file_size_bytes", info.file_size_bytes},
        {"format", "vsag hdf5"},
        {"vector_type", info.kind == VectorKind::SPARSE ? "sparse" : "dense"},
        {"type_attribute",
         info.type_attribute.empty() ? JsonType(nullptr) : JsonType(info.type_attribute)},
        {"distance_attribute",
         info.distance_attribute.empty() ? JsonType(nullptr) : JsonType(info.distance_attribute)},
        {"hdf5_datasets", datasets},
        {"eval_compatibility",
         JsonType{{"compatible", info.eval_compatibility_issues.empty()},
                  {"issues", issues_json(info.eval_compatibility_issues)}}},
        {"warnings", issues_json(warnings)}};
    JsonType config = options_to_json(options);
    config["metric"] = metric_name(metric);
    config["metric_source"] = metric_source;
    JsonType profiled = JsonType::array();
    for (const auto& split : splits) {
        profiled.push_back(split);
    }
    config["profiled_splits"] = profiled;
    report["config"] = config;
    report["splits"] = JsonType::object();

    auto make_context = [&](const std::string& name) {
        SplitContext context;
        context.info = &info;
        context.name = name;
        context.options = &options;
        context.metric = metric;
        context.progress = progress;
        return context;
    };

    const bool check_truth = both && knn && options.Enabled(SECTION_GROUND_TRUTH) &&
                             info.has_neighbors && info.has_distances;
    GroundTruthCheck truth;
    if (info.kind == VectorKind::DENSE) {
        std::unique_ptr<DenseSplitProfiler> test;
        std::unique_ptr<DenseSplitProfiler> train;
        DenseQuerySet cross;
        if (profile_test) {
            test = std::make_unique<DenseSplitProfiler>(make_context("test"));
            test->Run(nullptr);
            if (profile_train && knn) {
                cross = test->MakeQuerySet(options.knn_queries);
            }
            if (check_truth) {
                truth =
                    prepare_ground_truth_check(info,
                                               test->AnalysisRowIds(options.knn_queries),
                                               std::min<uint64_t>(options.knn_k, info.train.rows));
            }
        }
        if (profile_train) {
            train = std::make_unique<DenseSplitProfiler>(make_context("train"));
            train->Run(cross.Size() > 0 ? &cross : nullptr, check_truth ? &truth.probe : nullptr);
            report["splits"]["train"] = train->Json();
        }
        if (test) {
            report["splits"]["test"] = test->Json();
        }
        if (both) {
            const auto train_norms = sorted_doubles(train->SortedNorms());
            const auto test_norms = sorted_doubles(test->SortedNorms());
            std::vector<double> shift(train->DimensionMeans().size(), 0.0);
            for (uint64_t d = 0; d < shift.size(); ++d) {
                const double spread = train->DimensionStds()[d];
                shift[d] =
                    spread > 0.0
                        ? std::abs(test->DimensionMeans()[d] - train->DimensionMeans()[d]) / spread
                        : 0.0;
            }
            std::vector<double> shift_sorted = shift;
            JsonType cross_json = JsonType{
                {"scope", "mixed"},
                {"definition",
                 "Query (test) versus base (train) distribution: norm KS statistic and "
                 "per-dimension "
                 "standardized mean shift over full splits; out-of-distribution ratio from exact "
                 "k-NN "
                 "of sampled queries."},
                {"norm_ks_statistic",
                 JsonType{{"scope", "full"}, {"value", ks_statistic(train_norms, test_norms)}}},
                {"standardized_mean_shift",
                 JsonType{{"scope", "full"}, {"summary", summarize_values(shift_sorted)}}}};
            if (!train->CrossGeometric().empty() && !train->SelfFirstNeighborGeometric().empty()) {
                std::vector<double> query_first;
                for (const auto& distances : train->CrossGeometric()) {
                    if (!distances.empty()) {
                        query_first.push_back(distances.front());
                    }
                }
                double threshold = 0.0;
                const double ratio = fraction_above_percentile(
                    query_first, train->SelfFirstNeighborGeometric(), OOD_PERCENTILE, threshold);
                std::vector<double> query_sorted = query_first;
                const JsonType sample = query_sample_json(info.test.rows, query_first.size());
                cross_json["query_first_neighbor_distance"] = JsonType{
                    {"scope", "sample"},
                    {"definition",
                     "Geometric distance from sampled test queries to their nearest train row."},
                    {"sample", sample},
                    {"summary", summarize_values(query_sorted)}};
                cross_json["out_of_distribution_ratio"] = JsonType{
                    {"scope", "sample"},
                    {"definition",
                     "Share of sampled queries whose nearest-train distance exceeds the 95th "
                     "percentile "
                     "of train rows' own nearest-neighbor distances."},
                    {"sample", sample},
                    {"value", ratio},
                    {"threshold", threshold},
                    {"median_distance_ratio",
                     json_number(safe_ratio(median_of(query_first),
                                            median_of(train->SelfFirstNeighborGeometric())))}};
            }
            if (train->HasPca() && options.projection_points > 0) {
                const uint64_t points = std::min<uint64_t>(options.projection_points / 2, 1000);
                const std::vector<float> rows = test->AnalysisRows(points);
                const uint64_t count = rows.size() / std::max<uint64_t>(info.train.dim, 1);
                JsonType coordinates = JsonType::array();
                for (uint64_t i = 0; i < count; ++i) {
                    double x = 0.0;
                    double y = 0.0;
                    project_2d(train->Pca(), rows.data() + i * info.train.dim, x, y);
                    coordinates.push_back(
                        JsonType::array({round_significant(x, 6), round_significant(y, 6)}));
                }
                cross_json["query_projection"] = JsonType{
                    {"scope", "sample"},
                    {"definition", "Sampled test rows projected on the train principal axes."},
                    {"sample", query_sample_json(info.test.rows, count)},
                    {"points", coordinates}};
            }
            report["cross_split"] = cross_json;
            if (check_truth) {
                report["ground_truth"] = ground_truth_report(
                    info, truth, train->CrossNeighbors(), metric, info.train.rows);
            }
        }
        if (options.Enabled(SECTION_QUANTIZATION) && profile_train &&
            info.train.storage_type != "int8" && info.train.storage_type != "uint8") {
            const uint64_t dim = info.train.dim;
            std::vector<float> base;
            std::vector<float> queries;
            uint64_t base_rows = 0;
            uint64_t query_rows = 0;
            std::string query_source;
            const std::vector<float> train_rows =
                train->AnalysisRows(options.quantization_base + options.quantization_queries);
            const uint64_t available = train_rows.size() / std::max<uint64_t>(dim, 1);
            if (test) {
                queries = test->AnalysisRows(options.quantization_queries);
                query_rows = queries.size() / std::max<uint64_t>(dim, 1);
                base_rows = std::min<uint64_t>(options.quantization_base, available);
                query_source = "sampled test rows";
            } else {
                query_rows = std::min<uint64_t>(options.quantization_queries, available / 5);
                base_rows = std::min<uint64_t>(options.quantization_base, available - query_rows);
                queries.assign(
                    train_rows.begin() + static_cast<int64_t>(base_rows * dim),
                    train_rows.begin() + static_cast<int64_t>((base_rows + query_rows) * dim));
                query_source = "sampled train rows disjoint from the base sample";
            }
            base.assign(train_rows.begin(),
                        train_rows.begin() + static_cast<int64_t>(base_rows * dim));
            if (base_rows >= 2 && query_rows >= 1) {
                JsonType quantization = simulate_quantization(
                    base, base_rows, queries, query_rows, dim, metric, options.quantization_k);
                JsonType section = JsonType{
                    {"scope", "sample"},
                    {"definition",
                     "Recall@k of simulated VSAG quantizers (ranges and floor encoding as in "
                     "ScalarQuantizationTrainer) against exact search on a sampled base. A model, "
                     "not "
                     "a benchmark: confirm with eval_performance."},
                    {"query_source", query_source},
                    {"sample",
                     JsonType{
                         {"method", "uniform without replacement (Floyd), seeded shuffle prefix"},
                         {"population", info.train.rows},
                         {"used_rows", base_rows},
                         {"subset", "first quantization_base valid sampled train rows"}}}};
                section.update(quantization);
                report["splits"]["train"]["quantization"] = section;
            }
        }
    } else {
        std::unique_ptr<SparseSplitProfiler> test;
        std::unique_ptr<SparseSplitProfiler> train;
        SparseQuerySet cross;
        if (profile_test) {
            test = std::make_unique<SparseSplitProfiler>(make_context("test"));
            test->Run(nullptr);
            if (profile_train && knn) {
                cross = test->MakeQuerySet(options.knn_queries);
            }
            if (check_truth) {
                // The train record count may only be known after its scan; the report trims
                // the comparison to the k actually searched.
                truth = prepare_ground_truth_check(
                    info, test->AnalysisRowIds(options.knn_queries), options.knn_k);
            }
        }
        if (profile_train) {
            train = std::make_unique<SparseSplitProfiler>(make_context("train"));
            train->Run(cross.Size() > 0 ? &cross : nullptr, check_truth ? &truth.probe : nullptr);
            report["splits"]["train"] = train->Json();
        }
        if (test) {
            report["splits"]["test"] = test->Json();
        }
        if (both) {
            const auto train_norms = sorted_doubles(train->SortedNorms());
            const auto test_norms = sorted_doubles(test->SortedNorms());
            JsonType cross_json = JsonType{
                {"scope", "mixed"},
                {"definition",
                 "Query (test) versus base (train): KS statistics over full splits; term coverage "
                 "and inverted-list work estimated from sampled queries."},
                {"norm_ks_statistic",
                 JsonType{{"scope", "full"}, {"value", ks_statistic(train_norms, test_norms)}}},
                {"nonzeros_ks_statistic",
                 JsonType{
                     {"scope", "full"},
                     {"value", ks_statistic(train->SortedNonzeros(), test->SortedNonzeros())}}}};
            const SparseQuerySet queries = test->MakeQuerySet(options.knn_queries);
            uint64_t covered = 0;
            uint64_t terms = 0;
            std::vector<double> postings;
            for (uint64_t q = 0; q < queries.Size(); ++q) {
                double work = 0.0;
                for (uint64_t e = queries.offsets[q]; e < queries.offsets[q + 1]; ++e) {
                    const uint64_t frequency = train->DocumentFrequency(queries.ids[e]);
                    covered += frequency > 0 ? 1 : 0;
                    ++terms;
                    work += static_cast<double>(frequency);
                }
                postings.push_back(work);
            }
            std::vector<double> postings_sorted = postings;
            const JsonType sample =
                query_sample_json(test->Json()["rows"].get<uint64_t>(), queries.Size());
            cross_json["query_term_coverage"] = JsonType{
                {"scope", "sample"},
                {"definition",
                 "Share of sampled query terms that occur in at least one train record."},
                {"sample", sample},
                {"value", safe_ratio(static_cast<double>(covered), static_cast<double>(terms))}};
            cross_json["postings_per_query"] = JsonType{
                {"scope", "sample"},
                {"definition",
                 "Sum of train document frequencies over a query's terms: postings an exhaustive "
                 "inverted-index scan would touch."},
                {"sample", sample},
                {"summary", summarize_values(postings_sorted)}};
            if (!train->CrossNeighbors().empty() && !train->SelfFirstNeighborDistance().empty()) {
                std::vector<double> query_first;
                for (const auto& neighbors : train->CrossNeighbors()) {
                    query_first.push_back(neighbors.front().distance);
                }
                double threshold = 0.0;
                const double ratio = fraction_above_percentile(
                    query_first, train->SelfFirstNeighborDistance(), OOD_PERCENTILE, threshold);
                cross_json["out_of_distribution_ratio"] =
                    JsonType{{"scope", "sample"},
                             {"definition",
                              "Share of sampled queries whose nearest-train distance exceeds the "
                              "95th percentile "
                              "of train records' own nearest-neighbor distances."},
                             {"sample", sample},
                             {"value", ratio},
                             {"threshold", threshold}};
            }
            report["cross_split"] = cross_json;
            if (check_truth) {
                report["ground_truth"] = ground_truth_report(info,
                                                             truth,
                                                             train->CrossNeighbors(),
                                                             metric,
                                                             train->Json()["rows"].get<uint64_t>());
            }
        }
    }

    if (options.Enabled(SECTION_LABELS) && info.has_train_labels) {
        std::vector<std::string> label_splits;
        for (const auto& split : splits) {
            if ((split == "train" && info.has_train_labels) ||
                (split == "test" && info.has_test_labels)) {
                label_splits.push_back(split);
            }
        }
        if (!label_splits.empty() &&
            std::find(label_splits.begin(), label_splits.end(), "train") == label_splits.end() &&
            std::find(label_splits.begin(), label_splits.end(), "test") != label_splits.end()) {
            label_splits.insert(label_splits.begin(), "train");
        }
        report["labels"] = labels_report(info, label_splits);
    }

    report["signals"] = derive_signals(report);
    report["determinism"] = JsonType{
        {"metrics_sha256", metrics_digest(report)},
        {"excluded_sections", non_deterministic_sections()},
        {"definition",
         "SHA-256 of every section except run and ai_analysis. Identical inputs, options and "
         "seed give the same digest for any thread count or chunk size."}};
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    report["run"] = JsonType{{"started_at", started_at},
                             {"duration_seconds", seconds},
                             {"threads", omp_get_max_threads()},
                             {"chunk_mb", options.chunk_mb},
                             {"peak_rss_bytes", peak_rss_bytes()}};
    return report;
}

}  // namespace vsag::dataset_profile
