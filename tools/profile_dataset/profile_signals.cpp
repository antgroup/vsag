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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "dataset_profiler.h"

namespace vsag::dataset_profile {

namespace {

std::string
format_number(double value) {
    char buffer[32];
    if (value != 0.0 && (std::abs(value) >= 1e6 || std::abs(value) < 1e-3)) {
        std::snprintf(buffer, sizeof(buffer), "%.3g", value);
    } else if (value == std::floor(value) && std::abs(value) < 1e15) {
        std::snprintf(buffer, sizeof(buffer), "%.0f", value);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%.3f", value);
    }
    return buffer;
}

std::string
percent(double ratio) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.2f%%", ratio * 100.0);
    return buffer;
}

class SignalBuilder {
public:
    explicit SignalBuilder(const JsonType& report) : report_(report) {
    }

    /// Reads a numeric report value; returns false when the key is absent or not a number.
    bool
    Number(const std::string& path, double& value) const {
        const JsonType* node = find_path(report_, path);
        if (node == nullptr || !node->is_number()) {
            return false;
        }
        value = node->get<double>();
        return true;
    }

    [[nodiscard]] std::string
    String(const std::string& path) const {
        const JsonType* node = find_path(report_, path);
        return node != nullptr && node->is_string() ? node->get<std::string>() : "";
    }

    void
    Add(const std::string& id,
        const std::string& severity,
        const std::string& message,
        const std::vector<std::string>& evidence,
        const std::string& rule) {
        JsonType keys = JsonType::array();
        for (const auto& key : evidence) {
            if (find_path(report_, key) != nullptr) {
                keys.push_back(key);
            }
        }
        signals_.push_back(JsonType{{"id", id},
                                    {"severity", severity},
                                    {"message", message},
                                    {"evidence", keys},
                                    {"rule", rule}});
    }

    /// Signals ordered by severity (critical first); insertion order breaks ties.
    [[nodiscard]] JsonType
    Result() const {
        auto rank = [](const JsonType& signal) {
            const std::string severity = signal.value("severity", "info");
            return severity == "critical"  ? 0
                   : severity == "warning" ? 1
                   : severity == "notice"  ? 2
                                           : 3;
        };
        std::vector<JsonType> ordered(signals_.begin(), signals_.end());
        std::stable_sort(
            ordered.begin(), ordered.end(), [&rank](const JsonType& a, const JsonType& b) {
                return rank(a) < rank(b);
            });
        JsonType result = JsonType::array();
        for (auto& signal : ordered) {
            result.push_back(std::move(signal));
        }
        return result;
    }

private:
    const JsonType& report_;
    JsonType signals_ = JsonType::array();
};

void
dense_split_signals(SignalBuilder& s, const std::string& split, const std::string& metric) {
    const std::string p = "splits." + split + ".";
    double value = 0.0;
    double other = 0.0;
    if (s.Number(p + "invalid_values.total", value) && value > 0) {
        s.Number(p + "invalid_values.rows_with_invalid", other);
        s.Add(split + ".invalid_values",
              "critical",
              format_number(value) + " NaN/Inf values in " + format_number(other) + " " + split +
                  " rows; distances involving these rows are undefined, so clean or drop them "
                  "before building an index.",
              {p + "invalid_values.total", p + "invalid_values.rows_with_invalid"},
              "invalid_values.total > 0");
    }
    if (s.Number(p + "norms.zero_norm_rows", value) && value > 0) {
        s.Add(split + ".zero_vectors",
              metric == "cosine" ? "critical" : "warning",
              format_number(value) +
                  " all-zero rows; cosine similarity is undefined for them and they "
                  "coincide under l2/ip.",
              {p + "norms.zero_norm_rows"},
              "norms.zero_norm_rows > 0");
    }
    if (s.Number(p + "duplicates.duplicate_ratio", value) && value > 0) {
        s.Add(split + ".exact_duplicates",
              value > 0.01 ? "warning" : "info",
              percent(value) +
                  " of rows duplicate another row; duplicates waste memory and can crowd "
                  "top-k results with identical items.",
              {p + "duplicates.duplicate_ratio",
               p + "duplicates.groups",
               p + "duplicates.largest_group"},
              "duplicates.duplicate_ratio > 0 (warning above 1%)");
    }
    if (s.Number(p + "nearest_neighbors.near_duplicate_ratio", value) && value > 0.05) {
        s.Add(split + ".near_duplicates",
              "notice",
              percent(value) +
                  " of sampled rows have a near-identical neighbor; graph indexes spend "
                  "edges on them and recall@k can saturate early.",
              {p + "nearest_neighbors.near_duplicate_ratio",
               p + "nearest_neighbors.near_duplicate_threshold.geometric_distance"},
              "nearest_neighbors.near_duplicate_ratio > 0.05");
    }
    if (s.Number(p + "norms.unit_norm_ratio", value) && value >= 0.99) {
        s.Add(split + ".unit_normalized",
              "info",
              "Rows are L2-normalized (" + percent(value) +
                  " within 1e-3 of norm 1): l2, ip and cosine "
                  "rank neighbors identically.",
              {p + "norms.unit_norm_ratio"},
              "norms.unit_norm_ratio >= 0.99");
    } else if (metric == "ip" && s.Number(p + "norms.coefficient_of_variation", value) &&
               value > 0.25) {
        s.Add(split + ".ip_norm_spread",
              "warning",
              "Norms vary widely (CV " + format_number(value) +
                  ") under ip: large-norm rows dominate "
                  "inner-product results. Confirm that ip rather than cosine is intended.",
              {p + "norms.coefficient_of_variation",
               p + "norms.summary.p05",
               p + "norms.summary.p95"},
              "metric == ip and norms.coefficient_of_variation > 0.25");
    }
    if (s.Number(p + "dimensions.summary.constant_dimensions", value) && value > 0) {
        s.Add(split + ".constant_dimensions",
              "notice",
              format_number(value) +
                  " constant dimensions carry no information but still cost memory "
                  "and distance work.",
              {p + "dimensions.summary.constant_dimensions", "splits." + split + ".dim"},
              "dimensions.summary.constant_dimensions > 0");
    }
    if (s.Number(p + "dimensions.summary.range_ratio_max_to_median", value) && value > 4.0) {
        s.Add(split + ".heterogeneous_ranges",
              "notice",
              "The widest dimension range is " + format_number(value) +
                  "x the median; one global range "
                  "(sq8_uniform / sq4_uniform) spends codes unevenly, so compare with "
                  "per-dimension sq8 or "
                  "rotate first (tq chain rom/fht).",
              {p + "dimensions.summary.range_ratio_max_to_median",
               p + "dimensions.summary.std_ratio_max_to_median"},
              "dimensions.summary.range_ratio_max_to_median > 4");
    }
    if (s.Number(p + "values.outliers_6_sigma.ratio", value) && value > 1e-3) {
        s.Add(split + ".heavy_tails",
              "notice",
              percent(value) +
                  " of coordinates lie beyond 6 sigma of their dimension; min/max-trained "
                  "scalar quantizers lose resolution for the bulk of values.",
              {p + "values.outliers_6_sigma.ratio",
               p + "dimensions.summary.heavy_tailed_dimension_ratio"},
              "values.outliers_6_sigma.ratio > 0.001");
    }
    double dim = 0.0;
    if (s.Number(p + "pca.components_for_variance.90", value) &&
        s.Number("splits." + split + ".dim", dim) && value <= 0.25 * dim) {
        s.Add(split + ".variance_concentrated",
              "info",
              "90% of the sample variance lies in " + format_number(value) + " of " +
                  format_number(dim) +
                  " principal components; a PCA stage in a tq chain (e.g. \"pca, rom, "
                  "sq8_uniform\") is a "
                  "candidate to evaluate.",
              {p + "pca.components_for_variance.90",
               p + "pca.effective_rank",
               "splits." + split + ".dim"},
              "pca.components_for_variance.90 <= 0.25 * dim");
    }
    if (s.Number(p + "relative_contrast.summary.p50", value) && value < 2.0) {
        s.Add(split + ".low_relative_contrast",
              value < 1.5 ? "warning" : "notice",
              "Median relative contrast is " + format_number(value) +
                  ": nearest neighbors are barely "
                  "closer than random rows, so expect larger ef_search (or scan_buckets_count) "
                  "for a target recall.",
              {p + "relative_contrast.summary.p50", p + "pairwise_distances.concentration_ratio"},
              "relative_contrast.summary.p50 < 2 (warning below 1.5)");
    }
    if (s.Number(p + "intrinsic_dimensionality.lid_mle.summary.p50", value) && value > 20.0) {
        s.Add(split + ".high_intrinsic_dimensionality",
              "notice",
              "Median local intrinsic dimensionality is " + format_number(value) +
                  "; high-LID data needs "
                  "more candidates per query for the same recall.",
              {p + "intrinsic_dimensionality.lid_mle.summary.p50",
               p + "intrinsic_dimensionality.two_nn"},
              "intrinsic_dimensionality.lid_mle.summary.p50 > 20");
    }
    if (s.Number(p + "hubness.skewness", value) && value > 1.0) {
        s.Number(p + "hubness.anti_hub_ratio", other);
        s.Add(split + ".hubness",
              "notice",
              "k-occurrence skewness is " + format_number(value) + " (" + percent(other) +
                  " of sampled rows "
                  "never appear as neighbors): hubs attract graph traversals while anti-hubs are "
                  "hard to reach.",
              {p + "hubness.skewness", p + "hubness.anti_hub_ratio"},
              "hubness.skewness > 1");
    }
    if (s.Number(p + "nearest_neighbors.isolated_ratio", value) && value > 0.01) {
        s.Add(split + ".isolated_rows",
              "notice",
              percent(value) +
                  " of sampled rows are far from every other row; such rows are poorly "
                  "connected in graph indexes.",
              {p + "nearest_neighbors.isolated_ratio"},
              "nearest_neighbors.isolated_ratio > 0.01");
    }
    if (s.Number(p + "anisotropy.average_cosine_similarity.value", value) && value > 0.5) {
        s.Add(split + ".narrow_cone",
              "notice",
              "Average pairwise cosine similarity is " + format_number(value) +
                  ": rows share a dominant "
                  "direction, which compresses cosine distances; centering or a rotation can help "
                  "quantization.",
              {p + "anisotropy.average_cosine_similarity.value",
               p + "anisotropy.mean_vector_norm_ratio.value"},
              "anisotropy.average_cosine_similarity.value > 0.5");
    }
    if (s.Number(p + "clustering.max_to_mean_size", value) && value > 3.0) {
        s.Add(split + ".imbalanced_clusters",
              "notice",
              "The largest k-means cluster is " + format_number(value) +
                  "x the mean size; IVF buckets "
                  "trained on this data may be imbalanced.",
              {p + "clustering.max_to_mean_size", p + "clustering.size_cv"},
              "clustering.max_to_mean_size > 3");
    }
    if (s.Number(p + "quantization.schemes.sq8.recall_at_k", value) && value < 0.95) {
        s.Add(split + ".sq8_recall",
              "notice",
              "Simulated sq8 recall@k is " + format_number(value) +
                  "; plan for use_reorder with a precise "
                  "fp32 store when using 8-bit codes.",
              {p + "quantization.schemes.sq8.recall_at_k",
               p + "quantization.schemes.fp16.recall_at_k"},
              "quantization.schemes.sq8.recall_at_k < 0.95");
    }
    if (s.Number(p + "quantization.schemes.sq8.recall_at_k", value) &&
        s.Number(p + "quantization.schemes.sq8_uniform.recall_at_k", other) &&
        value - other > 0.02) {
        s.Add(split + ".uniform_range_loss",
              "notice",
              "Simulated sq8_uniform recall@k (" + format_number(other) +
                  ") trails per-dimension sq8 (" + format_number(value) +
                  "); one global range costs recall here, so a rotation (tq chain rom/fht) is a "
                  "candidate to evaluate.",
              {p + "quantization.schemes.sq8.recall_at_k",
               p + "quantization.schemes.sq8_uniform.recall_at_k"},
              "sq8 recall - sq8_uniform recall > 0.02");
    }
}

void
sparse_split_signals(SignalBuilder& s, const std::string& split) {
    const std::string p = "splits." + split + ".";
    double value = 0.0;
    if (s.Number(p + "invalid_values.total", value) && value > 0) {
        s.Add(split + ".invalid_values",
              "critical",
              format_number(value) +
                  " NaN/Inf stored values; scores involving these records are undefined.",
              {p + "invalid_values.total", p + "invalid_values.rows_with_invalid"},
              "invalid_values.total > 0");
    }
    if (s.Number(p + "basic.rows_with_repeated_ids", value) && value > 0) {
        s.Add(split + ".repeated_term_ids",
              "warning",
              format_number(value) +
                  " records repeat a term id; inverted-index scores for them depend on "
                  "how duplicates are merged.",
              {p + "basic.rows_with_repeated_ids"},
              "basic.rows_with_repeated_ids > 0");
    }
    if (s.Number(p + "values.negative_values", value) && value > 0) {
        s.Add(split + ".negative_values",
              "notice",
              format_number(value) +
                  " negative values; SINDI doc pruning ranks terms by value mass, which "
                  "assumes non-negative weights.",
              {p + "values.negative_values"},
              "values.negative_values > 0");
    }
    if (s.Number(p + "terms.min_term_id_limit", value) && value > 0) {
        s.Add(split + ".term_id_limit",
              "info",
              "The largest term id requires SINDI term_id_limit >= " + format_number(value) + ".",
              {p + "terms.min_term_id_limit", p + "terms.max_term_id"},
              "always reported for sparse data");
    }
    if (s.Number(p + "terms.term_id_density", value) && value < 0.1 && value > 0) {
        s.Add(split + ".sparse_term_ids",
              "info",
              "Only " + percent(value) +
                  " of ids below the maximum are used; remap_term_ids can shrink "
                  "per-term structures.",
              {p + "terms.term_id_density", p + "terms.distinct_terms"},
              "terms.term_id_density < 0.1");
    }
    if (s.Number(p + "terms.doc_prune_retained_term_fraction.by_ratio.10", value) && value < 0.7) {
        s.Add(split + ".prunable_mass",
              "info",
              "doc_prune_ratio 0.1 would keep " + percent(value) +
                  " of terms while keeping 90% of the value "
                  "mass; pruning is a candidate to trade a little recall for memory and latency.",
              {p + "terms.doc_prune_retained_term_fraction.by_ratio.10",
               p + "terms.doc_prune_retained_term_fraction.by_ratio.30"},
              "doc_prune_retained_term_fraction.by_ratio.10 < 0.7");
    }
    if (s.Number(p + "terms.top_1pct_terms_posting_share", value) && value > 0.5) {
        s.Add(split + ".dominant_terms",
              "notice",
              "The top 1% of terms hold " + percent(value) +
                  " of all postings; their long lists dominate "
                  "inverted-index work, so query/term pruning matters.",
              {p + "terms.top_1pct_terms_posting_share", p + "terms.zipf_exponent"},
              "terms.top_1pct_terms_posting_share > 0.5");
    }
    if (s.Number(p + "duplicates.duplicate_ratio", value) && value > 0) {
        s.Add(split + ".exact_duplicates",
              value > 0.01 ? "warning" : "info",
              percent(value) + " of records duplicate another record.",
              {p + "duplicates.duplicate_ratio", p + "duplicates.groups"},
              "duplicates.duplicate_ratio > 0 (warning above 1%)");
    }
    if (s.Number(p + "nearest_neighbors.queries_with_fewer_than_k_overlaps_ratio", value) &&
        value > 0.1) {
        s.Add(split + ".few_overlaps",
              "notice",
              percent(value) +
                  " of sampled records share terms with fewer than k other records; their "
                  "top-k lists are padded with zero-score results.",
              {p + "nearest_neighbors.queries_with_fewer_than_k_overlaps_ratio"},
              "queries_with_fewer_than_k_overlaps_ratio > 0.1");
    }
}

}  // namespace

JsonType
derive_signals(const JsonType& report) {
    SignalBuilder s(report);
    const std::string metric = s.String("config.metric");
    const bool sparse = s.String("input.vector_type") == "sparse";
    const JsonType* splits = find_path(report, "splits");
    if (splits != nullptr && splits->is_object()) {
        for (const auto& item : splits->items()) {
            if (sparse) {
                sparse_split_signals(s, item.key());
            } else {
                dense_split_signals(s, item.key(), metric);
            }
        }
    }
    const JsonType* issues = find_path(report, "input.eval_compatibility.issues");
    if (issues != nullptr && issues->is_array()) {
        for (uint64_t i = 0; i < issues->size(); ++i) {
            s.Add("input.eval_compatibility." + std::to_string(i),
                  "warning",
                  "eval_performance compatibility: " + (*issues)[i].get<std::string>(),
                  {"input.eval_compatibility.issues." + std::to_string(i)},
                  "reported by the HDF5 inspection");
        }
    }
    const JsonType* warnings = find_path(report, "input.warnings");
    if (warnings != nullptr && warnings->is_array()) {
        for (uint64_t i = 0; i < warnings->size(); ++i) {
            s.Add("input.warning." + std::to_string(i),
                  "warning",
                  (*warnings)[i].get<std::string>(),
                  {"input.warnings." + std::to_string(i), "config.metric_source"},
                  "metric resolution");
        }
    }
    double value = 0.0;
    double other = 0.0;
    const bool has_ks = s.Number("cross_split.norm_ks_statistic.value", value);
    const bool has_ood = s.Number("cross_split.out_of_distribution_ratio.value", other);
    if ((has_ks && value > 0.1) || (has_ood && other > 0.1)) {
        s.Add("cross_split.query_shift",
              "warning",
              "Queries differ from the base data (norm KS " + format_number(value) + ", " +
                  percent(other) +
                  " of sampled queries farther from the base than 95% of base rows are from each "
                  "other); "
                  "tune and evaluate with real queries.",
              {"cross_split.norm_ks_statistic.value",
               "cross_split.out_of_distribution_ratio.value",
               "cross_split.standardized_mean_shift.summary.max"},
              "norm KS > 0.1 or out-of-distribution ratio > 0.1");
    }
    if (s.Number("ground_truth.distance_recall_at_k", value) && value < 0.99) {
        s.Add("ground_truth.mismatch",
              "critical",
              "Stored ground truth agrees with recomputed exact neighbors for only " +
                  percent(value) + " of entries under the " + metric +
                  " metric; recall measured against it would be wrong.",
              {"ground_truth.distance_recall_at_k", "ground_truth.id_recall_at_k", "config.metric"},
              "ground_truth.distance_recall_at_k < 0.99");
    }
    if (s.Number("ground_truth.best_convention_mean_relative_error", value) && value > 1e-3) {
        s.Add("ground_truth.distance_convention",
              "warning",
              "Stored /distances match no supported convention closely (best: " +
                  s.String("ground_truth.best_matching_distance_convention") +
                  ", mean relative error " + format_number(value) + ").",
              {"ground_truth.best_matching_distance_convention",
               "ground_truth.best_convention_mean_relative_error"},
              "best_convention_mean_relative_error > 0.001");
    } else if (metric == "l2" &&
               s.String("ground_truth.best_matching_distance_convention") == "squared euclidean") {
        s.Add("ground_truth.squared_distances",
              "notice",
              "Stored /distances are squared L2, while dataset_format.md documents euclidean as "
              "sqrt(L2Sqr).",
              {"ground_truth.best_matching_distance_convention"},
              "best convention == squared euclidean");
    }
    if (s.Number("ground_truth.out_of_range_ids", value) && value > 0) {
        s.Add("ground_truth.out_of_range_ids",
              "critical",
              format_number(value) + " stored neighbor ids fall outside /train.",
              {"ground_truth.out_of_range_ids"},
              "ground_truth.out_of_range_ids > 0");
    }
    if (s.Number("labels.query_selectivity.queries_without_matching_base_rows", value) &&
        value > 0) {
        s.Add("labels.empty_filters",
              "notice",
              format_number(value) +
                  " queries have a label that no base row carries; filtered search returns "
                  "nothing for them.",
              {"labels.query_selectivity.queries_without_matching_base_rows"},
              "queries_without_matching_base_rows > 0");
    }
    return s.Result();
}

}  // namespace vsag::dataset_profile
