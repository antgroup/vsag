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

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

#include "deterministic_random.h"
#include "kmeans.h"
#include "quantization_simulation.h"
#include "split_profiler.h"
#include "statistics.h"
#include "vector_math.h"

namespace vsag::dataset_profile {

namespace {

constexpr double UNIT_NORM_TOLERANCE = 1e-3;
constexpr double OUTLIER_SIGMA = 6.0;
constexpr double TUKEY_FAR_OUT = 3.0;
constexpr uint64_t SILHOUETTE_ROWS = 2000;
constexpr uint64_t CONTRAST_REFERENCE_ROWS = 1000;
constexpr uint64_t PCA_RATIO_ITEMS = 32;
constexpr uint64_t DUPLICATE_BYTES_PER_ROW = 24;

struct DimensionPartial {
    std::vector<Moments> moments;
    std::vector<float> minimum;
    std::vector<float> maximum;
    std::vector<uint64_t> zeros;
    std::vector<uint64_t> nans;
    std::vector<uint64_t> positive_infinities;
    std::vector<uint64_t> negative_infinities;

    explicit DimensionPartial(uint64_t dim = 0)
        : moments(dim),
          minimum(dim, std::numeric_limits<float>::infinity()),
          maximum(dim, -std::numeric_limits<float>::infinity()),
          zeros(dim, 0),
          nans(dim, 0),
          positive_infinities(dim, 0),
          negative_infinities(dim, 0) {
    }

    void
    Merge(const DimensionPartial& other) {
        for (uint64_t d = 0; d < moments.size(); ++d) {
            moments[d].Merge(other.moments[d]);
            minimum[d] = std::min(minimum[d], other.minimum[d]);
            maximum[d] = std::max(maximum[d], other.maximum[d]);
            zeros[d] += other.zeros[d];
            nans[d] += other.nans[d];
            positive_infinities[d] += other.positive_infinities[d];
            negative_infinities[d] += other.negative_infinities[d];
        }
    }
};

struct SecondPassPartial {
    Histogram values;
    std::vector<uint64_t> outliers;
};

std::string
elapsed_since(std::chrono::steady_clock::time_point start) {
    const auto seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.1fs", seconds);
    return buffer;
}

std::vector<double>
to_doubles(const std::vector<float>& values) {
    return {values.begin(), values.end()};
}

JsonType
indices_json(const std::vector<uint64_t>& indices) {
    JsonType array = JsonType::array();
    for (const auto index : indices) {
        array.push_back(index);
    }
    return array;
}

std::vector<uint64_t>
top_indices(const std::vector<double>& values, uint64_t count, bool largest) {
    std::vector<uint64_t> order(values.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&values, largest](uint64_t a, uint64_t b) {
        return largest ? values[a] > values[b] : values[a] < values[b];
    });
    order.resize(std::min<uint64_t>(count, order.size()));
    return order;
}

/// Tukey far-out fences with a spread floor, so unit-normalized data does not flag
/// rounding noise as outliers.
void
tukey_fences(const std::vector<double>& sorted, double& lower, double& upper) {
    const double q1 = quantile_sorted(sorted, 0.25);
    const double q3 = quantile_sorted(sorted, 0.75);
    const double median = quantile_sorted(sorted, 0.5);
    const double spread = std::max(q3 - q1, 1e-3 * std::abs(median));
    lower = q1 - TUKEY_FAR_OUT * spread;
    upper = q3 + TUKEY_FAR_OUT * spread;
}

}  // namespace

bool
streamed_value_range(const std::vector<float>& sample,
                     double minimum,
                     double maximum,
                     double& lower,
                     double& upper) {
    constexpr uint64_t max_values = 1000000;
    const uint64_t stride = std::max<uint64_t>(1, sample.size() / max_values);
    std::vector<double> values;
    values.reserve(std::min<uint64_t>(sample.size(), max_values + 1));
    for (uint64_t i = 0; i < sample.size(); i += stride) {
        if (is_finite_value(sample[i])) {
            values.push_back(sample[i]);
        }
    }
    std::sort(values.begin(), values.end());
    return robust_range(values, minimum, maximum, lower, upper);
}

JsonType
sample_json(const SamplePlan& plan, uint64_t used, const std::string& subset) {
    return JsonType{{"method", "uniform without replacement (Floyd), seeded shuffle prefix"},
                    {"population", plan.population},
                    {"sampled_rows", plan.rows.size()},
                    {"valid_sampled_rows", plan.analysis_order.size()},
                    {"used_rows", used},
                    {"subset", subset}};
}

DenseSplitProfiler::DenseSplitProfiler(SplitContext context)
    : context_(std::move(context)), options_(*context_.options) {
    const SplitInfo& split = context_.info->Split(context_.name);
    rows_ = split.rows;
    dim_ = split.dim;
    const uint64_t bytes_per_row = std::max<uint64_t>(dim_ * sizeof(float), 1);
    const uint64_t target_rows = options_.chunk_mb * 1024ULL * 1024ULL / bytes_per_row;
    chunk_rows_ = std::max<uint64_t>(1, target_rows / BLOCK_ROWS) * BLOCK_ROWS;
}

void
DenseSplitProfiler::Run(const DenseQuerySet* cross_queries, RowDistanceProbe* probe) {
    probe_ = probe;
    prepare_sample();
    first_pass();
    second_pass(cross_queries);
    finalize();
}

void
DenseSplitProfiler::prepare_sample() {
    plan_.population = rows_;
    DeterministicRng rng(derive_seed(options_.seed, "sample/" + context_.name));
    plan_.rows = sample_without_replacement(rows_, options_.sample_size, rng);
    sample_data_.assign(plan_.rows.size() * dim_, 0.0F);
    row_norms_.assign(rows_, 0.0);
    row_nonzeros_.assign(rows_, 0);
    row_valid_.assign(rows_, 0);
    full_duplicates_ =
        options_.Enabled(SECTION_DUPLICATES) &&
        rows_ * DUPLICATE_BYTES_PER_ROW <= options_.duplicate_memory_mb * 1024ULL * 1024ULL;
    if (full_duplicates_) {
        hash_high_.assign(rows_, 0);
        hash_low_.assign(rows_, 0);
    }
}

void
DenseSplitProfiler::first_pass() {
    const auto start = std::chrono::steady_clock::now();
    DenseSplitReader reader(context_.info->path, context_.name);
    std::vector<float> buffer(chunk_rows_ * dim_);
    DimensionPartial total(dim_);
    const uint64_t dim = dim_;
    const bool full_duplicates = full_duplicates_;
    std::vector<uint8_t>& row_valid = row_valid_;
    std::vector<double>& row_norms = row_norms_;
    std::vector<uint32_t>& row_nonzeros = row_nonzeros_;
    std::vector<uint64_t>& hash_high = hash_high_;
    std::vector<uint64_t>& hash_low = hash_low_;
    for (uint64_t begin = 0; begin < rows_; begin += chunk_rows_) {
        const uint64_t count = std::min(chunk_rows_, rows_ - begin);
        reader.ReadRows(begin, count, buffer.data());
        const uint64_t blocks = (count + BLOCK_ROWS - 1) / BLOCK_ROWS;
        std::vector<DimensionPartial> partials(blocks, DimensionPartial(dim_));
        const auto signed_blocks = static_cast<int64_t>(blocks);
        // clang-format off
#pragma omp parallel for schedule(dynamic, 1) default(none) \
    shared(buffer, count, begin, partials, signed_blocks, dim, row_valid, row_norms, \
           row_nonzeros, full_duplicates, hash_high, hash_low)
        // clang-format on
        for (int64_t b = 0; b < signed_blocks; ++b) {
            const uint64_t block_begin = static_cast<uint64_t>(b) * BLOCK_ROWS;
            const uint64_t block_end = std::min(count, block_begin + BLOCK_ROWS);
            DimensionPartial& partial = partials[b];
            std::vector<double> sum(dim, 0.0);
            std::vector<uint64_t> finite(dim, 0);
            std::vector<uint32_t> words(dim);
            for (uint64_t r = block_begin; r < block_end; ++r) {
                const float* row = buffer.data() + r * dim;
                const uint64_t global_row = begin + r;
                bool valid = true;
                double norm = 0.0;
                uint32_t nonzeros = 0;
                for (uint64_t d = 0; d < dim; ++d) {
                    const float value = row[d];
                    if (is_finite_value(value)) {
                        sum[d] += value;
                        ++finite[d];
                        partial.minimum[d] = std::min(partial.minimum[d], value);
                        partial.maximum[d] = std::max(partial.maximum[d], value);
                        if (value == 0.0F) {
                            ++partial.zeros[d];
                        } else {
                            ++nonzeros;
                        }
                        norm += static_cast<double>(value) * static_cast<double>(value);
                    } else {
                        valid = false;
                        if (is_nan_value(value)) {
                            ++partial.nans[d];
                        } else if (value > 0.0F) {
                            ++partial.positive_infinities[d];
                        } else {
                            ++partial.negative_infinities[d];
                        }
                    }
                    words[d] = canonical_bits(value);
                }
                row_valid[global_row] = valid ? 1 : 0;
                row_norms[global_row] =
                    valid ? std::sqrt(norm) : std::numeric_limits<double>::quiet_NaN();
                row_nonzeros[global_row] = nonzeros;
                if (full_duplicates) {
                    hash_words(words.data(), dim, hash_high[global_row], hash_low[global_row]);
                }
            }
            std::vector<double> mean(dim, 0.0);
            for (uint64_t d = 0; d < dim; ++d) {
                mean[d] = finite[d] > 0 ? sum[d] / static_cast<double>(finite[d]) : 0.0;
            }
            std::vector<double> m2(dim, 0.0);
            std::vector<double> m3(dim, 0.0);
            std::vector<double> m4(dim, 0.0);
            for (uint64_t r = block_begin; r < block_end; ++r) {
                const float* row = buffer.data() + r * dim;
                for (uint64_t d = 0; d < dim; ++d) {
                    if (is_finite_value(row[d])) {
                        const double centered = static_cast<double>(row[d]) - mean[d];
                        const double squared = centered * centered;
                        m2[d] += squared;
                        m3[d] += squared * centered;
                        m4[d] += squared * squared;
                    }
                }
            }
            for (uint64_t d = 0; d < dim; ++d) {
                partial.moments[d] = Moments{finite[d], mean[d], m2[d], m3[d], m4[d]};
            }
        }
        for (const auto& partial : partials) {
            total.Merge(partial);
        }
        // Capture the sampled rows that fall into this chunk.
        auto it = std::lower_bound(plan_.rows.begin(), plan_.rows.end(), begin);
        for (; it != plan_.rows.end() && *it < begin + count; ++it) {
            const auto position = static_cast<uint64_t>(it - plan_.rows.begin());
            std::copy_n(
                buffer.data() + (*it - begin) * dim_, dim_, sample_data_.data() + position * dim_);
        }
    }

    // Per-dimension statistics.
    dim_mean_.assign(dim_, 0.0);
    dim_std_.assign(dim_, 0.0);
    std::vector<double> minimum(dim_);
    std::vector<double> maximum(dim_);
    std::vector<double> zero_ratio(dim_);
    std::vector<double> skewness(dim_);
    std::vector<double> kurtosis(dim_);
    std::vector<double> ranges;
    Moments all_values;
    uint64_t nans = 0;
    uint64_t positive_infinities = 0;
    uint64_t negative_infinities = 0;
    uint64_t zeros = 0;
    uint64_t constant_dimensions = 0;
    uint64_t always_zero_dimensions = 0;
    uint64_t heavy_tailed = 0;
    double absolute_skew = 0.0;
    global_min_ = std::numeric_limits<double>::infinity();
    global_max_ = -std::numeric_limits<double>::infinity();
    for (uint64_t d = 0; d < dim_; ++d) {
        const Moments& m = total.moments[d];
        all_values.Merge(m);
        dim_mean_[d] = m.mean;
        dim_std_[d] = m.Std();
        const bool has_values = m.count > 0;
        minimum[d] = has_values ? total.minimum[d] : std::numeric_limits<double>::quiet_NaN();
        maximum[d] = has_values ? total.maximum[d] : std::numeric_limits<double>::quiet_NaN();
        zero_ratio[d] =
            safe_ratio(static_cast<double>(total.zeros[d]), static_cast<double>(m.count));
        skewness[d] = m.Skewness();
        kurtosis[d] = m.ExcessKurtosis();
        nans += total.nans[d];
        positive_infinities += total.positive_infinities[d];
        negative_infinities += total.negative_infinities[d];
        zeros += total.zeros[d];
        if (has_values) {
            global_min_ = std::min(global_min_, minimum[d]);
            global_max_ = std::max(global_max_, maximum[d]);
            ranges.push_back(maximum[d] - minimum[d]);
        }
        if (has_values && maximum[d] == minimum[d]) {
            ++constant_dimensions;
        }
        if (has_values && total.zeros[d] == m.count) {
            ++always_zero_dimensions;
        }
        if (kurtosis[d] > 3.0) {
            ++heavy_tailed;
        }
        absolute_skew += std::abs(skewness[d]);
    }
    if (!std::isfinite(global_min_)) {
        global_min_ = 0.0;
        global_max_ = 0.0;
    }
    value_clipped_ =
        streamed_value_range(sample_data_, global_min_, global_max_, value_low_, value_high_);

    uint64_t invalid_rows = 0;
    uint64_t zero_rows = 0;
    std::vector<double> finite_norms;
    finite_norms.reserve(rows_);
    std::vector<double> nonzero_counts;
    nonzero_counts.reserve(rows_);
    for (uint64_t r = 0; r < rows_; ++r) {
        if (row_valid_[r] == 0) {
            ++invalid_rows;
            continue;
        }
        finite_norms.push_back(row_norms_[r]);
        nonzero_counts.push_back(row_nonzeros_[r]);
        if (row_nonzeros_[r] == 0) {
            ++zero_rows;
        }
    }
    sorted_norms_.assign(finite_norms.begin(), finite_norms.end());
    std::sort(sorted_norms_.begin(), sorted_norms_.end());

    const auto total_values = static_cast<double>(rows_ * dim_);
    const uint64_t invalid_values = nans + positive_infinities + negative_infinities;
    json_["kind"] = "dense";
    json_["rows"] = rows_;
    json_["dim"] = dim_;
    json_["storage_type"] = context_.info->Split(context_.name).storage_type;
    json_["basic"] =
        JsonType{{"scope", "full"},
                 {"definition", "Row count, dimensionality and element type of the split."},
                 {"rows", rows_},
                 {"dim", dim_},
                 {"values", rows_ * dim_},
                 {"storage_type", context_.info->Split(context_.name).storage_type},
                 {"float32_bytes", rows_ * dim_ * sizeof(float)}};
    json_["invalid_values"] = JsonType{
        {"scope", "full"},
        {"definition",
         "NaN and infinite coordinates (IEEE-754 bit patterns). Rows containing any are excluded "
         "from norm and distance metrics."},
        {"nan", nans},
        {"positive_infinity", positive_infinities},
        {"negative_infinity", negative_infinities},
        {"total", invalid_values},
        {"ratio", safe_ratio(static_cast<double>(invalid_values), total_values)},
        {"rows_with_invalid", invalid_rows},
        {"row_ratio", safe_ratio(static_cast<double>(invalid_rows), static_cast<double>(rows_))}};

    std::vector<double> norms_sorted = finite_norms;
    JsonType norm_summary = summarize_values(norms_sorted);
    uint64_t unit_norm = 0;
    uint64_t zero_norm = 0;
    for (const double norm : norms_sorted) {
        unit_norm += std::abs(norm - 1.0) <= UNIT_NORM_TOLERANCE ? 1 : 0;
        zero_norm += norm == 0.0 ? 1 : 0;
    }
    double lower_fence = 0.0;
    double upper_fence = 0.0;
    JsonType norm_outliers = JsonType{{"method", "Tukey far-out fences: Q1 - 3*IQR, Q3 + 3*IQR"}};
    if (!norms_sorted.empty()) {
        tukey_fences(norms_sorted, lower_fence, upper_fence);
        uint64_t low = 0;
        uint64_t high = 0;
        std::vector<std::pair<double, uint64_t>> extremes;
        const double median = quantile_sorted(norms_sorted, 0.5);
        for (uint64_t r = 0; r < rows_; ++r) {
            if (row_valid_[r] == 0) {
                continue;
            }
            const double norm = row_norms_[r];
            if (norm < lower_fence || norm > upper_fence) {
                low += norm < lower_fence ? 1 : 0;
                high += norm > upper_fence ? 1 : 0;
                extremes.emplace_back(-std::abs(norm - median), r);
            }
        }
        std::sort(extremes.begin(), extremes.end());
        JsonType examples = JsonType::array();
        for (uint64_t i = 0; i < std::min<uint64_t>(5, extremes.size()); ++i) {
            examples.push_back(
                JsonType{{"row", extremes[i].second}, {"norm", row_norms_[extremes[i].second]}});
        }
        norm_outliers["lower_fence"] = lower_fence;
        norm_outliers["upper_fence"] = upper_fence;
        norm_outliers["low"] = low;
        norm_outliers["high"] = high;
        norm_outliers["ratio"] =
            safe_ratio(static_cast<double>(low + high), static_cast<double>(norms_sorted.size()));
        norm_outliers["examples"] = examples;
    }
    const double norm_mean = norm_summary.value("mean", 0.0);
    const double norm_std = norm_summary.value("std", 0.0);
    json_["norms"] = JsonType{
        {"scope", "full"},
        {"definition", "L2 norm of every row with finite values."},
        {"summary", norm_summary},
        {"histogram", histogram_of_sorted(norms_sorted, options_.histogram_bins)},
        {"coefficient_of_variation", safe_ratio(norm_std, norm_mean)},
        {"zero_norm_rows", zero_norm},
        {"unit_norm_ratio",
         safe_ratio(static_cast<double>(unit_norm), static_cast<double>(norms_sorted.size()))},
        {"unit_norm_tolerance", UNIT_NORM_TOLERANCE},
        {"outliers", norm_outliers}};

    // Mean vector relative to the average norm: close to 1 means rows share one direction.
    double mean_vector_norm = 0.0;
    double total_variance = 0.0;
    for (uint64_t d = 0; d < dim_; ++d) {
        mean_vector_norm += dim_mean_[d] * dim_mean_[d];
        total_variance += dim_std_[d] * dim_std_[d];
    }
    mean_vector_norm = std::sqrt(mean_vector_norm);
    std::vector<double> variances(dim_);
    for (uint64_t d = 0; d < dim_; ++d) {
        variances[d] = dim_std_[d] * dim_std_[d];
    }
    std::vector<double> sorted_variances = variances;
    std::sort(sorted_variances.begin(), sorted_variances.end(), std::greater<>());
    double top_share = 0.0;
    const uint64_t top_count = std::max<uint64_t>(1, dim_ / 10);
    for (uint64_t i = 0; i < top_count && i < sorted_variances.size(); ++i) {
        top_share += sorted_variances[i];
    }
    std::vector<double> stds = dim_std_;
    std::vector<double> nonconstant_stds;
    for (const double value : stds) {
        if (value > 0.0) {
            nonconstant_stds.push_back(value);
        }
    }
    std::sort(nonconstant_stds.begin(), nonconstant_stds.end());
    std::sort(ranges.begin(), ranges.end());
    const double median_std = quantile_sorted(nonconstant_stds, 0.5);
    const double median_range = quantile_sorted(ranges, 0.5);
    JsonType std_summary = summarize_values(stds);
    json_["dimensions"] = JsonType{
        {"scope", "full"},
        {"definition",
         "Per-dimension moments over finite values (population std; skewness g1; excess "
         "kurtosis g2 - 3)."},
        {"summary",
         JsonType{{"constant_dimensions", constant_dimensions},
                  {"always_zero_dimensions", always_zero_dimensions},
                  {"std", std_summary},
                  {"std_ratio_max_to_median",
                   json_number(nonconstant_stds.empty()
                                   ? 0.0
                                   : safe_ratio(nonconstant_stds.back(), median_std))},
                  {"range_ratio_max_to_median",
                   json_number(ranges.empty() ? 0.0 : safe_ratio(ranges.back(), median_range))},
                  {"variance_share_top_10pct_dimensions", safe_ratio(top_share, total_variance)},
                  {"mean_abs_skewness", safe_ratio(absolute_skew, static_cast<double>(dim_))},
                  {"heavy_tailed_dimension_ratio",
                   safe_ratio(static_cast<double>(heavy_tailed), static_cast<double>(dim_))},
                  {"top_variance_dimensions", indices_json(top_indices(variances, 5, true))},
                  {"lowest_variance_dimensions", indices_json(top_indices(variances, 5, false))},
                  {"mean_vector_norm_ratio", safe_ratio(mean_vector_norm, norm_mean)}}},
        {"per_dimension",
         JsonType{{"mean", dim_mean_},
                  {"std", dim_std_},
                  {"min", minimum},
                  {"max", maximum},
                  {"zero_ratio", zero_ratio},
                  {"skewness", skewness},
                  {"excess_kurtosis", kurtosis}}}};
    for (auto* array : {&json_["dimensions"]["per_dimension"]["min"],
                        &json_["dimensions"]["per_dimension"]["max"]}) {
        for (auto& value : *array) {
            if (value.is_number_float() && !std::isfinite(value.get<double>())) {
                value = nullptr;
            }
        }
    }

    std::vector<double> nonzero_sorted = nonzero_counts;
    std::sort(nonzero_sorted.begin(), nonzero_sorted.end());
    const uint64_t finite_values = all_values.count;
    json_["sparsity"] = JsonType{
        {"scope", "full"},
        {"definition", "Exact zeros among finite coordinates and non-zeros per valid row."},
        {"zero_ratio", safe_ratio(static_cast<double>(zeros), static_cast<double>(finite_values))},
        {"zero_rows", zero_rows},
        {"always_zero_dimensions", always_zero_dimensions},
        {"nonzeros_per_row",
         JsonType{{"summary", summarize_values(nonzero_counts)},
                  {"histogram", integer_histogram(nonzero_sorted, options_.histogram_bins)}}}};
    json_["values"] = JsonType{{"scope", "full"},
                               {"definition", "Distribution of every finite coordinate."},
                               {"count", finite_values},
                               {"min", json_number(global_min_)},
                               {"max", json_number(global_max_)},
                               {"mean", json_number(all_values.mean)},
                               {"std", json_number(all_values.Std())},
                               {"skewness", json_number(all_values.Skewness())},
                               {"excess_kurtosis", json_number(all_values.ExcessKurtosis())}};

    // Exact duplicates.
    if (options_.Enabled(SECTION_DUPLICATES)) {
        if (full_duplicates_) {
            std::vector<uint64_t> ids(rows_);
            std::iota(ids.begin(), ids.end(), 0);
            JsonType report = duplicate_report(hash_high_, hash_low_, ids, rows_);
            report["scope"] = "full";
            json_["duplicates"] = report;
            hash_high_.clear();
            hash_high_.shrink_to_fit();
            hash_low_.clear();
            hash_low_.shrink_to_fit();
        } else {
            std::vector<uint64_t> high(plan_.rows.size());
            std::vector<uint64_t> low(plan_.rows.size());
            std::vector<uint32_t> words(dim_);
            for (uint64_t i = 0; i < plan_.rows.size(); ++i) {
                for (uint64_t d = 0; d < dim_; ++d) {
                    words[d] = canonical_bits(sample_data_[i * dim_ + d]);
                }
                hash_words(words.data(), dim_, high[i], low[i]);
            }
            JsonType report = duplicate_report(high, low, plan_.rows, plan_.rows.size());
            report["scope"] = "sample";
            report["reason"] = "row count exceeds --duplicate-memory-mb";
            report["sample"] = sample_json(plan_, plan_.rows.size(), "all sampled rows");
            json_["duplicates"] = report;
        }
    }

    // Seeded analysis order over valid sampled rows.
    std::vector<uint64_t> order(plan_.rows.size());
    std::iota(order.begin(), order.end(), 0);
    DeterministicRng order_rng(derive_seed(options_.seed, "order/" + context_.name));
    shuffle_in_place(order, order_rng);
    plan_.analysis_order.clear();
    for (const uint64_t position : order) {
        const uint64_t row = plan_.rows[position];
        const bool usable =
            row_valid_[row] != 0 && (context_.metric != Metric::COSINE || row_norms_[row] > 0.0);
        if (usable) {
            plan_.analysis_order.push_back(position);
        }
    }
    if (context_.progress) {
        context_.progress("[" + context_.name + "] pass 1/2 (full scan) done in " +
                          elapsed_since(start));
    }
}

const float*
DenseSplitProfiler::sample_row(uint64_t analysis_position) const {
    return sample_data_.data() + plan_.analysis_order[analysis_position] * dim_;
}

DenseQuerySet
DenseSplitProfiler::MakeQuerySet(uint64_t count) const {
    DenseQuerySet queries;
    queries.dim = dim_;
    count = std::min<uint64_t>(count, plan_.analysis_order.size());
    queries.vectors.resize(count * dim_);
    for (uint64_t i = 0; i < count; ++i) {
        std::copy_n(sample_row(i), dim_, queries.vectors.data() + i * dim_);
        queries.norms.push_back(std::sqrt(squared_norm(sample_row(i), dim_)));
        queries.self_rows.push_back(NO_ROW);
    }
    return queries;
}

std::vector<uint64_t>
DenseSplitProfiler::AnalysisRowIds(uint64_t count) const {
    count = std::min<uint64_t>(count, plan_.analysis_order.size());
    std::vector<uint64_t> ids(count);
    for (uint64_t i = 0; i < count; ++i) {
        ids[i] = plan_.rows[plan_.analysis_order[i]];
    }
    return ids;
}

std::vector<float>
DenseSplitProfiler::AnalysisRows(uint64_t count) const {
    count = std::min<uint64_t>(count, plan_.analysis_order.size());
    std::vector<float> rows(count * dim_);
    for (uint64_t i = 0; i < count; ++i) {
        const float* source = sample_row(i);
        float* target = rows.data() + i * dim_;
        std::copy_n(source, dim_, target);
        if (context_.metric == Metric::COSINE) {
            const double norm = std::sqrt(squared_norm(source, dim_));
            for (uint64_t d = 0; d < dim_; ++d) {
                target[d] = static_cast<float>(target[d] / norm);
            }
        }
    }
    return rows;
}

void
DenseSplitProfiler::second_pass(const DenseQuerySet* cross_queries) {
    const auto start = std::chrono::steady_clock::now();
    const bool run_knn = options_.Enabled(SECTION_KNN);
    DenseQuerySet self_queries = MakeQuerySet(run_knn ? options_.knn_queries : 0);
    for (uint64_t i = 0; i < self_queries.Size(); ++i) {
        self_queries.self_rows[i] = plan_.rows[plan_.analysis_order[i]];
    }
    const uint64_t k = std::min<uint64_t>(options_.knn_k, rows_ > 1 ? rows_ - 1 : 1);
    DenseKnnSearch self_search(self_queries, context_.metric, k);
    DenseQuerySet empty;
    empty.dim = dim_;
    const DenseQuerySet& cross = cross_queries != nullptr ? *cross_queries : empty;
    DenseKnnSearch cross_search(cross, context_.metric, std::min<uint64_t>(options_.knn_k, rows_));

    Histogram value_histogram(value_low_, value_high_, options_.histogram_bins);
    std::vector<uint64_t> outliers(dim_, 0);
    DenseSplitReader reader(context_.info->path, context_.name);
    std::vector<float> buffer(chunk_rows_ * dim_);
    std::vector<double> norms(chunk_rows_);
    std::vector<uint8_t> valid(chunk_rows_);
    const double value_low = value_low_;
    const double value_high = value_high_;
    const uint64_t bins = options_.histogram_bins;
    const uint64_t dim = dim_;
    const Metric metric = context_.metric;
    const std::vector<double>& dim_std = dim_std_;
    const std::vector<double>& dim_mean = dim_mean_;
    const std::vector<uint8_t>& row_valid = row_valid_;
    for (uint64_t begin = 0; begin < rows_; begin += chunk_rows_) {
        const uint64_t count = std::min(chunk_rows_, rows_ - begin);
        reader.ReadRows(begin, count, buffer.data());
        const uint64_t blocks = (count + BLOCK_ROWS - 1) / BLOCK_ROWS;
        std::vector<SecondPassPartial> partials(blocks);
        const auto signed_blocks = static_cast<int64_t>(blocks);
        // clang-format off
#pragma omp parallel for schedule(dynamic, 1) default(none) \
    shared(buffer, count, begin, partials, signed_blocks, norms, valid, value_low, value_high, \
           bins, dim, dim_std, dim_mean, row_valid, metric)
        // clang-format on
        for (int64_t b = 0; b < signed_blocks; ++b) {
            const uint64_t block_begin = static_cast<uint64_t>(b) * BLOCK_ROWS;
            const uint64_t block_end = std::min(count, block_begin + BLOCK_ROWS);
            SecondPassPartial& partial = partials[b];
            partial.values = Histogram(value_low, value_high, bins);
            partial.outliers.assign(dim, 0);
            for (uint64_t r = block_begin; r < block_end; ++r) {
                const float* row = buffer.data() + r * dim;
                for (uint64_t d = 0; d < dim; ++d) {
                    const float value = row[d];
                    if (!is_finite_value(value)) {
                        continue;
                    }
                    partial.values.Add(value);
                    if (dim_std[d] > 0.0 && std::abs(static_cast<double>(value) - dim_mean[d]) >
                                                OUTLIER_SIGMA * dim_std[d]) {
                        ++partial.outliers[d];
                    }
                }
                const uint64_t global_row = begin + r;
                norms[r] = std::sqrt(squared_norm(row, dim));
                valid[r] =
                    row_valid[global_row] != 0 && (metric != Metric::COSINE || norms[r] > 0.0) ? 1
                                                                                               : 0;
            }
        }
        for (const auto& partial : partials) {
            value_histogram.Merge(partial.values);
            for (uint64_t d = 0; d < dim_; ++d) {
                outliers[d] += partial.outliers[d];
            }
        }
        if (self_queries.Size() > 0) {
            self_search.Scan(buffer.data(), norms.data(), valid.data(), begin, count);
        }
        if (cross.Size() > 0) {
            cross_search.Scan(buffer.data(), norms.data(), valid.data(), begin, count);
        }
        if (probe_ != nullptr && cross.Size() > 0) {
            auto it = std::lower_bound(probe_->rows.begin(), probe_->rows.end(), begin);
            for (; it != probe_->rows.end() && *it < begin + count; ++it) {
                const auto entry = static_cast<uint64_t>(it - probe_->rows.begin());
                const uint64_t r = *it - begin;
                if (valid[r] != 0) {
                    probe_->distances[entry] = dense_metric_distance(
                        cross, probe_->queries[entry], buffer.data() + r * dim_, norms[r], metric);
                }
            }
        }
    }
    uint64_t total_outliers = 0;
    std::vector<double> outlier_ratio(dim_);
    const uint64_t finite_values = json_["values"]["count"].get<uint64_t>();
    for (uint64_t d = 0; d < dim_; ++d) {
        total_outliers += outliers[d];
        outlier_ratio[d] = safe_ratio(static_cast<double>(outliers[d]), static_cast<double>(rows_));
    }
    json_["values"]["histogram"] = value_histogram.ToJson();
    json_["values"]["histogram"]["clipped"] = value_clipped_;
    json_["values"]["outliers_6_sigma"] = JsonType{
        {"definition", "Coordinates with |x - mean_d| > 6 * std_d of their dimension."},
        {"count", total_outliers},
        {"ratio",
         safe_ratio(static_cast<double>(total_outliers), static_cast<double>(finite_values))},
        {"top_dimensions", indices_json(top_indices(outlier_ratio, 5, true))}};
    json_["dimensions"]["per_dimension"]["outlier_ratio_6_sigma"] = outlier_ratio;

    self_neighbors_ = self_search.MetricNeighbors();
    self_geometric_ = self_search.GeometricDistances();
    cross_neighbors_ = cross_search.MetricNeighbors();
    cross_geometric_ = cross_search.GeometricDistances();
    if (context_.progress) {
        context_.progress("[" + context_.name + "] pass 2/2 (histograms, outliers, exact k-NN of " +
                          std::to_string(self_queries.Size() + cross.Size()) +
                          " queries) done in " + elapsed_since(start));
    }
}

void
DenseSplitProfiler::finalize() {
    const auto start = std::chrono::steady_clock::now();
    const uint64_t valid_rows = plan_.analysis_order.size();
    const std::string& name = context_.name;
    const Metric metric = context_.metric;
    std::vector<double> sample_norms(valid_rows);
    for (uint64_t i = 0; i < valid_rows; ++i) {
        sample_norms[i] = std::sqrt(squared_norm(sample_row(i), dim_));
    }
    const uint64_t dim = dim_;
    std::vector<const float*> sample_rows(valid_rows);
    for (uint64_t i = 0; i < valid_rows; ++i) {
        sample_rows[i] = sample_row(i);
    }
    const std::vector<std::vector<double>>& self_geometric = self_geometric_;

    // Pairwise distances between random sampled pairs.
    std::vector<double> pair_metric;
    std::vector<double> pair_geometric;
    std::vector<double> pair_cosine;
    double median_pair_geometric = 0.0;
    if (valid_rows >= 2) {
        DeterministicRng pair_rng(derive_seed(options_.seed, "pairs/" + name));
        const uint64_t pairs = options_.pair_count;
        std::vector<std::pair<uint64_t, uint64_t>> indices(pairs);
        for (auto& pair : indices) {
            const uint64_t a = pair_rng.Uniform(valid_rows);
            uint64_t b = pair_rng.Uniform(valid_rows - 1);
            b += b >= a ? 1 : 0;
            pair = {a, b};
        }
        pair_metric.resize(pairs);
        pair_geometric.resize(pairs);
        pair_cosine.resize(pairs);
        const auto signed_pairs = static_cast<int64_t>(pairs);
        // clang-format off
#pragma omp parallel for schedule(static) default(none) \
    shared(indices, sample_rows, sample_norms, pair_cosine, pair_metric, pair_geometric, metric, \
           dim, signed_pairs)
        // clang-format on
        for (int64_t p = 0; p < signed_pairs; ++p) {
            const auto [a, b] = indices[p];
            const float* x = sample_rows[a];
            const float* y = sample_rows[b];
            double ip = 0.0;
            double l2 = 0.0;
            inner_product_and_l2_sqr(x, y, dim, ip, l2);
            const double denominator = sample_norms[a] * sample_norms[b];
            const double cosine = denominator > 0.0 ? ip / denominator : 0.0;
            pair_cosine[p] = cosine;
            if (metric == Metric::L2) {
                pair_metric[p] = l2;
                pair_geometric[p] = std::sqrt(l2);
            } else if (metric == Metric::IP) {
                pair_metric[p] = 1.0 - ip;
                pair_geometric[p] = std::sqrt(l2);
            } else {
                pair_metric[p] = 1.0 - cosine;
                pair_geometric[p] = std::sqrt(std::max(0.0, 2.0 * (1.0 - cosine)));
            }
        }
        std::vector<double> geometric_sorted = pair_geometric;
        JsonType geometric_summary = summarize_values(geometric_sorted);
        median_pair_geometric = quantile_sorted(geometric_sorted, 0.5);
        std::vector<double> metric_sorted = pair_metric;
        JsonType metric_summary = summarize_values(metric_sorted);
        std::vector<double> cosine_sorted = pair_cosine;
        JsonType cosine_summary = summarize_values(cosine_sorted);
        json_["pairwise_distances"] = JsonType{
            {"scope", "sample"},
            {"definition",
             "Distances between uniformly drawn pairs of distinct sampled rows; metric distance "
             "uses the VSAG convention, geometric distance is Euclidean (chord for cosine)."},
            {"sample", sample_json(plan_, valid_rows, "all valid sampled rows")},
            {"pairs", pairs},
            {"metric_distance",
             JsonType{{"summary", metric_summary},
                      {"histogram", histogram_of_sorted(metric_sorted, options_.histogram_bins)}}},
            {"geometric_distance",
             JsonType{
                 {"summary", geometric_summary},
                 {"histogram", histogram_of_sorted(geometric_sorted, options_.histogram_bins)}}},
            {"concentration_ratio",
             json_number(safe_ratio(geometric_summary.value("std", 0.0),
                                    geometric_summary.value("mean", 0.0)))},
            {"cosine_similarity",
             JsonType{{"summary", cosine_summary},
                      {"histogram", histogram_of_sorted(cosine_sorted, options_.histogram_bins)}}}};
        json_["anisotropy"] = JsonType{
            {"scope", "mixed"},
            {"definition",
             "Directional concentration: mean pairwise cosine similarity (sample), mean-vector "
             "norm over the average row norm (full)."},
            {"average_cosine_similarity",
             JsonType{{"scope", "sample"},
                      {"value", json_number(cosine_summary.value("mean", 0.0))}}},
            {"mean_vector_norm_ratio",
             JsonType{{"scope", "full"},
                      {"value", json_["dimensions"]["summary"]["mean_vector_norm_ratio"]}}}};
    }

    // Exact k-NN of the sampled queries against the full split.
    if (!self_neighbors_.empty()) {
        const uint64_t queries = self_neighbors_.size();
        std::vector<double> first_metric;
        std::vector<double> kth_metric;
        std::vector<double> first_geometric;
        std::vector<double> lids;
        std::vector<double> ratios;
        uint64_t tied_queries = 0;
        uint64_t exact_duplicates = 0;
        uint64_t near_duplicates = 0;
        const double near_threshold = options_.near_duplicate_threshold * median_pair_geometric;
        for (uint64_t q = 0; q < queries; ++q) {
            if (self_neighbors_[q].empty()) {
                continue;
            }
            first_metric.push_back(self_neighbors_[q].front().distance);
            kth_metric.push_back(self_neighbors_[q].back().distance);
            const auto& geometric = self_geometric_[q];
            first_geometric.push_back(geometric.front());
            exact_duplicates += geometric.front() == 0.0 ? 1 : 0;
            near_duplicates += geometric.front() <= near_threshold ? 1 : 0;
            lids.push_back(lid_mle(geometric));
            if (geometric.size() >= 2 && geometric[0] > 0.0) {
                ratios.push_back(geometric[1] / geometric[0]);
            }
            if (geometric.size() >= 2 && (geometric[0] == 0.0 || geometric[1] == geometric[0])) {
                ++tied_queries;
            }
        }
        self_first_geometric_ = first_geometric;
        std::vector<double> first_sorted = first_geometric;
        std::sort(first_sorted.begin(), first_sorted.end());
        double isolated_fence_low = 0.0;
        double isolated_fence = 0.0;
        tukey_fences(first_sorted, isolated_fence_low, isolated_fence);
        uint64_t isolated = 0;
        for (const double distance : first_geometric) {
            isolated += distance > isolated_fence ? 1 : 0;
        }
        const auto searched = static_cast<double>(first_metric.size());
        std::vector<double> first_metric_sorted = first_metric;
        JsonType first_summary = summarize_values(first_metric_sorted);
        std::vector<double> kth_sorted = kth_metric;
        JsonType kth_summary = summarize_values(kth_sorted);
        json_["nearest_neighbors"] = JsonType{
            {"scope", "sample"},
            {"definition",
             "Exact k nearest neighbors of sampled query rows searched in the full split "
             "(the query row itself excluded), VSAG metric convention."},
            {"sample", sample_json(plan_, queries, "first knn_queries valid sampled rows")},
            {"k", self_neighbors_.front().size()},
            {"searched_rows", rows_},
            {"first_neighbor_distance",
             JsonType{
                 {"summary", first_summary},
                 {"histogram", histogram_of_sorted(first_metric_sorted, options_.histogram_bins)}}},
            {"kth_neighbor_distance",
             JsonType{{"summary", kth_summary},
                      {"histogram", histogram_of_sorted(kth_sorted, options_.histogram_bins)}}},
            {"exact_duplicate_neighbor_ratio",
             safe_ratio(static_cast<double>(exact_duplicates), searched)},
            {"near_duplicate_ratio", safe_ratio(static_cast<double>(near_duplicates), searched)},
            {"near_duplicate_threshold",
             JsonType{{"relative_to_median_pairwise_distance", options_.near_duplicate_threshold},
                      {"geometric_distance", near_threshold}}},
            {"isolated_ratio", safe_ratio(static_cast<double>(isolated), searched)},
            {"isolated_definition",
             "First-neighbor geometric distance above the Tukey far-out fence of all queries."}};

        std::vector<double> lid_sorted = lids;
        JsonType lid_summary = summarize_values(lid_sorted);
        json_["intrinsic_dimensionality"] = JsonType{
            {"scope", "sample"},
            {"definition",
             "Local intrinsic dimensionality: Levina-Bickel MLE from the k geometric neighbor "
             "distances of each query; TwoNN (Facco et al.) from first/second neighbor ratios."},
            {"sample", sample_json(plan_, queries, "first knn_queries valid sampled rows")},
            {"lid_mle",
             JsonType{{"k", self_neighbors_.front().size()},
                      {"summary", lid_summary},
                      {"histogram", histogram_of_sorted(lid_sorted, options_.histogram_bins)}}},
            {"two_nn", json_number(two_nn_dimension(ratios))},
            {"two_nn_excluded_ratio", safe_ratio(static_cast<double>(tied_queries), searched)},
            {"two_nn_excluded_definition",
             "Queries whose first neighbor is a duplicate or ties the second neighbor; TwoNN skips "
             "them."},
            {"lid_to_dim_ratio",
             json_number(safe_ratio(lid_summary.value("p50", 0.0), static_cast<double>(dim_)))}};

        // Relative contrast: mean distance to random sampled rows over the first-neighbor
        // distance (He, Kumar and Chang, ICML 2012).
        const uint64_t references = std::min<uint64_t>(CONTRAST_REFERENCE_ROWS, valid_rows);
        std::vector<double> contrast(queries, std::numeric_limits<double>::quiet_NaN());
        const auto signed_queries = static_cast<int64_t>(queries);
        // clang-format off
#pragma omp parallel for schedule(dynamic, 8) default(none) \
    shared(self_geometric, sample_rows, sample_norms, references, metric, dim, contrast, \
           signed_queries)
        // clang-format on
        for (int64_t q = 0; q < signed_queries; ++q) {
            if (self_geometric[q].empty() || self_geometric[q].front() <= 0.0) {
                continue;
            }
            const float* query = sample_rows[q];
            double sum = 0.0;
            uint64_t count = 0;
            for (uint64_t r = 0; r < references; ++r) {
                if (r == static_cast<uint64_t>(q)) {
                    continue;
                }
                sum += geometric_distance(
                    metric, query, sample_rows[r], dim, sample_norms[q], sample_norms[r]);
                ++count;
            }
            if (count > 0) {
                contrast[q] = sum / static_cast<double>(count) / self_geometric[q].front();
            }
        }
        std::vector<double> contrast_sorted = contrast;
        JsonType contrast_summary = summarize_values(contrast_sorted);
        json_["relative_contrast"] = JsonType{
            {"scope", "sample"},
            {"definition",
             "Per query: mean geometric distance to reference sampled rows divided by the "
             "first-neighbor distance. Lower values mean harder nearest-neighbor search."},
            {"sample", sample_json(plan_, queries, "first knn_queries valid sampled rows")},
            {"reference_rows", references},
            {"summary", contrast_summary},
            {"histogram", histogram_of_sorted(contrast_sorted, options_.histogram_bins)}};
    }

    // Hubness of the within-sample k-NN graph.
    if (options_.Enabled(SECTION_HUBNESS) && valid_rows > options_.hubness_k) {
        const uint64_t hub_rows = std::min<uint64_t>(options_.hubness_sample, valid_rows);
        DenseQuerySet hub_queries;
        hub_queries.dim = dim_;
        hub_queries.vectors.resize(hub_rows * dim_);
        for (uint64_t i = 0; i < hub_rows; ++i) {
            std::copy_n(sample_row(i), dim_, hub_queries.vectors.data() + i * dim_);
            hub_queries.norms.push_back(sample_norms[i]);
            hub_queries.self_rows.push_back(i);
        }
        const uint64_t hub_k = std::min<uint64_t>(options_.hubness_k, hub_rows - 1);
        DenseKnnSearch hub_search(hub_queries, metric, hub_k);
        std::vector<uint8_t> all_valid(hub_rows, 1);
        hub_search.Scan(
            hub_queries.vectors.data(), hub_queries.norms.data(), all_valid.data(), 0, hub_rows);
        std::vector<double> occurrences(hub_rows, 0.0);
        for (const auto& neighbors : hub_search.MetricNeighbors()) {
            for (const auto& neighbor : neighbors) {
                occurrences[neighbor.index] += 1.0;
            }
        }
        Moments moments;
        uint64_t anti_hubs = 0;
        for (const double value : occurrences) {
            moments.Add(value);
            anti_hubs += value == 0.0 ? 1 : 0;
        }
        std::vector<double> sorted_occurrences = occurrences;
        std::sort(sorted_occurrences.begin(), sorted_occurrences.end(), std::greater<>());
        const uint64_t top = std::max<uint64_t>(1, hub_rows / 100);
        double top_sum = 0.0;
        for (uint64_t i = 0; i < top; ++i) {
            top_sum += sorted_occurrences[i];
        }
        std::vector<double> ascending(sorted_occurrences.rbegin(), sorted_occurrences.rend());
        json_["hubness"] = JsonType{
            {"scope", "sample"},
            {"definition",
             "k-occurrence N_k: how often each sampled row appears in the k-NN lists of the "
             "other sampled rows (Radovanovic et al., JMLR 2010). Skewness > 1 indicates hubs."},
            {"sample", sample_json(plan_, hub_rows, "first hubness_sample valid sampled rows")},
            {"k", hub_k},
            {"skewness", json_number(moments.Skewness())},
            {"max_occurrence", sorted_occurrences.front()},
            {"anti_hub_ratio",
             safe_ratio(static_cast<double>(anti_hubs), static_cast<double>(hub_rows))},
            {"top_1pct_occurrence_share",
             safe_ratio(top_sum, static_cast<double>(hub_rows * hub_k))},
            {"histogram", integer_histogram(ascending, options_.histogram_bins)}};
    }

    // PCA, k-means and the 2-D projection share the same prefix of the analysis order.
    const uint64_t structure_rows = valid_rows;
    std::vector<float> structure = AnalysisRows(structure_rows);
    const char* structure_input = metric == Metric::COSINE ? "l2-normalized rows" : "raw rows";
    if (options_.Enabled(SECTION_PCA) && structure_rows >= 3) {
        if (dim_ > options_.pca_max_dim) {
            json_["pca"] = JsonType{{"scope", "sample"},
                                    {"skipped", true},
                                    {"reason",
                                     "dim " + std::to_string(dim_) + " exceeds --pca-max-dim " +
                                         std::to_string(options_.pca_max_dim)}};
        } else {
            pca_ = compute_pca(structure, structure_rows, dim_);
            has_pca_ = true;
            double total = 0.0;
            double squares = 0.0;
            for (const double value : pca_.eigenvalues) {
                total += value;
                squares += value * value;
            }
            JsonType ratios = JsonType::array();
            JsonType cumulative = JsonType::array();
            double running = 0.0;
            double entropy = 0.0;
            for (uint64_t i = 0; i < pca_.eigenvalues.size(); ++i) {
                const double share = safe_ratio(pca_.eigenvalues[i], total);
                running += share;
                if (i < PCA_RATIO_ITEMS) {
                    ratios.push_back(share);
                    cumulative.push_back(running);
                }
                if (share > 0.0) {
                    entropy -= share * std::log(share);
                }
            }
            JsonType components = JsonType::object();
            for (const double threshold : {0.5, 0.8, 0.9, 0.95, 0.99}) {
                double sum = 0.0;
                uint64_t needed = pca_.eigenvalues.size();
                for (uint64_t i = 0; i < pca_.eigenvalues.size(); ++i) {
                    sum += safe_ratio(pca_.eigenvalues[i], total);
                    if (sum >= threshold - 1e-12) {
                        needed = i + 1;
                        break;
                    }
                }
                // Keys are percentages ("90") so that report paths contain no dots.
                components[std::to_string(std::lround(threshold * 100.0))] = needed;
            }
            json_["pca"] = JsonType{
                {"scope", "sample"},
                {"definition",
                 "Eigenvalues of the sample covariance (n - 1 denominator). Effective rank is "
                 "exp(entropy of the normalized spectrum); participation ratio is "
                 "(sum lambda)^2 / sum lambda^2."},
                {"sample", sample_json(plan_, structure_rows, "all valid sampled rows")},
                {"input", structure_input},
                {"explained_variance_ratio", ratios},
                {"cumulative_explained_variance", cumulative},
                {"components_for_variance", components},
                {"effective_rank", json_number(std::exp(entropy))},
                {"participation_ratio", json_number(safe_ratio(total * total, squares))},
                {"top1_ratio", json_number(safe_ratio(pca_.eigenvalues.front(), total))},
                {"total_variance", json_number(total)}};
            json_["anisotropy"]["pca_top1_ratio"] =
                JsonType{{"scope", "sample"}, {"value", json_["pca"]["top1_ratio"]}};
        }
    }

    std::vector<uint32_t> assignments;
    if (options_.Enabled(SECTION_CLUSTERING) && options_.clusters >= 2 &&
        structure_rows > options_.clusters) {
        const KMeansResult kmeans = run_kmeans(structure,
                                               structure_rows,
                                               dim_,
                                               options_.clusters,
                                               options_.kmeans_iterations,
                                               derive_seed(options_.seed, "kmeans/" + name));
        assignments = kmeans.assignments;
        std::vector<double> sizes(kmeans.sizes.begin(), kmeans.sizes.end());
        std::sort(sizes.begin(), sizes.end(), std::greater<>());
        Moments size_moments;
        double entropy = 0.0;
        for (const double size : sizes) {
            size_moments.Add(size);
            const double share = size / static_cast<double>(structure_rows);
            if (share > 0.0) {
                entropy -= share * std::log(share);
            }
        }
        const uint64_t silhouette_rows = std::min<uint64_t>(SILHOUETTE_ROWS, structure_rows);
        json_["clustering"] = JsonType{
            {"scope", "sample"},
            {"definition",
             "k-means (k-means++ seeding, Lloyd iterations) on sampled rows; balance and "
             "separation summarize coarse partition structure, e.g. for IVF bucket balance."},
            {"sample", sample_json(plan_, structure_rows, "all valid sampled rows")},
            {"input", structure_input},
            {"k", kmeans.k},
            {"iterations", kmeans.iterations},
            {"converged", kmeans.converged},
            {"inertia", json_number(kmeans.inertia)},
            {"sizes", sizes},
            {"size_cv", json_number(safe_ratio(size_moments.Std(), size_moments.mean))},
            {"max_to_mean_size", json_number(safe_ratio(sizes.front(), size_moments.mean))},
            {"normalized_entropy",
             json_number(kmeans.k > 1 ? entropy / std::log(static_cast<double>(kmeans.k)) : 0.0)},
            {"silhouette",
             json_number(
                 silhouette_score(structure, silhouette_rows, dim_, assignments, kmeans.k))},
            {"silhouette_rows", silhouette_rows}};
    }

    if (has_pca_ && options_.projection_points > 0) {
        const uint64_t points = std::min<uint64_t>(options_.projection_points, structure_rows);
        JsonType coordinates = JsonType::array();
        for (uint64_t i = 0; i < points; ++i) {
            double x = 0.0;
            double y = 0.0;
            project_2d(pca_, structure.data() + i * dim_, x, y);
            JsonType point = JsonType::array({round_significant(x, 6), round_significant(y, 6)});
            point.push_back(assignments.empty() ? 0 : assignments[i]);
            coordinates.push_back(point);
        }
        json_["projection"] = JsonType{
            {"scope", "sample"},
            {"definition",
             "Sampled rows projected on the two leading principal axes; the third value is the "
             "k-means cluster."},
            {"sample", sample_json(plan_, points, "first projection_points valid sampled rows")},
            {"points", coordinates}};
    }

    if (context_.progress) {
        context_.progress("[" + name + "] sample analyses done in " + elapsed_since(start));
    }
}

void
hash_words(const uint32_t* words, uint64_t count, uint64_t& high, uint64_t& low) {
    uint64_t h1 = 0x243F6A8885A308D3ULL ^ count;
    uint64_t h2 = 0x13198A2E03707344ULL + count;
    for (uint64_t i = 0; i < count; ++i) {
        const uint64_t word = words[i];
        h1 = (h1 ^ word) * 0x9E3779B97F4A7C15ULL;
        h1 = (h1 << 29U) | (h1 >> 35U);
        h2 = (h2 + word) * 0xC2B2AE3D27D4EB4FULL;
        h2 ^= h2 >> 31U;
    }
    auto finalize = [](uint64_t value) {
        value ^= value >> 33U;
        value *= 0xFF51AFD7ED558CCDULL;
        value ^= value >> 33U;
        value *= 0xC4CEB9FE1A85EC53ULL;
        value ^= value >> 33U;
        return value;
    };
    high = finalize(h1);
    low = finalize(h2);
}

JsonType
duplicate_report(const std::vector<uint64_t>& high,
                 const std::vector<uint64_t>& low,
                 const std::vector<uint64_t>& rows,
                 uint64_t population) {
    std::vector<uint64_t> order(high.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint64_t a, uint64_t b) {
        if (high[a] != high[b]) {
            return high[a] < high[b];
        }
        if (low[a] != low[b]) {
            return low[a] < low[b];
        }
        return rows[a] < rows[b];
    });
    uint64_t duplicate_rows = 0;
    uint64_t groups = 0;
    uint64_t largest = high.empty() ? 0 : 1;
    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> examples;
    for (uint64_t i = 0; i < order.size();) {
        uint64_t j = i + 1;
        while (j < order.size() && high[order[j]] == high[order[i]] &&
               low[order[j]] == low[order[i]]) {
            ++j;
        }
        const uint64_t size = j - i;
        if (size > 1) {
            duplicate_rows += size - 1;
            ++groups;
            largest = std::max(largest, size);
            std::vector<uint64_t> members;
            for (uint64_t m = i; m < j && members.size() < 5; ++m) {
                members.push_back(rows[order[m]]);
            }
            examples.emplace_back(size, members);
        }
        i = j;
    }
    std::sort(examples.begin(), examples.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first > b.first : a.second.front() < b.second.front();
    });
    JsonType example_json = JsonType::array();
    for (uint64_t i = 0; i < std::min<uint64_t>(5, examples.size()); ++i) {
        example_json.push_back(
            JsonType{{"size", examples[i].first}, {"rows", indices_json(examples[i].second)}});
    }
    return JsonType{
        {"definition",
         "Rows with identical canonical values (-0.0 == 0.0, one NaN), grouped by a 128-bit hash; "
         "the false-positive probability is about N^2 / 2^129."},
        {"duplicate_rows", duplicate_rows},
        {"duplicate_ratio",
         safe_ratio(static_cast<double>(duplicate_rows), static_cast<double>(population))},
        {"groups", groups},
        {"largest_group", largest},
        {"examples", example_json}};
}

}  // namespace vsag::dataset_profile
