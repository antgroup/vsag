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
#include <cstdio>
#include <functional>
#include <limits>
#include <numeric>
#include <unordered_map>

#include "deterministic_random.h"
#include "split_profiler.h"
#include "statistics.h"
#include "vector_math.h"

namespace vsag::dataset_profile {

namespace {

constexpr uint64_t BLOCKS_PER_CHUNK = 16;
constexpr uint32_t DENSE_TERM_LIMIT = 1U << 26U;
constexpr uint64_t ZIPF_FIT_RANKS = 1000;
constexpr uint64_t TOP_TERMS = 20;
constexpr std::array<double, 4> PRUNE_RATIOS = {0.1, 0.2, 0.3, 0.5};

struct SparseBlockPartial {
    Moments values;
    double value_min{std::numeric_limits<double>::infinity()};
    double value_max{-std::numeric_limits<double>::infinity()};
    uint64_t negative_values{0};
    uint64_t zero_values{0};
    uint64_t nan_values{0};
    uint64_t infinite_values{0};
    uint64_t invalid_rows{0};
    uint64_t empty_rows{0};
    std::array<double, PRUNE_RATIOS.size()> retained_terms{};
    uint64_t retention_rows{0};
    uint32_t max_id{0};
    bool has_ids{false};
};

std::string
elapsed_since(std::chrono::steady_clock::time_point start) {
    const auto seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.1fs", seconds);
    return buffer;
}

/// Fraction of a row's terms kept by SINDI's doc_prune_ratio rule: terms sorted by value
/// (descending, ties by id) are kept until their running sum reaches (1 - ratio) of the
/// row's total value mass (src/algorithm/sindi/sindi.cpp).
double
sindi_retained_fraction(std::vector<std::pair<float, uint32_t>>& terms, double ratio) {
    if (terms.size() <= 1) {
        return 1.0;
    }
    double total = 0.0;
    for (const auto& term : terms) {
        total += term.first;
    }
    const double target = total * (1.0 - ratio);
    double current = 0.0;
    uint64_t kept = 0;
    while (current < target && kept < terms.size()) {
        current += terms[kept].first;
        ++kept;
    }
    return static_cast<double>(kept) / static_cast<double>(terms.size());
}

double
row_norm(const SparseBatch& batch, uint64_t r) {
    double sum = 0.0;
    for (uint64_t e = batch.offsets[r]; e < batch.offsets[r + 1]; ++e) {
        sum += static_cast<double>(batch.values[e]) * static_cast<double>(batch.values[e]);
    }
    return std::sqrt(sum);
}

}  // namespace

SparseSplitProfiler::SparseSplitProfiler(SplitContext context)
    : context_(std::move(context)), options_(*context_.options) {
}

void
SparseSplitProfiler::Run(const SparseQuerySet* cross_queries, RowDistanceProbe* probe) {
    probe_ = probe;
    prepare_sample();
    first_pass();
    second_pass(cross_queries);
    finalize();
}

void
SparseSplitProfiler::prepare_sample() {
    const SplitInfo& split = context_.info->Split(context_.name);
    rows_ = split.has_offsets
                ? split.rows
                : count_sparse_rows(
                      context_.info->path, context_.name, options_.chunk_mb * 1024ULL * 1024ULL);
    if (rows_ == 0) {
        throw ProfileError("sparse dataset '/" + context_.name + "' contains no records");
    }
    plan_.population = rows_;
    DeterministicRng rng(derive_seed(options_.seed, "sample/" + context_.name));
    plan_.rows = sample_without_replacement(rows_, options_.sample_size, rng);
    row_norms_.assign(rows_, 0.0);
    row_valid_.assign(rows_, 0);
    full_duplicates_ = options_.Enabled(SECTION_DUPLICATES) &&
                       rows_ * 24 <= options_.duplicate_memory_mb * 1024ULL * 1024ULL;
    if (full_duplicates_) {
        hash_high_.assign(rows_, 0);
        hash_low_.assign(rows_, 0);
    }
}

void
SparseSplitProfiler::first_pass() {
    const auto start = std::chrono::steady_clock::now();
    SparseSplitReader reader(
        context_.info->path, context_.name, options_.chunk_mb * 1024ULL * 1024ULL);
    std::vector<SparseBatch> batches(BLOCKS_PER_CHUNK);
    std::vector<double> nonzeros;
    nonzeros.reserve(rows_);
    std::vector<std::vector<float>> sample_values(plan_.rows.size());
    std::vector<std::vector<uint32_t>> sample_ids(plan_.rows.size());
    SparseBlockPartial total;
    uint64_t unsorted_rows = 0;
    uint64_t duplicate_id_rows = 0;
    uint64_t parsed_rows = 0;
    const bool full_duplicates = full_duplicates_;
    std::vector<uint8_t>& row_valid = row_valid_;
    std::vector<double>& row_norms = row_norms_;
    std::vector<uint64_t>& hash_high = hash_high_;
    std::vector<uint64_t>& hash_low = hash_low_;
    bool more = true;
    while (more) {
        uint64_t loaded = 0;
        uint64_t loaded_rows = 0;
        while (loaded < BLOCKS_PER_CHUNK && reader.NextBatch(BLOCK_ROWS, batches[loaded])) {
            loaded_rows += batches[loaded].Rows();
            ++loaded;
        }
        more = loaded == BLOCKS_PER_CHUNK;
        if (loaded == 0) {
            break;
        }
        // Per-row arrays are sized from the offsets; never index them past the declared count.
        if (loaded_rows > rows_ - parsed_rows) {
            throw ProfileError("sparse dataset '/" + context_.name +
                               "' has more records than its offsets declare (" +
                               std::to_string(rows_) + ")");
        }
        std::vector<SparseBlockPartial> partials(loaded);
        const auto signed_loaded = static_cast<int64_t>(loaded);
        // clang-format off
#pragma omp parallel for schedule(dynamic, 1) default(none) \
    shared(batches, partials, signed_loaded, PRUNE_RATIOS, row_valid, row_norms, \
           full_duplicates, hash_high, hash_low)
        // clang-format on
        for (int64_t b = 0; b < signed_loaded; ++b) {
            const SparseBatch& batch = batches[b];
            SparseBlockPartial& partial = partials[b];
            double sum = 0.0;
            uint64_t finite = 0;
            std::vector<std::pair<float, uint32_t>> terms;
            std::vector<uint32_t> words;
            for (uint64_t r = 0; r < batch.Rows(); ++r) {
                const uint64_t row = batch.first_row + r;
                const uint64_t begin = batch.offsets[r];
                const uint64_t end = batch.offsets[r + 1];
                bool valid = true;
                double norm = 0.0;
                words.clear();
                words.push_back(static_cast<uint32_t>(end - begin));
                for (uint64_t e = begin; e < end; ++e) {
                    const float value = batch.values[e];
                    words.push_back(batch.ids[e]);
                    words.push_back(canonical_bits(value));
                    partial.max_id = std::max(partial.max_id, batch.ids[e]);
                    partial.has_ids = true;
                    if (!is_finite_value(value)) {
                        valid = false;
                        if (is_nan_value(value)) {
                            ++partial.nan_values;
                        } else {
                            ++partial.infinite_values;
                        }
                        continue;
                    }
                    sum += value;
                    ++finite;
                    norm += static_cast<double>(value) * static_cast<double>(value);
                    partial.value_min = std::min<double>(partial.value_min, value);
                    partial.value_max = std::max<double>(partial.value_max, value);
                    partial.negative_values += value < 0.0F ? 1 : 0;
                    partial.zero_values += value == 0.0F ? 1 : 0;
                }
                row_valid[row] = valid ? 1 : 0;
                row_norms[row] = valid ? std::sqrt(norm) : std::numeric_limits<double>::quiet_NaN();
                partial.invalid_rows += valid ? 0 : 1;
                partial.empty_rows += begin == end ? 1 : 0;
                if (full_duplicates) {
                    hash_words(words.data(), words.size(), hash_high[row], hash_low[row]);
                }
                if (valid && end > begin) {
                    terms.clear();
                    for (uint64_t e = begin; e < end; ++e) {
                        terms.emplace_back(batch.values[e], batch.ids[e]);
                    }
                    std::sort(terms.begin(), terms.end(), [](const auto& a, const auto& b) {
                        return a.first != b.first ? a.first > b.first : a.second < b.second;
                    });
                    for (uint64_t i = 0; i < PRUNE_RATIOS.size(); ++i) {
                        partial.retained_terms[i] +=
                            sindi_retained_fraction(terms, PRUNE_RATIOS[i]);
                    }
                    ++partial.retention_rows;
                }
            }
            const double mean = finite > 0 ? sum / static_cast<double>(finite) : 0.0;
            double m2 = 0.0;
            double m3 = 0.0;
            double m4 = 0.0;
            for (const float value : batch.values) {
                if (is_finite_value(value)) {
                    const double centered = static_cast<double>(value) - mean;
                    const double squared = centered * centered;
                    m2 += squared;
                    m3 += squared * centered;
                    m4 += squared * squared;
                }
            }
            partial.values = Moments{finite, mean, m2, m3, m4};
        }
        for (uint64_t b = 0; b < loaded; ++b) {
            const SparseBatch& batch = batches[b];
            const SparseBlockPartial& partial = partials[b];
            total.values.Merge(partial.values);
            total.value_min = std::min(total.value_min, partial.value_min);
            total.value_max = std::max(total.value_max, partial.value_max);
            total.negative_values += partial.negative_values;
            total.zero_values += partial.zero_values;
            total.nan_values += partial.nan_values;
            total.infinite_values += partial.infinite_values;
            total.invalid_rows += partial.invalid_rows;
            total.empty_rows += partial.empty_rows;
            for (uint64_t i = 0; i < PRUNE_RATIOS.size(); ++i) {
                total.retained_terms[i] += partial.retained_terms[i];
            }
            total.retention_rows += partial.retention_rows;
            if (partial.has_ids) {
                total.max_id = std::max(total.max_id, partial.max_id);
                total.has_ids = true;
            }
            unsorted_rows += batch.unsorted_rows;
            duplicate_id_rows += batch.duplicate_id_rows;
            // Document frequency: count each distinct id once per row (ids are sorted).
            for (uint64_t r = 0; r < batch.Rows(); ++r) {
                nonzeros.push_back(static_cast<double>(batch.offsets[r + 1] - batch.offsets[r]));
                for (uint64_t e = batch.offsets[r]; e < batch.offsets[r + 1]; ++e) {
                    if (e > batch.offsets[r] && batch.ids[e] == batch.ids[e - 1]) {
                        continue;
                    }
                    const uint32_t id = batch.ids[e];
                    if (id < DENSE_TERM_LIMIT) {
                        if (id >= term_frequency_.size()) {
                            term_frequency_.resize(static_cast<uint64_t>(id) + 1, 0);
                        }
                        ++term_frequency_[id];
                    } else {
                        ++overflow_terms_[id];
                    }
                }
            }
            // Capture sampled rows.
            auto it = std::lower_bound(plan_.rows.begin(), plan_.rows.end(), batch.first_row);
            for (; it != plan_.rows.end() && *it < batch.first_row + batch.Rows(); ++it) {
                const auto position = static_cast<uint64_t>(it - plan_.rows.begin());
                const uint64_t r = *it - batch.first_row;
                sample_ids[position].assign(
                    batch.ids.begin() + static_cast<int64_t>(batch.offsets[r]),
                    batch.ids.begin() + static_cast<int64_t>(batch.offsets[r + 1]));
                sample_values[position].assign(
                    batch.values.begin() + static_cast<int64_t>(batch.offsets[r]),
                    batch.values.begin() + static_cast<int64_t>(batch.offsets[r + 1]));
            }
            parsed_rows += batch.Rows();
        }
    }
    if (parsed_rows != rows_) {
        throw ProfileError("sparse dataset '/" + context_.name + "' has " +
                           std::to_string(parsed_rows) + " records but its offsets declare " +
                           std::to_string(rows_));
    }
    value_min_ = std::isfinite(total.value_min) ? total.value_min : 0.0;
    value_max_ = std::isfinite(total.value_max) ? total.value_max : 0.0;
    std::vector<float> sampled_values;
    for (const auto& values : sample_values) {
        sampled_values.insert(sampled_values.end(), values.begin(), values.end());
    }
    value_clipped_ =
        streamed_value_range(sampled_values, value_min_, value_max_, value_low_, value_high_);

    // Seeded analysis order over valid sampled rows, materialized as a query set.
    std::vector<uint64_t> order(plan_.rows.size());
    std::iota(order.begin(), order.end(), 0);
    DeterministicRng order_rng(derive_seed(options_.seed, "order/" + context_.name));
    shuffle_in_place(order, order_rng);
    for (const uint64_t position : order) {
        const uint64_t row = plan_.rows[position];
        const bool usable =
            row_valid_[row] != 0 && (context_.metric != Metric::COSINE || row_norms_[row] > 0.0);
        if (!usable) {
            continue;
        }
        plan_.analysis_order.push_back(position);
        sample_.ids.insert(
            sample_.ids.end(), sample_ids[position].begin(), sample_ids[position].end());
        sample_.values.insert(
            sample_.values.end(), sample_values[position].begin(), sample_values[position].end());
        sample_.offsets.push_back(sample_.ids.size());
        sample_.norms.push_back(row_norms_[row]);
        sample_.self_rows.push_back(row);
    }

    // Report sections that only need the full scan.
    const SplitInfo& split = context_.info->Split(context_.name);
    std::vector<double> norms;
    norms.reserve(rows_);
    uint64_t unit_norm = 0;
    for (uint64_t r = 0; r < rows_; ++r) {
        if (row_valid_[r] != 0) {
            norms.push_back(row_norms_[r]);
            unit_norm += std::abs(row_norms_[r] - 1.0) <= 1e-3 ? 1 : 0;
        }
    }
    sorted_norms_.assign(norms.begin(), norms.end());
    std::sort(sorted_norms_.begin(), sorted_norms_.end());
    sorted_nonzeros_ = nonzeros;
    std::sort(sorted_nonzeros_.begin(), sorted_nonzeros_.end());
    const uint64_t entries = total.values.count + total.nan_values + total.infinite_values;
    json_["kind"] = "sparse";
    json_["rows"] = rows_;
    json_["storage_type"] = split.storage_type;
    json_["basic"] =
        JsonType{{"scope", "full"},
                 {"definition", "Record count, stored entries and byte size of the sparse split."},
                 {"rows", rows_},
                 {"stored_entries", entries},
                 {"stream_bytes", split.stored_bytes},
                 {"max_nonzeros", sorted_nonzeros_.empty() ? 0.0 : sorted_nonzeros_.back()},
                 {"offsets_present", split.has_offsets},
                 {"unsorted_rows", unsorted_rows},
                 {"rows_with_repeated_ids", duplicate_id_rows}};
    json_["invalid_values"] = JsonType{
        {"scope", "full"},
        {"definition",
         "NaN and infinite stored values; rows containing any are excluded from distance metrics."},
        {"nan", total.nan_values},
        {"infinite", total.infinite_values},
        {"total", total.nan_values + total.infinite_values},
        {"ratio",
         safe_ratio(static_cast<double>(total.nan_values + total.infinite_values),
                    static_cast<double>(entries))},
        {"rows_with_invalid", total.invalid_rows},
        {"row_ratio",
         safe_ratio(static_cast<double>(total.invalid_rows), static_cast<double>(rows_))}};
    std::vector<double> nonzero_copy = nonzeros;
    JsonType nonzero_summary = summarize_values(nonzero_copy);
    const double id_space = total.has_ids ? static_cast<double>(total.max_id) + 1.0 : 0.0;
    json_["sparsity"] = JsonType{
        {"scope", "full"},
        {"definition",
         "Stored (non-zero) entries per record; density is the mean entry count over the id "
         "space [0, max_term_id]."},
        {"empty_rows", total.empty_rows},
        {"nonzeros_per_row",
         JsonType{{"summary", nonzero_summary},
                  {"histogram", integer_histogram(sorted_nonzeros_, options_.histogram_bins)}}},
        {"density", safe_ratio(nonzero_summary.value("mean", 0.0), id_space)}};
    std::vector<double> norms_sorted = norms;
    JsonType norm_summary = summarize_values(norms_sorted);
    json_["norms"] = JsonType{
        {"scope", "full"},
        {"definition", "L2 norm of every record with finite values."},
        {"summary", norm_summary},
        {"histogram", histogram_of_sorted(norms_sorted, options_.histogram_bins)},
        {"coefficient_of_variation",
         safe_ratio(norm_summary.value("std", 0.0), norm_summary.value("mean", 0.0))},
        {"unit_norm_ratio",
         safe_ratio(static_cast<double>(unit_norm), static_cast<double>(norms_sorted.size()))}};
    json_["values"] = JsonType{{"scope", "full"},
                               {"definition", "Distribution of stored finite values."},
                               {"count", total.values.count},
                               {"min", json_number(value_min_)},
                               {"max", json_number(value_max_)},
                               {"mean", json_number(total.values.mean)},
                               {"std", json_number(total.values.Std())},
                               {"skewness", json_number(total.values.Skewness())},
                               {"excess_kurtosis", json_number(total.values.ExcessKurtosis())},
                               {"negative_values", total.negative_values},
                               {"explicit_zero_values", total.zero_values}};

    // Term statistics.
    std::vector<std::pair<uint64_t, uint32_t>> frequencies;
    for (uint64_t id = 0; id < term_frequency_.size(); ++id) {
        if (term_frequency_[id] > 0) {
            frequencies.emplace_back(term_frequency_[id], static_cast<uint32_t>(id));
        }
    }
    for (const auto& [id, frequency] : overflow_terms_) {
        frequencies.emplace_back(frequency, id);
    }
    std::sort(frequencies.begin(), frequencies.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    });
    std::vector<double> df_values;
    df_values.reserve(frequencies.size());
    uint64_t postings = 0;
    for (const auto& item : frequencies) {
        df_values.push_back(static_cast<double>(item.first));
        postings += item.first;
    }
    std::vector<double> df_ascending(df_values.rbegin(), df_values.rend());
    std::vector<double> log_df;
    log_df.reserve(df_ascending.size());
    for (const double value : df_ascending) {
        log_df.push_back(std::log10(value));
    }
    double top_share = 0.0;
    const uint64_t top_terms = std::max<uint64_t>(1, frequencies.size() / 100);
    for (uint64_t i = 0; i < top_terms && i < df_values.size(); ++i) {
        top_share += df_values[i];
    }
    // Zipf exponent: least-squares slope of log(df) against log(rank).
    double zipf = std::numeric_limits<double>::quiet_NaN();
    const uint64_t fit = std::min<uint64_t>(ZIPF_FIT_RANKS, df_values.size());
    if (fit >= 3) {
        double sx = 0.0;
        double sy = 0.0;
        double sxx = 0.0;
        double sxy = 0.0;
        for (uint64_t i = 0; i < fit; ++i) {
            const double x = std::log(static_cast<double>(i + 1));
            const double y = std::log(df_values[i]);
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        const auto n = static_cast<double>(fit);
        const double denominator = n * sxx - sx * sx;
        if (denominator != 0.0) {
            zipf = -(n * sxy - sx * sy) / denominator;
        }
    }
    JsonType top = JsonType::array();
    for (uint64_t i = 0; i < std::min<uint64_t>(TOP_TERMS, frequencies.size()); ++i) {
        top.push_back(JsonType{{"term", frequencies[i].second},
                               {"document_frequency", frequencies[i].first}});
    }
    std::vector<double> df_copy = df_values;
    JsonType retention = JsonType::object();
    for (uint64_t i = 0; i < PRUNE_RATIOS.size(); ++i) {
        // Keys are percentages ("10") so that report paths contain no dots.
        retention[std::to_string(std::lround(PRUNE_RATIOS[i] * 100.0))] =
            safe_ratio(total.retained_terms[i], static_cast<double>(total.retention_rows));
    }
    const uint64_t term_id_limit = total.has_ids ? static_cast<uint64_t>(total.max_id) + 1 : 0;
    json_["terms"] = JsonType{
        {"scope", "full"},
        {"definition",
         "Document frequency (records containing a term id), the posting-list length of an "
         "inverted index such as SINDI."},
        {"distinct_terms", frequencies.size()},
        {"max_term_id", total.has_ids ? JsonType(total.max_id) : JsonType(nullptr)},
        {"min_term_id_limit", term_id_limit},
        {"term_id_density",
         safe_ratio(static_cast<double>(frequencies.size()), static_cast<double>(term_id_limit))},
        {"postings", postings},
        {"document_frequency",
         JsonType{{"summary", summarize_values(df_copy)},
                  {"log10_histogram", histogram_of_sorted(log_df, options_.histogram_bins)}}},
        {"zipf_exponent", json_number(zipf)},
        {"gini", json_number(gini_sorted(df_ascending))},
        {"top_1pct_terms_posting_share", safe_ratio(top_share, static_cast<double>(postings))},
        {"top_terms", top},
        {"doc_prune_retained_term_fraction",
         JsonType{{"definition",
                   "Mean fraction of a record's terms kept by SINDI doc_prune_ratio (terms sorted "
                   "by value are kept until (1 - ratio) of the value mass is reached)."},
                  {"by_ratio", retention}}}};

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
            std::vector<uint32_t> words;
            for (uint64_t i = 0; i < plan_.rows.size(); ++i) {
                words.clear();
                words.push_back(static_cast<uint32_t>(sample_ids[i].size()));
                for (uint64_t e = 0; e < sample_ids[i].size(); ++e) {
                    words.push_back(sample_ids[i][e]);
                    words.push_back(canonical_bits(sample_values[i][e]));
                }
                hash_words(words.data(), words.size(), high[i], low[i]);
            }
            JsonType report = duplicate_report(high, low, plan_.rows, plan_.rows.size());
            report["scope"] = "sample";
            report["reason"] = "row count exceeds --duplicate-memory-mb";
            report["sample"] = sample_json(plan_, plan_.rows.size(), "all sampled rows");
            json_["duplicates"] = report;
        }
    }
    if (context_.progress) {
        context_.progress("[" + context_.name + "] pass 1/2 (full scan) done in " +
                          elapsed_since(start));
    }
}

SparseQuerySet
SparseSplitProfiler::MakeQuerySet(uint64_t count) const {
    count = std::min<uint64_t>(count, sample_.Size());
    SparseQuerySet queries;
    queries.ids.assign(sample_.ids.begin(),
                       sample_.ids.begin() + static_cast<int64_t>(sample_.offsets[count]));
    queries.values.assign(sample_.values.begin(),
                          sample_.values.begin() + static_cast<int64_t>(sample_.offsets[count]));
    queries.offsets.assign(sample_.offsets.begin(),
                           sample_.offsets.begin() + static_cast<int64_t>(count + 1));
    queries.norms.assign(sample_.norms.begin(),
                         sample_.norms.begin() + static_cast<int64_t>(count));
    queries.self_rows.assign(count, NO_ROW);
    return queries;
}

std::vector<uint64_t>
SparseSplitProfiler::AnalysisRowIds(uint64_t count) const {
    count = std::min<uint64_t>(count, sample_.Size());
    return {sample_.self_rows.begin(), sample_.self_rows.begin() + static_cast<int64_t>(count)};
}

uint64_t
SparseSplitProfiler::DocumentFrequency(uint32_t term) const {
    if (term < term_frequency_.size()) {
        return term_frequency_[term];
    }
    const auto it = overflow_terms_.find(term);
    return it == overflow_terms_.end() ? 0 : it->second;
}

void
SparseSplitProfiler::second_pass(const SparseQuerySet* cross_queries) {
    const auto start = std::chrono::steady_clock::now();
    const bool run_knn = options_.Enabled(SECTION_KNN);
    SparseQuerySet self_queries = MakeQuerySet(run_knn ? options_.knn_queries : 0);
    for (uint64_t i = 0; i < self_queries.Size(); ++i) {
        self_queries.self_rows[i] = sample_.self_rows[i];
    }
    const uint64_t k = std::min<uint64_t>(options_.knn_k, rows_ > 1 ? rows_ - 1 : 1);
    SparseKnnSearch self_search(self_queries, context_.metric, k);
    SparseQuerySet empty;
    const SparseQuerySet& cross = cross_queries != nullptr ? *cross_queries : empty;
    SparseKnnSearch cross_search(cross, context_.metric, std::min<uint64_t>(options_.knn_k, rows_));
    Histogram histogram(value_low_, value_high_, options_.histogram_bins);
    SparseSplitReader reader(
        context_.info->path, context_.name, options_.chunk_mb * 1024ULL * 1024ULL);
    SparseBatch batch;
    std::vector<double> norms;
    std::vector<uint8_t> valid;
    while (reader.NextBatch(BLOCK_ROWS, batch)) {
        for (const float value : batch.values) {
            if (is_finite_value(value)) {
                histogram.Add(value);
            }
        }
        norms.resize(batch.Rows());
        valid.resize(batch.Rows());
        for (uint64_t r = 0; r < batch.Rows(); ++r) {
            norms[r] = row_norm(batch, r);
            // Pass 1 validated the record count, so these row ids are in range.
            valid[r] = row_valid_[batch.first_row + r] != 0 &&
                               (context_.metric != Metric::COSINE || norms[r] > 0.0)
                           ? 1
                           : 0;
        }
        if (self_queries.Size() > 0) {
            self_search.Scan(batch, norms, valid);
        }
        if (cross.Size() > 0) {
            cross_search.Scan(batch, norms, valid);
        }
        if (probe_ != nullptr && cross.Size() > 0) {
            // Same products in the same (ascending id) order as SparseKnnSearch.
            auto it = std::lower_bound(probe_->rows.begin(), probe_->rows.end(), batch.first_row);
            for (; it != probe_->rows.end() && *it < batch.first_row + batch.Rows(); ++it) {
                const auto entry = static_cast<uint64_t>(it - probe_->rows.begin());
                const uint64_t r = *it - batch.first_row;
                const uint64_t q = probe_->queries[entry];
                if (valid[r] == 0) {
                    continue;
                }
                const double dot = sparse_dot(cross.ids.data() + cross.offsets[q],
                                              cross.values.data() + cross.offsets[q],
                                              cross.offsets[q + 1] - cross.offsets[q],
                                              batch.ids.data() + batch.offsets[r],
                                              batch.values.data() + batch.offsets[r],
                                              batch.offsets[r + 1] - batch.offsets[r]);
                double distance = 1.0 - dot;
                if (context_.metric == Metric::COSINE) {
                    const double denominator = cross.norms[q] * norms[r];
                    distance = denominator > 0.0 ? 1.0 - dot / denominator : 1.0;
                }
                probe_->distances[entry] = distance;
            }
        }
    }
    json_["values"]["histogram"] = histogram.ToJson();
    json_["values"]["histogram"]["clipped"] = value_clipped_;
    self_neighbors_ = self_search.MetricNeighbors();
    self_few_overlaps_ = self_search.QueriesWithFewOverlaps();
    cross_neighbors_ = cross_search.MetricNeighbors();
    if (context_.progress) {
        context_.progress("[" + context_.name + "] pass 2/2 (value histogram, exact k-NN of " +
                          std::to_string(self_queries.Size() + cross.Size()) +
                          " queries) done in " + elapsed_since(start));
    }
}

void
SparseSplitProfiler::finalize() {
    const uint64_t valid_rows = sample_.Size();
    const Metric metric = context_.metric;
    const SparseQuerySet& sample = sample_;
    auto row_ids = [&sample](uint64_t i) { return sample.ids.data() + sample.offsets[i]; };
    auto row_values = [&sample](uint64_t i) { return sample.values.data() + sample.offsets[i]; };
    auto row_length = [&sample](uint64_t i) { return sample.offsets[i + 1] - sample.offsets[i]; };

    if (valid_rows >= 2) {
        DeterministicRng pair_rng(derive_seed(options_.seed, "pairs/" + context_.name));
        const uint64_t pairs = options_.pair_count;
        std::vector<double> distances(pairs);
        std::vector<double> cosines(pairs);
        std::vector<uint8_t> overlap(pairs, 0);
        std::vector<std::pair<uint64_t, uint64_t>> indices(pairs);
        for (auto& pair : indices) {
            const uint64_t a = pair_rng.Uniform(valid_rows);
            uint64_t b = pair_rng.Uniform(valid_rows - 1);
            b += b >= a ? 1 : 0;
            pair = {a, b};
        }
        const auto signed_pairs = static_cast<int64_t>(pairs);
        // clang-format off
#pragma omp parallel for schedule(static) default(none) \
    shared(indices, row_ids, row_values, row_length, sample, metric, distances, cosines, \
           overlap, signed_pairs)
        // clang-format on
        for (int64_t p = 0; p < signed_pairs; ++p) {
            const auto [a, b] = indices[p];
            const double dot = sparse_dot(
                row_ids(a), row_values(a), row_length(a), row_ids(b), row_values(b), row_length(b));
            const double denominator = sample.norms[a] * sample.norms[b];
            const double cosine = denominator > 0.0 ? dot / denominator : 0.0;
            cosines[p] = cosine;
            distances[p] = metric == Metric::COSINE ? 1.0 - cosine : 1.0 - dot;
            // Shared ids imply an overlap even if their products cancel out.
            const uint32_t* ia = row_ids(a);
            const uint32_t* ib = row_ids(b);
            uint64_t i = 0;
            uint64_t j = 0;
            while (i < row_length(a) && j < row_length(b)) {
                if (ia[i] == ib[j]) {
                    overlap[p] = 1;
                    break;
                }
                if (ia[i] < ib[j]) {
                    ++i;
                } else {
                    ++j;
                }
            }
        }
        uint64_t overlapping = 0;
        for (const auto flag : overlap) {
            overlapping += flag;
        }
        std::vector<double> distance_sorted = distances;
        JsonType distance_summary = summarize_values(distance_sorted);
        std::vector<double> cosine_sorted = cosines;
        JsonType cosine_summary = summarize_values(cosine_sorted);
        json_["pairwise_distances"] = JsonType{
            {"scope", "sample"},
            {"definition",
             "Distances between uniformly drawn pairs of distinct sampled records (VSAG "
             "convention)."},
            {"sample", sample_json(plan_, valid_rows, "all valid sampled rows")},
            {"pairs", pairs},
            {"metric_distance",
             JsonType{
                 {"summary", distance_summary},
                 {"histogram", histogram_of_sorted(distance_sorted, options_.histogram_bins)}}},
            {"cosine_similarity",
             JsonType{{"summary", cosine_summary},
                      {"histogram", histogram_of_sorted(cosine_sorted, options_.histogram_bins)}}},
            {"zero_overlap_pair_ratio",
             safe_ratio(static_cast<double>(pairs - overlapping), static_cast<double>(pairs))}};
    }

    if (!self_neighbors_.empty()) {
        std::vector<double> first;
        std::vector<double> kth;
        for (const auto& neighbors : self_neighbors_) {
            first.push_back(neighbors.front().distance);
            kth.push_back(neighbors.back().distance);
        }
        self_first_distance_ = first;
        std::vector<double> first_sorted = first;
        JsonType first_summary = summarize_values(first_sorted);
        std::vector<double> kth_sorted = kth;
        JsonType kth_summary = summarize_values(kth_sorted);
        json_["nearest_neighbors"] = JsonType{
            {"scope", "sample"},
            {"definition",
             "Exact k nearest neighbors of sampled records searched in the full split through an "
             "inverted index; records sharing no term have distance 1."},
            {"sample",
             sample_json(plan_, self_neighbors_.size(), "first knn_queries valid sampled rows")},
            {"k", self_neighbors_.front().size()},
            {"searched_rows", rows_},
            {"first_neighbor_distance",
             JsonType{{"summary", first_summary},
                      {"histogram", histogram_of_sorted(first_sorted, options_.histogram_bins)}}},
            {"kth_neighbor_distance",
             JsonType{{"summary", kth_summary},
                      {"histogram", histogram_of_sorted(kth_sorted, options_.histogram_bins)}}},
            {"queries_with_fewer_than_k_overlaps", self_few_overlaps_},
            {"queries_with_fewer_than_k_overlaps_ratio",
             safe_ratio(static_cast<double>(self_few_overlaps_),
                        static_cast<double>(self_neighbors_.size()))}};
    }

    if (options_.Enabled(SECTION_HUBNESS) && valid_rows > options_.hubness_k) {
        const uint64_t hub_rows = std::min<uint64_t>(options_.hubness_sample, valid_rows);
        SparseQuerySet hub_queries = MakeQuerySet(hub_rows);
        for (uint64_t i = 0; i < hub_rows; ++i) {
            hub_queries.self_rows[i] = i;
        }
        const uint64_t hub_k = std::min<uint64_t>(options_.hubness_k, hub_rows - 1);
        SparseKnnSearch hub_search(hub_queries, metric, hub_k);
        SparseBatch batch;
        batch.Clear();
        batch.ids = hub_queries.ids;
        batch.values = hub_queries.values;
        batch.offsets = hub_queries.offsets;
        // Sampled rows are valid (and non-zero under cosine) by construction.
        hub_search.Scan(batch, hub_queries.norms, std::vector<uint8_t>(hub_rows, 1));
        std::vector<std::vector<uint8_t>> shares_term;
        const auto hub_neighbors = hub_search.MetricNeighbors(&shares_term);
        std::vector<double> occurrences(hub_rows, 0.0);
        for (uint64_t q = 0; q < hub_neighbors.size(); ++q) {
            // Rows without a shared term all tie at distance 1; counting them would reward
            // low row ids, so only neighbors sharing a term count.
            for (uint64_t j = 0; j < hub_neighbors[q].size(); ++j) {
                if (shares_term[q][j] != 0) {
                    occurrences[hub_neighbors[q][j].index] += 1.0;
                }
            }
        }
        Moments moments;
        uint64_t anti_hubs = 0;
        for (const double value : occurrences) {
            moments.Add(value);
            anti_hubs += value == 0.0 ? 1 : 0;
        }
        std::vector<double> ascending = occurrences;
        std::sort(ascending.begin(), ascending.end());
        json_["hubness"] = JsonType{
            {"scope", "sample"},
            {"definition",
             "k-occurrence N_k of sampled records in the k-NN lists of other sampled records; "
             "padding neighbors without shared terms are not counted."},
            {"sample", sample_json(plan_, hub_rows, "first hubness_sample valid sampled rows")},
            {"k", hub_k},
            {"skewness", json_number(moments.Skewness())},
            {"max_occurrence", ascending.back()},
            {"anti_hub_ratio",
             safe_ratio(static_cast<double>(anti_hubs), static_cast<double>(hub_rows))},
            {"histogram", integer_histogram(ascending, options_.histogram_bins)}};
    }
}

}  // namespace vsag::dataset_profile
