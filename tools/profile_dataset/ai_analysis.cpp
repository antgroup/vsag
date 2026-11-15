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

#include "ai_analysis.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <set>
#include <thread>

#include "sha256.h"
#include "statistics.h"

namespace vsag::dataset_profile {

namespace {

constexpr uint64_t MAX_EVIDENCE_BYTES = 48ULL * 1024ULL;
constexpr uint64_t MAX_ARRAY_ITEMS = 10;
constexpr uint64_t MAX_STRING_LENGTH = 300;
constexpr uint64_t MAX_REPLY_BYTES = 256ULL * 1024ULL;
// The provider's JSON envelope around the reply, escapes included; larger bodies are cut off
// while they arrive.
constexpr uint64_t MAX_RESPONSE_BYTES = 4ULL * MAX_REPLY_BYTES;
constexpr uint64_t MAX_ITEMS = 20;
constexpr uint64_t MAX_EVIDENCE_PER_ITEM = 12;
constexpr uint64_t MAX_TEXT = 2000;
constexpr uint64_t MAX_ERROR_TEXT = 300;

// Segments that never leave the machine: per-row or per-dimension data, ids, chart arrays,
// free text that is not a metric, and label values.
const std::set<std::string> DENIED_SEGMENTS = {"definition",
                                               "per_dimension",
                                               "histogram",
                                               "log10_histogram",
                                               "points",
                                               "projection",
                                               "query_projection",
                                               "examples",
                                               "top_terms",
                                               "top_variance_dimensions",
                                               "lowest_variance_dimensions",
                                               "top_dimensions",
                                               "rule",
                                               "evidence",
                                               "method",
                                               "subset",
                                               "isolated_definition",
                                               "sq8_uniform_bounds",
                                               "sq4_uniform_bounds",
                                               "sq8_bounds",
                                               "sq4_bounds",
                                               "min_label",
                                               "max_label"};

// Below splits, cross_split, ground_truth and labels, every path segment must be a key that the
// profiler writes (array indices and percent keys aside). Fields added to a saved profile, even
// inside known sections, therefore never reach a provider.
const std::set<std::string> REPORT_KEYS = {
    "1 - similarity",
    "above",
    "always_zero_dimensions",
    "anisotropy",
    "anti_hub_ratio",
    "average_cosine_similarity",
    "base_rows",
    "basic",
    "below",
    "best_convention_mean_relative_error",
    "best_matching_distance_convention",
    "bf16",
    "bin_width",
    "bits_per_dim",
    "by_ratio",
    "checked_queries",
    "clipped",
    "clustering",
    "clusters",
    "code_bytes_per_vector",
    "coefficient_of_variation",
    "compared_k",
    "compatible",
    "components_for_variance",
    "concentration_ratio",
    "constant_dimensions",
    "convention_mean_relative_errors",
    "converged",
    "cosine_similarity",
    "count",
    "counts",
    "cumulative_explained_variance",
    "density",
    "dim",
    "dimensions",
    "distance_attribute",
    "distance_recall_at_k",
    "distinct_labels",
    "distinct_terms",
    "doc_prune_retained_term_fraction",
    "document_frequency",
    "duplicate_ratio",
    "duplicate_rows",
    "duplicates",
    "effective_rank",
    "empty_rows",
    "encodes_query",
    "euclidean (sqrt of l2sqr)",
    "eval_compatibility",
    "exact_duplicate_neighbor_ratio",
    "excess_kurtosis",
    "explained_variance_ratio",
    "explicit_zero_values",
    "first_neighbor_distance",
    "float32_bytes",
    "fp16",
    "geometric_distance",
    "gini",
    "groups",
    "heavy_tailed_dimension_ratio",
    "high",
    "hubness",
    "hubness_k",
    "id",
    "id_recall_at_k",
    "inertia",
    "infinite",
    "input",
    "intrinsic_dimensionality",
    "invalid_values",
    "isolated_ratio",
    "issues",
    "iterations",
    "k",
    "kind",
    "knn_k",
    "knn_queries",
    "kth_neighbor_distance",
    "label_filtered",
    "label_filtered_queries",
    "largest_group",
    "largest_label_share",
    "length",
    "lid_mle",
    "lid_to_dim_ratio",
    "low",
    "lower",
    "lower_fence",
    "max",
    "max_nonzeros",
    "max_occurrence",
    "max_term_id",
    "max_to_mean_size",
    "mean",
    "mean_abs_skewness",
    "mean_relative_reconstruction_error",
    "mean_vector_norm_ratio",
    "median_distance_ratio",
    "message",
    "metric",
    "metric_distance",
    "min",
    "min_term_id_limit",
    "nan",
    "near_duplicate_ratio",
    "near_duplicate_threshold",
    "nearest_neighbors",
    "non_finite_stored_distances",
    "negative similarity",
    "negative_infinity",
    "negative_values",
    "nonzeros_ks_statistic",
    "nonzeros_per_row",
    "norm_ks_statistic",
    "normalized_entropy",
    "norms",
    "offsets_present",
    "out_of_distribution_ratio",
    "out_of_range_ids",
    "outlier_ratio_6_sigma",
    "outliers",
    "outliers_6_sigma",
    "p01",
    "p05",
    "p25",
    "p50",
    "p75",
    "p95",
    "p99",
    "pair_count",
    "pairs",
    "pairwise_distances",
    "participation_ratio",
    "pca",
    "pca_top1_ratio",
    "population",
    "positive_infinity",
    "postings",
    "postings_per_query",
    "profiled_splits",
    "quantization",
    "quantization_k",
    "queries_with_fewer_than_k_overlaps",
    "queries_with_fewer_than_k_overlaps_ratio",
    "queries_without_matching_base_rows",
    "query_first_neighbor_distance",
    "query_rows",
    "query_selectivity",
    "query_source",
    "query_term_coverage",
    "range_ratio_max_to_median",
    "ratio",
    "reason",
    "recall_at_k",
    "recomputed_stored_ids",
    "reference_rows",
    "relative_contrast",
    "relative_to_median_pairwise_distance",
    "repeated_stored_ids",
    "row_ratio",
    "rows",
    "rows_with_invalid",
    "rows_with_repeated_ids",
    "rows_with_unsorted_distances",
    "sample",
    "sample_size",
    "sampled_rows",
    "schemes",
    "scope",
    "scope_distance_recall",
    "search_scope",
    "searched_rows",
    "seed",
    "severity",
    "shifted_constant_dimensions",
    "silhouette",
    "silhouette_rows",
    "similarity",
    "size",
    "size_cv",
    "sizes",
    "skewness",
    "skipped",
    "sparsity",
    "sq4",
    "sq4_uniform",
    "sq8",
    "sq8_uniform",
    "squared euclidean",
    "standardized_mean_shift",
    "status",
    "std",
    "std_ratio_max_to_median",
    "storage_type",
    "stored_entries",
    "stored_k",
    "stored_queries",
    "stream_bytes",
    "summary",
    "term_id_density",
    "terms",
    "test",
    "threshold",
    "top1_ratio",
    "top_1pct_occurrence_share",
    "top_1pct_terms_posting_share",
    "total",
    "total_variance",
    "train",
    "two_nn",
    "two_nn_excluded_ratio",
    "unfiltered",
    "unit_norm_ratio",
    "unit_norm_tolerance",
    "unsorted_rows",
    "upper",
    "upper_fence",
    "used_rows",
    "valid_ratios",
    "valid_sampled_rows",
    "value",
    "values",
    "variance_share_top_10pct_dimensions",
    "vector_type",
    "warnings",
    "zero_norm_rows",
    "zero_overlap_pair_ratio",
    "zero_ratio",
    "zero_rows",
    "zipf_exponent",
};

// Aggregate arrays that may be sent, at most MAX_ARRAY_ITEMS items each. Any other array could
// carry rows, coordinates or ids.
const std::set<std::string> ARRAY_KEYS = {"cumulative_explained_variance",
                                          "explained_variance_ratio",
                                          "issues",
                                          "profiled_splits",
                                          "sizes",
                                          "warnings"};

// Objects whose keys are percent levels ("90", "50").
const std::set<std::string> NUMBERED_KEYS = {"by_ratio", "components_for_variance"};

// String values that may be sent: enumerations and fixed texts that the profiler writes.
const std::set<std::string> STRING_KEYS = {"best_matching_distance_convention",
                                           "distance_attribute",
                                           "id",
                                           "input",
                                           "issues",
                                           "kind",
                                           "message",
                                           "metric",
                                           "profiled_splits",
                                           "query_source",
                                           "scope",
                                           "search_scope",
                                           "severity",
                                           "status",
                                           "storage_type",
                                           "vector_type",
                                           "warnings"};

const std::set<std::string> ALLOWED_INPUT = {
    "vector_type", "distance_attribute", "eval_compatibility", "warnings"};
const std::set<std::string> ALLOWED_CONFIG = {"metric",
                                              "seed",
                                              "sample_size",
                                              "knn_queries",
                                              "knn_k",
                                              "pair_count",
                                              "clusters",
                                              "hubness_k",
                                              "quantization_k",
                                              "near_duplicate_threshold",
                                              "profiled_splits"};
const std::set<std::string> ALLOWED_SIGNAL_FIELDS = {"id", "severity", "message"};

const std::vector<std::string> OVERCLAIMS = {
    "guarantee",
    "always optimal",
    "always the best",
    "best possible",
    "optimal configuration",
    "definitely",
    "certainly",
    "no need to benchmark",
    "without benchmarking",
    "\xe4\xbf\x9d\xe8\xaf\x81",                           // "guarantee" in Chinese
    "\xe4\xb8\x80\xe5\xae\x9a\xe6\x9c\x80\xe4\xbc\x98",   // "certainly optimal"
    "\xe5\xbf\x85\xe7\x84\xb6\xe6\x9c\x80\xe4\xbc\x98",   // "necessarily optimal"
    "\xe7\xbb\x9d\xe5\xaf\xb9\xe6\x9c\x80\xe4\xbc\x98"};  // "absolutely optimal"

const std::set<std::string> SEVERITIES = {"info", "notice", "warning", "critical"};
const std::set<std::string> CATEGORIES = {
    "index", "metric", "quantization", "transformation", "evaluation", "data_quality"};
const std::set<std::string> CONFIDENCES = {"low", "medium", "high"};

bool
is_number_key(const std::string& segment) {
    return !segment.empty() && segment.find_first_not_of("0123456789") == std::string::npos;
}

/// True when every segment after the root is a known report key, an index of an allowed array
/// or a percent key.
bool
known_metric_path(const std::vector<std::string>& segments) {
    for (uint64_t i = 1; i < segments.size(); ++i) {
        if (is_number_key(segments[i])) {
            if (ARRAY_KEYS.count(segments[i - 1]) == 0 &&
                NUMBERED_KEYS.count(segments[i - 1]) == 0) {
                return false;
            }
        } else if (REPORT_KEYS.count(segments[i]) == 0) {
            return false;
        }
    }
    return true;
}

bool
denied(const std::vector<std::string>& segments, bool include_file_name) {
    if (segments.empty()) {
        return true;
    }
    for (const auto& segment : segments) {
        // Prose such as "isolated_definition" explains a metric; it carries no evidence.
        if (DENIED_SEGMENTS.count(segment) > 0 || segment.ends_with("definition")) {
            return true;
        }
    }
    const std::string& root = segments.front();
    if (root == "input") {
        if (segments.size() >= 2 && segments[1] == "file_name") {
            return !include_file_name;
        }
        return segments.size() < 2 || ALLOWED_INPUT.count(segments[1]) == 0;
    }
    if (root == "config") {
        return segments.size() < 2 || ALLOWED_CONFIG.count(segments[1]) == 0;
    }
    if (root == "signals") {
        return segments.size() < 3 || ALLOWED_SIGNAL_FIELDS.count(segments[2]) == 0;
    }
    if (root != "splits" && root != "cross_split" && root != "ground_truth" && root != "labels") {
        return true;
    }
    return !known_metric_path(segments);
}

/// Replaces every byte that is not part of a well-formed UTF-8 sequence (RFC 3629) with U+FFFD, so
/// model text, provider errors and file names can always be serialized as JSON.
std::string
sanitize_utf8(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    uint64_t i = 0;
    while (i < text.size()) {
        const auto lead = static_cast<unsigned char>(text[i]);
        uint64_t length = 0;
        unsigned char second_low = 0x80;
        unsigned char second_high = 0xBF;
        if (lead < 0x80) {
            length = 1;
        } else if (lead >= 0xC2 && lead <= 0xDF) {
            length = 2;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            length = 3;
            second_low = lead == 0xE0 ? 0xA0 : 0x80;   // no overlong forms
            second_high = lead == 0xED ? 0x9F : 0xBF;  // no surrogates
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            length = 4;
            second_low = lead == 0xF0 ? 0x90 : 0x80;
            second_high = lead == 0xF4 ? 0x8F : 0xBF;  // at most U+10FFFF
        }
        bool valid = length > 0 && i + length <= text.size();
        for (uint64_t j = 1; valid && j < length; ++j) {
            const auto byte = static_cast<unsigned char>(text[i + j]);
            valid =
                j == 1 ? byte >= second_low && byte <= second_high : byte >= 0x80 && byte <= 0xBF;
        }
        if (valid) {
            result.append(text, i, length);
            i += length;
        } else {
            result += "\xEF\xBF\xBD";
            ++i;
        }
    }
    return result;
}

/// The longest prefix of valid UTF-8 `text` with at most `limit` bytes that ends on a character
/// boundary.
std::string
utf8_prefix(const std::string& text, uint64_t limit) {
    if (text.size() <= limit) {
        return text;
    }
    uint64_t end = limit;
    while (end > 0 && (static_cast<unsigned char>(text[end]) & 0xC0U) == 0x80U) {
        --end;
    }
    return text.substr(0, end);
}

std::string
join(const std::vector<std::string>& segments) {
    std::string path;
    for (const auto& segment : segments) {
        path += path.empty() ? segment : "." + segment;
    }
    return path;
}

void
flatten(const JsonType& node,
        std::vector<std::string>& segments,
        bool include_file_name,
        JsonType& out) {
    if (node.is_object()) {
        for (const auto& item : node.items()) {
            segments.push_back(item.key());
            if (!denied(segments, include_file_name) || segments.size() == 1 ||
                (segments.front() == "signals" && segments.size() == 2)) {
                flatten(item.value(), segments, include_file_name, out);
            }
            segments.pop_back();
        }
        return;
    }
    if (node.is_array()) {
        const bool signal_list = segments.size() == 1 && segments.front() == "signals";
        if (!signal_list && (segments.empty() || ARRAY_KEYS.count(segments.back()) == 0)) {
            return;
        }
        const uint64_t limit = std::min<uint64_t>(node.size(), MAX_ARRAY_ITEMS);
        for (uint64_t i = 0; i < limit; ++i) {
            segments.push_back(std::to_string(i));
            flatten(node[i], segments, include_file_name, out);
            segments.pop_back();
        }
        return;
    }
    if (denied(segments, include_file_name)) {
        return;
    }
    const std::string key = join(segments);
    if (node.is_number_float()) {
        out[key] = round_significant(node.get<double>(), 6);
    } else if (node.is_number() || node.is_boolean() || node.is_null()) {
        out[key] = node;
    } else if (node.is_string()) {
        const bool indexed = is_number_key(segments.back()) && segments.size() > 1;
        const std::string& field = indexed ? segments[segments.size() - 2] : segments.back();
        const bool file_name =
            segments.size() == 2 && segments.front() == "input" && segments.back() == "file_name";
        if (STRING_KEYS.count(field) > 0 || file_name) {
            out[key] = utf8_prefix(sanitize_utf8(node.get<std::string>()), MAX_STRING_LENGTH);
        }
    }
}

/// Lower values are dropped first when the evidence exceeds MAX_EVIDENCE_BYTES.
int
evidence_priority(const std::string& key) {
    for (const char* suffix : {".p01", ".p05", ".p25", ".p75", ".p99"}) {
        const std::string tail = suffix;
        if (key.size() > tail.size() && key.ends_with(tail)) {
            return 0;
        }
    }
    if (key.find(".sample.") != std::string::npos) {
        return 1;
    }
    if (key.find(".cumulative_explained_variance.") != std::string::npos ||
        key.find(".explained_variance_ratio.") != std::string::npos ||
        key.find(".sizes.") != std::string::npos) {
        return 2;
    }
    return 3;
}

/// Keeps the least nested keys (ties in key order) whose compact JSON fits in
/// MAX_EVIDENCE_BYTES. Used when the evidence is still too large after the priority tiers.
JsonType
fit_evidence(const JsonType& evidence) {
    struct Entry {
        uint64_t depth;
        std::string key;
        uint64_t bytes;
    };
    std::vector<Entry> entries;
    entries.reserve(evidence.size());
    for (const auto& item : evidence.items()) {
        const std::string& key = item.key();
        // "key":value in compact form; commas are added below.
        entries.push_back(Entry{static_cast<uint64_t>(std::count(key.begin(), key.end(), '.')),
                                key,
                                JsonType(key).dump().size() + 1 + item.value().dump().size()});
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return a.depth != b.depth ? a.depth < b.depth : a.key < b.key;
    });
    JsonType kept = JsonType::object();
    uint64_t used = 2;  // the braces
    for (const auto& entry : entries) {
        const uint64_t cost = entry.bytes + (kept.empty() ? 0 : 1);
        if (used + cost <= MAX_EVIDENCE_BYTES) {
            kept[entry.key] = evidence.at(entry.key);
            used += cost;
        }
    }
    return kept;
}

std::string
lowercase(std::string text) {
    std::transform(
        text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
    return text;
}

/// Shortens text for messages and stored model output without splitting a UTF-8 character.
std::string
truncate(const std::string& text, uint64_t limit) {
    const std::string clean = sanitize_utf8(text);
    return clean.size() <= limit ? clean : utf8_prefix(clean, limit) + "...";
}

/// Extracts the JSON object from a reply that may be wrapped in a Markdown code fence or
/// surrounded by prose.
bool
extract_json_object(const std::string& reply, JsonType& parsed, std::string& error) {
    const auto begin = reply.find('{');
    const auto end = reply.rfind('}');
    if (begin == std::string::npos || end == std::string::npos || end < begin) {
        error = "reply does not contain a JSON object";
        return false;
    }
    try {
        parsed = JsonType::parse(reply.substr(begin, end - begin + 1));
    } catch (const std::exception& e) {
        error = std::string("reply is not valid JSON: ") + truncate(e.what(), 160);
        return false;
    }
    if (!parsed.is_object()) {
        error = "reply JSON is not an object";
        return false;
    }
    return true;
}

bool
bounded_string(const JsonType& item, const char* field, uint64_t limit, std::string& value) {
    if (!item.contains(field) || !item[field].is_string()) {
        return false;
    }
    value = item[field].get<std::string>();
    return !value.empty() && value.size() <= limit;
}

/// Keeps known evidence keys; returns false when none remain.
bool
filter_evidence(const JsonType& item,
                const JsonType& evidence,
                JsonType& kept,
                std::string& reason) {
    kept = JsonType::array();
    if (!item.contains("evidence") || !item["evidence"].is_array() || item["evidence"].empty()) {
        reason = "missing evidence keys";
        return false;
    }
    std::vector<std::string> unknown;
    for (const auto& key : item["evidence"]) {
        if (!key.is_string()) {
            continue;
        }
        const std::string text = key.get<std::string>();
        if (evidence.contains(text)) {
            if (kept.size() < MAX_EVIDENCE_PER_ITEM) {
                kept.push_back(text);
            }
        } else {
            unknown.push_back(text);
        }
    }
    if (kept.empty()) {
        reason = "no cited evidence key exists in the request";
        if (!unknown.empty()) {
            reason += " (unknown: " + truncate(unknown.front(), 80) + ")";
        }
        return false;
    }
    return true;
}

std::string
join_text(std::initializer_list<const std::string*> parts) {
    std::string text;
    for (const std::string* part : parts) {
        text += *part;
        text += ' ';
    }
    return text;
}

bool
overclaims(const std::string& text, std::string& phrase) {
    const std::string lowered = lowercase(text);
    for (const auto& candidate : OVERCLAIMS) {
        if (lowered.find(candidate) != std::string::npos) {
            phrase = candidate;
            return true;
        }
    }
    return false;
}

struct ParsedUrl {
    std::string scheme;
    std::string host;
    std::string path;
};

ParsedUrl
parse_url(const std::string& url) {
    ParsedUrl parsed;
    const auto scheme_end = url.find("://");
    if (scheme_end == std::string::npos) {
        throw ProfileError("AI endpoint must start with http:// or https://");
    }
    parsed.scheme = lowercase(url.substr(0, scheme_end));
    if (parsed.scheme != "http" && parsed.scheme != "https") {
        throw ProfileError("AI endpoint must start with http:// or https://");
    }
    const auto rest = url.substr(scheme_end + 3);
    const auto slash = rest.find('/');
    parsed.host = rest.substr(0, slash);
    parsed.path = slash == std::string::npos ? "" : rest.substr(slash);
    if (parsed.host.empty()) {
        throw ProfileError("AI endpoint has no host");
    }
    if (parsed.host.find('@') != std::string::npos) {
        throw ProfileError(
            "credentials in the AI endpoint URL are not allowed; use the API key environment "
            "variable");
    }
    return parsed;
}

std::string
request_url(const AiOptions& options) {
    std::string endpoint = options.endpoint;
    if (endpoint.empty()) {
        endpoint = options.provider == "anthropic" ? "https://api.anthropic.com"
                                                   : "https://api.openai.com/v1";
    }
    parse_url(endpoint);  // rejects malformed URLs and embedded credentials
    const auto query = endpoint.find('?');
    const std::string base = query == std::string::npos ? endpoint : endpoint.substr(0, query);
    const std::string suffix = query == std::string::npos ? "" : endpoint.substr(query);
    std::string trimmed = base;
    while (!trimmed.empty() && trimmed.back() == '/') {
        trimmed.pop_back();
    }
    const char* path = options.provider == "anthropic" ? "/v1/messages" : "/chat/completions";
    return trimmed.ends_with(path) ? trimmed + suffix : trimmed + path + suffix;
}

/// scheme://host/path with any query string removed (query strings may carry keys).
std::string
display_endpoint(const std::string& url) {
    const auto query = url.find('?');
    return query == std::string::npos ? url : url.substr(0, query);
}

std::string
reply_text(const std::string& provider, const std::string& body) {
    const JsonType parsed = JsonType::parse(body);
    if (provider == "anthropic") {
        std::string text;
        for (const auto& block : parsed.at("content")) {
            if (block.value("type", "") == "text") {
                text += block.at("text").get<std::string>();
            }
        }
        return text;
    }
    return parsed.at("choices").at(0).at("message").at("content").get<std::string>();
}

}  // namespace

void
validate_ai_options(const AiOptions& options) {
    if (options.provider != "none" && options.provider != "openai" &&
        options.provider != "anthropic") {
        throw ProfileError("unsupported AI provider '" + options.provider +
                           "'; expected none, openai or anthropic");
    }
    if (options.language != "en" && options.language != "zh") {
        throw ProfileError("unsupported AI language '" + options.language + "'; expected en or zh");
    }
    if (std::isnan(options.timeout_seconds) || options.timeout_seconds <= 0.0 ||
        options.timeout_seconds > MAX_AI_TIMEOUT_SECONDS) {
        throw ProfileError("--ai-timeout must be in (0, 3600] seconds");
    }
    if (options.max_retries > MAX_AI_RETRIES) {
        throw ProfileError("--ai-retries must be at most " + std::to_string(MAX_AI_RETRIES));
    }
    if (options.api_key_env.empty()) {
        throw ProfileError("--ai-api-key-env must name an environment variable");
    }
}

std::string
redact(const std::string& text, const std::string& secret) {
    if (secret.empty()) {
        return text;
    }
    std::string result = text;
    for (auto position = result.find(secret); position != std::string::npos;
         position = result.find(secret, position)) {
        result.replace(position, secret.size(), "[REDACTED]");
        position += 10;
    }
    return result;
}

/// Redacts `secret` in every string of a JSON value, member names included: a rejected item
/// keeps the provider's object, whose names may carry the key once decoded.
void
redact_strings(JsonType& node, const std::string& secret) {
    if (secret.empty()) {
        return;
    }
    if (node.is_string()) {
        node = redact(node.get<std::string>(), secret);
    } else if (node.is_array()) {
        for (auto& child : node) {
            redact_strings(child, secret);
        }
    } else if (node.is_object()) {
        JsonType redacted = JsonType::object();
        for (const auto& item : node.items()) {
            JsonType value = std::move(item.value());
            redact_strings(value, secret);
            redacted[redact(item.key(), secret)] = std::move(value);
        }
        node = std::move(redacted);
    }
}

JsonType
build_evidence(const JsonType& report, bool include_file_name, uint64_t* dropped_keys) {
    JsonType evidence = JsonType::object();
    std::vector<std::string> segments;
    flatten(report, segments, include_file_name, evidence);
    const uint64_t flattened = evidence.size();
    for (int tier = 0; tier < 3 && evidence.dump().size() > MAX_EVIDENCE_BYTES; ++tier) {
        JsonType reduced = JsonType::object();
        for (const auto& item : evidence.items()) {
            if (evidence_priority(item.key()) > tier) {
                reduced[item.key()] = item.value();
            }
        }
        evidence = reduced;
    }
    if (evidence.dump().size() > MAX_EVIDENCE_BYTES) {
        evidence = fit_evidence(evidence);
    }
    if (dropped_keys != nullptr) {
        *dropped_keys = flattened - evidence.size();
    }
    return evidence;
}

JsonType
build_prompt(const JsonType& evidence, const std::string& language) {
    const JsonType schema = JsonType{
        {"summary", "string, at most 800 characters"},
        {"findings",
         JsonType::array(
             {JsonType{{"title", "string"},
                       {"severity", "info | notice | warning | critical"},
                       {"detail", "string"},
                       {"evidence", JsonType::array({"EVIDENCE key copied exactly"})}}})},
        {"recommendations",
         JsonType::array({JsonType{
             {"title", "string"},
             {"category",
              "index | metric | quantization | transformation | evaluation | data_quality"},
             {"hypothesis", "string: what to try and the expected effect"},
             {"rationale", "string: why the cited metrics suggest it"},
             {"validation", "string: the benchmark that would confirm or reject it"},
             {"confidence", "low | medium | high"},
             {"evidence", JsonType::array({"EVIDENCE key copied exactly"})}}})},
        {"uncertainties", JsonType::array({"string"})},
        {"additional_measurements", JsonType::array({"string"})}};
    std::string system =
        R"(You are a vector-search data analyst. You explain a deterministic profile of a vector dataset to a user of VSAG, a vector indexing library.
Rules:
1. Use only the metrics in EVIDENCE. Do not invent values or assume facts that are not in it.
2. Every finding and recommendation must cite one or more EVIDENCE keys, copied exactly.
3. Recommendations are hypotheses to validate with reproducible VSAG benchmarks (tools/eval eval_performance at a fixed recall target), never guaranteed or optimal settings. Do not use words such as "guaranteed", "always optimal", "best possible" or "definitely".
4. Keys with sample in their scope have sampling error; quantization results are a simulation, not a benchmark; say so when they matter.
5. Use VSAG terms: dense indexes hgraph (graph, ef_search), ivf (buckets, scan_buckets_count), pyramid, brute_force; sparse index sindi (term_id_limit, doc_prune_ratio, use_quantization, window_size); base_quantization_type fp32, fp16, bf16, sq8, sq4, sq8_uniform, sq4_uniform, pq, rabitq, or tq with a tq_chain of pca/rom/fht/mrle; use_reorder with precise_quantization_type; metrics l2 (squared L2), ip (1 - inner product), cosine (1 - cosine similarity).
6. Cover notable characteristics, their likely effect on index construction and search, suggestions, uncertainty, and the measurements still needed before a decision.
7. Reply with one JSON object only, no Markdown, matching RESPONSE_SCHEMA.
)";
    if (language == "zh") {
        system +=
            R"(8. Write every text value in Simplified Chinese; keep EVIDENCE keys, enum values and VSAG parameter names in English.
)";
    }
    system += "RESPONSE_SCHEMA:\n";
    system += schema.dump(2);
    std::string user =
        "EVIDENCE (aggregate dataset metrics; no vectors, row ids or file contents):\n";
    user += evidence.dump(1);
    user += "\nReturn the JSON object now.";
    return JsonType{{"system", system}, {"user", user}, {"response_schema", schema}};
}

AiValidation
validate_ai_response(const std::string& reply, const JsonType& evidence) {
    AiValidation result;
    if (reply.size() > MAX_REPLY_BYTES) {
        result.errors.push_back("reply exceeds " + std::to_string(MAX_REPLY_BYTES) + " bytes");
        return result;
    }
    JsonType parsed;
    std::string error;
    if (!extract_json_object(reply, parsed, error)) {
        result.errors.push_back(error);
        return result;
    }
    std::string summary;
    if (!bounded_string(parsed, "summary", MAX_TEXT, summary)) {
        result.errors.emplace_back("summary must be a non-empty string of at most 2000 characters");
        return result;
    }
    for (const char* field :
         {"findings", "recommendations", "uncertainties", "additional_measurements"}) {
        if (!parsed.contains(field) || !parsed[field].is_array()) {
            result.errors.push_back(std::string(field) + " must be an array");
        }
    }
    if (!result.errors.empty()) {
        return result;
    }
    JsonType findings = JsonType::array();
    for (const auto& item : parsed["findings"]) {
        std::string title;
        std::string detail;
        std::string severity;
        std::string reason;
        JsonType kept;
        if (!item.is_object() || !bounded_string(item, "title", 200, title) ||
            !bounded_string(item, "detail", MAX_TEXT, detail) ||
            !bounded_string(item, "severity", 20, severity)) {
            reason = "malformed finding";
        } else if (SEVERITIES.count(severity) == 0) {
            reason = "unknown severity '" + truncate(severity, 20) + "'";
        } else if (!filter_evidence(item, evidence, kept, reason)) {
            // reason filled by filter_evidence
        } else if (findings.size() >= MAX_ITEMS) {
            reason = "too many findings";
        }
        if (!reason.empty()) {
            result.rejected.push_back(
                JsonType{{"reason", reason}, {"kind", "finding"}, {"item", item}});
            continue;
        }
        findings.push_back(JsonType{
            {"title", title}, {"severity", severity}, {"detail", detail}, {"evidence", kept}});
    }
    JsonType recommendations = JsonType::array();
    for (const auto& item : parsed["recommendations"]) {
        std::string title;
        std::string category;
        std::string hypothesis;
        std::string rationale;
        std::string validation;
        std::string confidence;
        std::string reason;
        std::string phrase;
        JsonType kept;
        if (!item.is_object() || !bounded_string(item, "title", 200, title) ||
            !bounded_string(item, "category", 40, category) ||
            !bounded_string(item, "hypothesis", MAX_TEXT, hypothesis) ||
            !bounded_string(item, "rationale", MAX_TEXT, rationale) ||
            !bounded_string(item, "validation", MAX_TEXT, validation) ||
            !bounded_string(item, "confidence", 20, confidence)) {
            reason =
                "malformed recommendation (title, category, hypothesis, rationale, validation and "
                "confidence are required)";
        } else if (CATEGORIES.count(category) == 0) {
            reason = "unknown category '" + truncate(category, 40) + "'";
        } else if (CONFIDENCES.count(confidence) == 0) {
            reason = "unknown confidence '" + truncate(confidence, 20) + "'";
        } else if (overclaims(join_text({&title, &hypothesis, &rationale, &validation}), phrase)) {
            reason = "presented as a guaranteed or optimal setting ('" + phrase +
                     "'); recommendations must be hypotheses";
        } else if (!filter_evidence(item, evidence, kept, reason)) {
            // reason filled by filter_evidence
        } else if (recommendations.size() >= MAX_ITEMS) {
            reason = "too many recommendations";
        }
        if (!reason.empty()) {
            result.rejected.push_back(
                JsonType{{"reason", reason}, {"kind", "recommendation"}, {"item", item}});
            continue;
        }
        recommendations.push_back(JsonType{{"title", title},
                                           {"category", category},
                                           {"hypothesis", hypothesis},
                                           {"rationale", rationale},
                                           {"validation", validation},
                                           {"confidence", confidence},
                                           {"evidence", kept}});
    }
    auto strings = [](const JsonType& array) {
        JsonType kept = JsonType::array();
        for (const auto& entry : array) {
            if (entry.is_string() && !entry.get<std::string>().empty() && kept.size() < MAX_ITEMS) {
                kept.push_back(truncate(entry.get<std::string>(), MAX_TEXT));
            }
        }
        return kept;
    };
    result.analysis =
        JsonType{{"summary", summary},
                 {"findings", findings},
                 {"recommendations", recommendations},
                 {"uncertainties", strings(parsed["uncertainties"])},
                 {"additional_measurements", strings(parsed["additional_measurements"])}};
    if (findings.empty() && recommendations.empty()) {
        result.errors.emplace_back("no finding or recommendation passed validation");
        return result;
    }
    result.ok = true;
    return result;
}

JsonType
run_ai_analysis(const JsonType& report,
                const AiOptions& options,
                HttpTransport& transport,
                std::ostream& warnings) {
    JsonType section = JsonType{{"status", "failed"},
                                {"provider", options.provider},
                                {"model", options.model},
                                {"prompt_version", AI_PROMPT_VERSION},
                                {"language", options.language},
                                {"raw_vectors_sent", false}};
    const char* raw_key = std::getenv(options.api_key_env.c_str());
    const std::string key = raw_key != nullptr ? raw_key : "";
    const auto start = std::chrono::steady_clock::now();
    try {
        validate_ai_options(options);
        const std::string url = request_url(options);
        section["endpoint"] = display_endpoint(url);
        if (!options.dry_run && options.model.empty()) {
            throw ProfileError("no AI model configured; set --ai-model or VSAG_PROFILE_AI_MODEL");
        }
        uint64_t dropped_keys = 0;
        const JsonType evidence = build_evidence(report, options.include_file_name, &dropped_keys);
        const JsonType prompt = build_prompt(evidence, options.language);
        JsonType body;
        if (options.provider == "anthropic") {
            body = JsonType{
                {"model", options.model},
                {"max_tokens", options.max_output_tokens},
                {"temperature", 0},
                {"system", prompt["system"]},
                {"messages",
                 JsonType::array({JsonType{{"role", "user"}, {"content", prompt["user"]}}})}};
        } else {
            body = JsonType{
                {"model", options.model},
                {"temperature", 0},
                {"max_tokens", options.max_output_tokens},
                {"messages",
                 JsonType::array({JsonType{{"role", "system"}, {"content", prompt["system"]}},
                                  JsonType{{"role", "user"}, {"content", prompt["user"]}}})}};
        }
        const std::string payload = body.dump();
        section["request_sha256"] = sha256_hex(payload);
        section["request_bytes"] = payload.size();
        section["evidence_key_count"] = evidence.size();
        section["evidence_dropped_keys"] = dropped_keys;
        if (!options.request_output.empty()) {
            std::ofstream file(options.request_output, std::ios::binary);
            if (!file) {
                throw ProfileError("cannot write --ai-request-out file " + options.request_output);
            }
            file << body.dump(2) << "\n";
        }
        if (options.dry_run) {
            section["status"] = "dry_run";
            section["message"] = "request built but not sent (--ai-dry-run)";
            return section;
        }
        if (options.provider == "anthropic" && key.empty()) {
            throw ProfileError("environment variable " + options.api_key_env + " is not set");
        }
        HttpRequest request;
        request.url = url;
        request.body = payload;
        request.max_response_bytes = MAX_RESPONSE_BYTES;
        request.headers.emplace_back("Content-Type", "application/json");
        if (options.provider == "anthropic") {
            request.headers.emplace_back("x-api-key", key);
            request.headers.emplace_back("anthropic-version", "2023-06-01");
        } else if (!key.empty()) {
            request.headers.emplace_back(
                options.auth_header,
                lowercase(options.auth_header) == "authorization" ? "Bearer " + key : key);
        }
        HttpResponse response;
        uint64_t attempts = 0;
        // Each attempt is limited to the timeout, and all attempts together with the delays
        // between them to (retries + 1) x timeout.
        using Clock = std::chrono::steady_clock;
        const auto deadline =
            Clock::now() +
            std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(
                options.timeout_seconds * static_cast<double>(options.max_retries + 1)));
        for (;;) {
            const double remaining = std::chrono::duration<double>(deadline - Clock::now()).count();
            request.timeout_seconds = std::min(options.timeout_seconds, std::max(remaining, 0.001));
            ++attempts;
            response = transport.Post(request);
            const bool retryable = (!response.error.empty() && !response.too_large) ||
                                   response.status == 429 || response.status >= 500;
            if (!retryable || attempts > options.max_retries) {
                break;
            }
            const auto delay =
                std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(
                    options.retry_delay_seconds * static_cast<double>(attempts)));
            if (Clock::now() + delay >= deadline) {
                break;
            }
            std::this_thread::sleep_for(delay);
        }
        section["attempts"] = attempts;
        // Redact before anything is truncated, parsed or reported: a key cut by truncation would
        // no longer match and could leak in part.
        response.error = redact(response.error, key);
        response.body = redact(response.body, key);
        if (response.too_large || response.body.size() > MAX_RESPONSE_BYTES) {
            throw ProfileError("provider response exceeds " + std::to_string(MAX_RESPONSE_BYTES) +
                               " bytes");
        }
        if (!response.error.empty()) {
            throw ProfileError("provider unreachable: " + truncate(response.error, MAX_ERROR_TEXT));
        }
        if (response.status != 200) {
            throw ProfileError("provider returned HTTP " + std::to_string(response.status) + ": " +
                               truncate(response.body, MAX_ERROR_TEXT));
        }
        std::string text;
        try {
            // JSON escapes can hide a key from the redaction of the raw body; redact the
            // decoded text as well.
            text = redact(reply_text(options.provider, response.body), key);
        } catch (const std::exception&) {
            section["status"] = "invalid_response";
            throw ProfileError("provider response does not follow the " + options.provider +
                               " API format");
        }
        AiValidation validation = validate_ai_response(text, evidence);
        section["rejected_items"] = validation.rejected;
        if (!validation.ok) {
            section["status"] = "invalid_response";
            JsonType errors = JsonType::array();
            for (const auto& error : validation.errors) {
                errors.push_back(error);
            }
            section["validation_errors"] = errors;
            throw ProfileError("model reply failed validation: " + validation.errors.front());
        }
        section["status"] = "ok";
        section["analysis"] = validation.analysis;
    } catch (const std::exception& error) {
        if (section["status"] != "invalid_response") {
            section["status"] = "failed";
        }
        section["message"] = sanitize_utf8(redact(error.what(), key));
        warnings << "warning: AI analysis " << section["status"].get<std::string>() << ": "
                 << section["message"].get<std::string>()
                 << ". The deterministic JSON and HTML reports are generated without it." << '\n';
    }
    section["latency_seconds"] =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    // Last line of defense for analysis, rejected items and validation errors.
    redact_strings(section, key);
    return section;
}

}  // namespace vsag::dataset_profile
