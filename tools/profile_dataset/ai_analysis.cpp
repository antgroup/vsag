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
    return root != "splits" && root != "cross_split" && root != "ground_truth" && root != "labels";
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
        std::string text = node.get<std::string>();
        if (text.size() > MAX_STRING_LENGTH) {
            text = text.substr(0, MAX_STRING_LENGTH);
        }
        out[key] = text;
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

std::string
lowercase(std::string text) {
    std::transform(
        text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
    return text;
}

std::string
truncate(const std::string& text, uint64_t limit) {
    return text.size() <= limit ? text : text.substr(0, limit) + "...";
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
    if (!(options.timeout_seconds > 0.0)) {
        throw ProfileError("--ai-timeout must be positive");
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

JsonType
build_evidence(const JsonType& report, bool include_file_name) {
    JsonType evidence = JsonType::object();
    std::vector<std::string> segments;
    flatten(report, segments, include_file_name, evidence);
    for (int tier = 0; tier < 3 && evidence.dump().size() > MAX_EVIDENCE_BYTES; ++tier) {
        JsonType reduced = JsonType::object();
        for (const auto& item : evidence.items()) {
            if (evidence_priority(item.key()) > tier) {
                reduced[item.key()] = item.value();
            }
        }
        evidence = reduced;
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
        const JsonType evidence = build_evidence(report, options.include_file_name);
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
        request.timeout_seconds = options.timeout_seconds;
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
        for (;;) {
            ++attempts;
            response = transport.Post(request);
            const bool retryable =
                !response.error.empty() || response.status == 429 || response.status >= 500;
            if (!retryable || attempts > options.max_retries) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::duration<double>(
                options.retry_delay_seconds * static_cast<double>(attempts)));
        }
        section["attempts"] = attempts;
        if (!response.error.empty()) {
            throw ProfileError("provider unreachable: " + truncate(response.error, MAX_ERROR_TEXT));
        }
        if (response.status != 200) {
            throw ProfileError("provider returned HTTP " + std::to_string(response.status) + ": " +
                               truncate(response.body, MAX_ERROR_TEXT));
        }
        std::string text;
        try {
            text = reply_text(options.provider, response.body);
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
        section["message"] = redact(error.what(), key);
        warnings << "warning: AI analysis " << section["status"].get<std::string>() << ": "
                 << section["message"].get<std::string>()
                 << ". The deterministic JSON and HTML reports are generated without it." << '\n';
    }
    section["latency_seconds"] =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return section;
}

}  // namespace vsag::dataset_profile
