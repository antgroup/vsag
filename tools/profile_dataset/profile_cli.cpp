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

#include "profile_cli.h"

#include <omp.h>

#include <argparse/argparse.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

#include "ai_analysis.h"
#include "dataset_profiler.h"
#include "html_report.h"
#include "version.h"

namespace vsag::dataset_profile {

namespace {

std::string
environment_or(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? std::string(value) : fallback;
}

std::vector<std::string>
split_list(const std::string& text) {
    std::vector<std::string> items;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        while (!item.empty() && item.front() == ' ') {
            item.erase(item.begin());
        }
        while (!item.empty() && item.back() == ' ') {
            item.pop_back();
        }
        if (!item.empty()) {
            items.push_back(item);
        }
    }
    return items;
}

/// Writes through a temporary file and renames it, so readers never see a partial report.
void
write_file(const std::string& path, const std::string& content) {
    const std::string temporary = path + ".tmp";
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file) {
            throw ProfileError("cannot write " + path);
        }
        file << content;
        if (!file) {
            throw ProfileError("failed while writing " + path);
        }
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        throw ProfileError("cannot move " + temporary + " to " + path + ": " + error.message());
    }
}

JsonType
read_report(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw ProfileError("cannot read profile " + path);
    }
    JsonType report;
    try {
        report = JsonType::parse(file);
    } catch (const std::exception& e) {
        throw ProfileError("profile " + path + " is not valid JSON: " + e.what());
    }
    if (!report.is_object() || report.value("schema", "") != REPORT_SCHEMA ||
        !report.value("schema_version", "").starts_with("1.")) {
        throw ProfileError("profile " + path + " is not a " + std::string(REPORT_SCHEMA) +
                           " v1 report");
    }
    // Split names become element ids in the HTML report; the schema only knows these two.
    const auto splits = report.find("splits");
    if (splits == report.end() || !splits->is_object()) {
        throw ProfileError("profile " + path + " has no splits object");
    }
    for (const auto& item : splits->items()) {
        if (item.key() != "train" && item.key() != "test") {
            throw ProfileError("profile " + path + " has an unsupported split name; expected " +
                               "train or test");
        }
    }
    return report;
}

void
add_arguments(argparse::ArgumentParser& parser) {
    const ProfileOptions defaults;
    const AiOptions ai_defaults;
    parser.add_description(
        "Profiles a VSAG-compatible HDF5 dataset (dense or sparse) into a versioned JSON report "
        "and a "
        "self-contained HTML report, with optional AI-assisted interpretation of the aggregate "
        "metrics.");
    parser.add_argument("--datapath", "-d")
        .default_value(std::string())
        .help("VSAG-compatible HDF5 file");
    parser.add_argument("--output", "-o")
        .default_value(std::string())
        .help("JSON report path (default: <dataset stem>.profile.json)");
    parser.add_argument("--html")
        .default_value(std::string())
        .help("HTML report path (default: <dataset stem>.profile.html)");
    parser.add_argument("--no-html")
        .default_value(false)
        .implicit_value(true)
        .help("do not write the HTML report");
    parser.add_argument("--from-profile")
        .default_value(std::string())
        .help(
            "load an existing JSON report instead of profiling (re-render HTML and/or run the AI "
            "analysis)");
    parser.add_argument("--splits")
        .default_value(std::string())
        .help("comma-separated splits to profile: train,test (default: all present)");
    parser.add_argument("--metric")
        .default_value(defaults.metric)
        .help("auto (HDF5 'distance' attribute), l2, ip or cosine");
    parser.add_argument("--seed")
        .default_value(defaults.seed)
        .scan<'u', uint64_t>()
        .help("random seed for every sample");
    parser.add_argument("--sample-size")
        .default_value(defaults.sample_size)
        .scan<'u', uint64_t>()
        .help("rows sampled per split");
    parser.add_argument("--knn-queries")
        .default_value(defaults.knn_queries)
        .scan<'u', uint64_t>()
        .help("sampled rows whose exact k-NN is searched in the full split");
    parser.add_argument("--knn-k")
        .default_value(defaults.knn_k)
        .scan<'u', uint64_t>()
        .help("neighbors per query");
    parser.add_argument("--pair-count")
        .default_value(defaults.pair_count)
        .scan<'u', uint64_t>()
        .help("random sampled pairs for pairwise distances");
    parser.add_argument("--histogram-bins")
        .default_value(defaults.histogram_bins)
        .scan<'u', uint64_t>();
    parser.add_argument("--clusters")
        .default_value(defaults.clusters)
        .scan<'u', uint64_t>()
        .help("k of the sampled k-means summary (0 disables)");
    parser.add_argument("--kmeans-iterations")
        .default_value(defaults.kmeans_iterations)
        .scan<'u', uint64_t>();
    parser.add_argument("--pca-max-dim")
        .default_value(defaults.pca_max_dim)
        .scan<'u', uint64_t>()
        .help("skip PCA above this dimensionality");
    parser.add_argument("--projection-points")
        .default_value(defaults.projection_points)
        .scan<'u', uint64_t>();
    parser.add_argument("--hubness-sample")
        .default_value(defaults.hubness_sample)
        .scan<'u', uint64_t>();
    parser.add_argument("--hubness-k").default_value(defaults.hubness_k).scan<'u', uint64_t>();
    parser.add_argument("--quantization-base")
        .default_value(defaults.quantization_base)
        .scan<'u', uint64_t>();
    parser.add_argument("--quantization-queries")
        .default_value(defaults.quantization_queries)
        .scan<'u', uint64_t>();
    parser.add_argument("--quantization-k")
        .default_value(defaults.quantization_k)
        .scan<'u', uint64_t>();
    parser.add_argument("--near-duplicate-threshold")
        .default_value(defaults.near_duplicate_threshold)
        .scan<'g', double>()
        .help("near-duplicate distance as a fraction of the median pairwise distance");
    parser.add_argument("--duplicate-memory-mb")
        .default_value(defaults.duplicate_memory_mb)
        .scan<'u', uint64_t>()
        .help("memory budget for exact duplicate detection (24 bytes per row)");
    parser.add_argument("--disable")
        .default_value(std::string())
        .help(
            "comma-separated optional sections: "
            "duplicates,knn,pca,clustering,hubness,quantization,ground_truth,labels");
    parser.add_argument("--threads")
        .default_value(defaults.threads)
        .scan<'u', uint64_t>()
        .help("worker threads, 0 = all cores (results do not depend on it)");
    parser.add_argument("--chunk-mb")
        .default_value(defaults.chunk_mb)
        .scan<'u', uint64_t>()
        .help("I/O chunk size (results do not depend on it)");
    parser.add_argument("--quiet", "-q")
        .default_value(false)
        .implicit_value(true)
        .help("print only warnings and errors");

    parser.add_argument("--ai-provider")
        .default_value(environment_or("VSAG_PROFILE_AI_PROVIDER", ai_defaults.provider))
        .help(
            "none (default), openai (OpenAI-compatible chat completions) or anthropic; env "
            "VSAG_PROFILE_AI_PROVIDER");
    parser.add_argument("--ai-endpoint")
        .default_value(environment_or("VSAG_PROFILE_AI_ENDPOINT", ""))
        .help("provider base URL; env VSAG_PROFILE_AI_ENDPOINT");
    parser.add_argument("--ai-model")
        .default_value(environment_or("VSAG_PROFILE_AI_MODEL", ""))
        .help("model name; env VSAG_PROFILE_AI_MODEL");
    parser.add_argument("--ai-api-key-env")
        .default_value(ai_defaults.api_key_env)
        .help(
            "name of the environment variable holding the API key (keys are never accepted as "
            "arguments)");
    parser.add_argument("--ai-auth-header")
        .default_value(ai_defaults.auth_header)
        .help("header carrying the key for openai-compatible endpoints");
    parser.add_argument("--ai-timeout")
        .default_value(ai_defaults.timeout_seconds)
        .scan<'g', double>()
        .help("seconds per request");
    parser.add_argument("--ai-retries")
        .default_value(ai_defaults.max_retries)
        .scan<'u', uint64_t>();
    parser.add_argument("--ai-max-tokens")
        .default_value(ai_defaults.max_output_tokens)
        .scan<'u', uint64_t>();
    parser.add_argument("--ai-language")
        .default_value(environment_or("VSAG_PROFILE_AI_LANGUAGE", ai_defaults.language))
        .help("en or zh");
    parser.add_argument("--ai-include-file-name")
        .default_value(false)
        .implicit_value(true)
        .help("also send the dataset file name");
    parser.add_argument("--ai-dry-run")
        .default_value(false)
        .implicit_value(true)
        .help("build the request without sending it");
    parser.add_argument("--ai-request-out")
        .default_value(std::string())
        .help("save the exact request body for review");
    parser.add_argument("--ai-output")
        .default_value(std::string())
        .help("also write the AI analysis section to this JSON file");
    parser.add_argument("--ai-required")
        .default_value(false)
        .implicit_value(true)
        .help("exit with code 3 when the AI analysis does not succeed (reports are still written)");
}

std::string
default_path(const std::string& source, const std::string& suffix) {
    const std::filesystem::path path(source);
    return (path.stem().string().empty() ? std::string("dataset") : path.stem().string()) + suffix;
}

void
print_summary(const JsonType& report, std::ostream& out) {
    std::map<std::string, uint64_t> severities;
    for (const auto& signal : report.value("signals", JsonType::array())) {
        ++severities[signal.value("severity", "info")];
    }
    out << "  observations:";
    for (const char* severity : {"critical", "warning", "notice", "info"}) {
        out << ' ' << severities[severity] << ' ' << severity;
    }
    out << "\n";
    for (const auto& signal : report.value("signals", JsonType::array())) {
        const std::string severity = signal.value("severity", "info");
        if (severity == "critical" || severity == "warning") {
            out << "    [" << severity << "] " << signal.value("message", "") << "\n";
        }
    }
}

}  // namespace

int
run_cli(int argc, char** argv, std::ostream& out, std::ostream& err, HttpTransport* transport) {
    argparse::ArgumentParser parser(
        "profile_dataset", VSAG_VERSION, argparse::default_arguments::all, false);
    add_arguments(parser);
    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& error) {
        err << "profile_dataset: " << error.what() << "\n" << parser;
        return EXIT_FAILURE_INPUT;
    }
    try {
        ProfileOptions options;
        options.data_path = parser.get<std::string>("--datapath");
        options.splits = split_list(parser.get<std::string>("--splits"));
        options.metric = parser.get<std::string>("--metric");
        options.seed = parser.get<uint64_t>("--seed");
        options.sample_size = parser.get<uint64_t>("--sample-size");
        options.knn_queries = parser.get<uint64_t>("--knn-queries");
        options.knn_k = parser.get<uint64_t>("--knn-k");
        options.pair_count = parser.get<uint64_t>("--pair-count");
        options.histogram_bins = parser.get<uint64_t>("--histogram-bins");
        options.clusters = parser.get<uint64_t>("--clusters");
        options.kmeans_iterations = parser.get<uint64_t>("--kmeans-iterations");
        options.pca_max_dim = parser.get<uint64_t>("--pca-max-dim");
        options.projection_points = parser.get<uint64_t>("--projection-points");
        options.hubness_sample = parser.get<uint64_t>("--hubness-sample");
        options.hubness_k = parser.get<uint64_t>("--hubness-k");
        options.quantization_base = parser.get<uint64_t>("--quantization-base");
        options.quantization_queries = parser.get<uint64_t>("--quantization-queries");
        options.quantization_k = parser.get<uint64_t>("--quantization-k");
        options.near_duplicate_threshold = parser.get<double>("--near-duplicate-threshold");
        options.duplicate_memory_mb = parser.get<uint64_t>("--duplicate-memory-mb");
        for (const auto& section : split_list(parser.get<std::string>("--disable"))) {
            options.disabled_sections.insert(section);
        }
        options.threads = parser.get<uint64_t>("--threads");
        options.chunk_mb = parser.get<uint64_t>("--chunk-mb");
        options.quiet = parser.get<bool>("--quiet");

        AiOptions ai;
        ai.provider = parser.get<std::string>("--ai-provider");
        ai.endpoint = parser.get<std::string>("--ai-endpoint");
        ai.model = parser.get<std::string>("--ai-model");
        ai.api_key_env = parser.get<std::string>("--ai-api-key-env");
        ai.auth_header = parser.get<std::string>("--ai-auth-header");
        ai.timeout_seconds = parser.get<double>("--ai-timeout");
        ai.max_retries = parser.get<uint64_t>("--ai-retries");
        ai.max_output_tokens = parser.get<uint64_t>("--ai-max-tokens");
        ai.language = parser.get<std::string>("--ai-language");
        ai.include_file_name = parser.get<bool>("--ai-include-file-name");
        ai.dry_run = parser.get<bool>("--ai-dry-run");
        ai.request_output = parser.get<std::string>("--ai-request-out");
        const auto ai_output = parser.get<std::string>("--ai-output");
        const bool ai_required = parser.get<bool>("--ai-required");
        validate_ai_options(ai);
        if (ai.dry_run && !ai.Enabled()) {
            throw ProfileError("--ai-dry-run needs --ai-provider");
        }

        const auto from_profile = parser.get<std::string>("--from-profile");
        if (from_profile.empty() == options.data_path.empty()) {
            throw ProfileError("pass exactly one of --datapath or --from-profile");
        }
        if (options.threads > 0) {
            omp_set_num_threads(static_cast<int>(options.threads));
        }
        const bool write_html = !parser.get<bool>("--no-html");
        auto json_path = parser.get<std::string>("--output");
        auto html_path = parser.get<std::string>("--html");
        JsonType report;
        if (!from_profile.empty()) {
            report = read_report(from_profile);
            report.erase("ai_analysis");
            if (json_path.empty()) {
                json_path = from_profile;
            }
            if (html_path.empty()) {
                const std::filesystem::path path(from_profile);
                html_path = (path.parent_path() / path.stem()).string() + ".html";
            }
        } else {
            validate_options(options);
            if (json_path.empty()) {
                json_path = default_path(options.data_path, ".profile.json");
            }
            if (html_path.empty()) {
                html_path = default_path(options.data_path, ".profile.html");
            }
            const ProgressFn progress =
                options.quiet ? ProgressFn()
                              : [&err](const std::string& message) { err << message << '\n'; };
            report = profile_dataset(options, progress);
            // Deterministic reports first: they must exist even if the AI step hangs or fails.
            write_file(json_path, report.dump(2) + "\n");
            if (write_html) {
                write_file(html_path, render_html_report(report));
            }
        }

        std::string ai_status = "disabled";
        if (ai.Enabled()) {
            std::unique_ptr<HttpTransport> owned;
            if (transport == nullptr) {
                owned = make_http_transport();
                transport = owned.get();
            }
            if (!options.quiet) {
                err << "running AI analysis (" << ai.provider << (ai.dry_run ? ", dry run" : "")
                    << "): sending aggregate metrics only" << '\n';
            }
            JsonType section = run_ai_analysis(report, ai, *transport, err);
            ai_status = section.value("status", "failed");
            report["ai_analysis"] = section;
            if (!ai_output.empty()) {
                write_file(ai_output, section.dump(2) + "\n");
            }
        }
        if (ai.Enabled() || from_profile.empty()) {
            write_file(json_path, report.dump(2) + "\n");
        }
        if (write_html) {
            write_file(html_path, render_html_report(report));
        }
        if (!options.quiet) {
            const JsonType& input = report.at("input");
            out << "profile_dataset: " << input.value("file_name", "") << " ("
                << input.value("vector_type", "") << ", metric "
                << report.at("config").value("metric", "") << ")\n"
                << "  JSON report: " << json_path << "\n";
            if (write_html) {
                out << "  HTML report: " << html_path << "\n";
            }
            out << "  metrics SHA-256: " << report.at("determinism").value("metrics_sha256", "")
                << "\n";
            print_summary(report, out);
            out << "  AI analysis: " << ai_status;
            if (report.contains("ai_analysis") && report["ai_analysis"].contains("analysis")) {
                const JsonType& analysis = report["ai_analysis"]["analysis"];
                out << " (" << analysis["findings"].size() << " findings, "
                    << analysis["recommendations"].size() << " recommendations, "
                    << report["ai_analysis"].value("rejected_items", JsonType::array()).size()
                    << " rejected)";
            }
            out << '\n';
        }
        if (ai_required && ai.Enabled() && ai_status != "ok" && ai_status != "dry_run") {
            return EXIT_AI_REQUIRED_FAILED;
        }
        return EXIT_OK;
    } catch (const std::exception& error) {
        err << "profile_dataset: " << error.what() << '\n';
        return EXIT_FAILURE_INPUT;
    }
}

}  // namespace vsag::dataset_profile
