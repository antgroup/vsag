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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "vsag/dataset.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/index.h"

namespace vsag::autotune {

using JsonType = nlohmann::json;

/// Metrics exposed by the experimental AutoTune evaluator.
enum class Metric {
    UNSPECIFIED = 0,
    RECALL_AT_K,
    LATENCY_AVG_MS,
    LATENCY_P99_MS,
    QPS,
    INDEX_MEMORY_MB,
    INDEX_SIZE_MB,
    BUILD_SECONDS,
    SEARCH_SECONDS,
    BUILD_AND_SEARCH_SECONDS,
};

/// An inclusive upper or lower bound, according to the metric's direction.
struct Constraint {
    Metric metric{Metric::UNSPECIFIED};
    /// Minimum for recall/QPS and maximum for all other metrics.
    double value{0.0};
};

/// Query workload evaluated for every search candidate.
struct Workload {
    /// Query vectors. Buffers referenced by a non-owning Dataset must outlive the tuning call.
    DatasetPtr queries;
    /// Ground truth required when recall is a constraint or objective.
    DatasetPtr ground_truth;
    /// Number of neighbors requested from every query.
    uint64_t top_k{0};
    /// Number of evaluator search threads.
    uint64_t concurrency{1};
};

/// Execution and report options shared by index and search tuning.
struct Config {
    /// Directory for generated reports and index artifacts.
    std::string workspace_path{"/tmp/vsag_autotune"};
    /// Optional report destination; an empty value generates a path under workspace_path.
    std::string report_path;
    /// Maximum number of concrete search trials planned for the request.
    uint64_t max_trials{1000};
    /// Keep every generated index artifact instead of only the selected artifact.
    bool keep_intermediate{false};
    bool include_raw_evaluation{false};
};

/// Candidate space for one concrete index type.
struct IndexSpace {
    /// Concrete index type, currently hgraph or ivf.
    std::string name;
    /// JSON object whose scalar, array, and range leaves define build candidates.
    std::string create_parameter_space{"{}"};
    /// JSON object whose scalar, array, and range leaves define search candidates.
    std::string search_parameter_space{"{}"};
};

/// Input for jointly tuning index construction and search.
struct IndexRequest {
    /// Base vectors and IDs. Buffers referenced by a non-owning Dataset must outlive TuneIndex.
    DatasetPtr base;
    /// l2, ip, or cosine.
    std::string metric_type;
    Workload workload;
    std::vector<IndexSpace> index_spaces;
    std::vector<Constraint> constraints;
    Metric objective{Metric::UNSPECIFIED};
    Config config;
};

/// Input for tuning search parameters on an already built index.
struct SearchRequest {
    /// Existing index reused in place; TuneSearch neither rebuilds nor takes exclusive ownership.
    IndexPtr index;
    /// Concrete type of index, including pyramid for search-only tuning.
    std::string index_name;
    /// Base vectors and IDs used to evaluate recall and validate ground truth.
    DatasetPtr base;
    /// l2, ip, or cosine; it must match the existing index and datasets.
    std::string metric_type;
    Workload workload;
    /// JSON search candidate space; missing supported fields receive built-in proposals.
    std::string parameter_space{"{}"};
    std::vector<Constraint> constraints;
    Metric objective{Metric::UNSPECIFIED};
    Config config;
};

/// Successful index-tuning result.
struct IndexResult {
    /// Loaded selected index, ready for queries.
    IndexPtr index;
    std::string index_name;
    /// Concrete JSON parameters required to recreate and deserialize the selected index.
    std::string create_parameters;
    /// Concrete JSON parameters recommended for queries.
    std::string search_parameters;
    /// Validated metric values as a JSON object.
    std::string metrics;
    /// Selected serialized artifact; the caller removes it when it is no longer needed.
    std::string artifact_path;
    std::string report_path;
    /// Complete report as JSON.
    std::string report;
};

/// Successful search-only tuning result.
struct SearchResult {
    /// Concrete JSON parameters recommended for the existing index.
    std::string parameters;
    /// Validated metric values as a JSON object.
    std::string metrics;
    std::string report_path;
    /// Complete report as JSON.
    std::string report;
};

/**
 * Experimental, synchronous build-tree tool API. It is not installed as part of the VSAG SDK.
 *
 * Returns INVALID_ARGUMENT for invalid requests or no feasible candidate, and an execution error
 * for evaluation, artifact, or report failures. On success, only the selected artifact is retained
 * unless Config::keep_intermediate is true.
 */
tl::expected<IndexResult, Error>
TuneIndex(const IndexRequest& request);

/**
 * Synchronously evaluates search candidates against SearchRequest::index without rebuilding it.
 *
 * Returns INVALID_ARGUMENT for invalid requests or no feasible candidate, and an execution error
 * for evaluation or report failures.
 */
tl::expected<SearchResult, Error>
TuneSearch(const SearchRequest& request);

/// Offline JSON adapter used by the CLI. Validation failures are returned as structured JSON.
JsonType
RunAutoTune(const JsonType& request);

/// Returns the compact, recommendation-first JSON shown by the CLI.
std::string
FormatResultSummaryForCli(const JsonType& report);

}  // namespace vsag::autotune
