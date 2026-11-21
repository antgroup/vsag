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

/// Evaluation options shared by index and search tuning.
struct Config {
    /// Directory for generated index artifacts.
    std::string workspace_path{"/tmp/vsag_autotune"};
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
    /// JSON object with concrete values, explicit $choices, or $range build expressions. Bare
    /// arrays are concrete values.
    std::string create_parameter_space{"{}"};
    /// JSON object with concrete values, explicit $choices, or $range search expressions. Bare
    /// arrays are concrete values.
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
    Workload workload;
    /// JSON search space with explicit $choices or $range candidate expressions; missing supported
    /// fields receive built-in proposals.
    std::string parameter_space{"{}"};
    std::vector<Constraint> constraints;
    Metric objective{Metric::UNSPECIFIED};
    Config config;
};

enum class TuneStatus {
    SUCCESS = 0,
    NO_FEASIBLE_CANDIDATE,
};

/// Completed index-tuning result. Recommendation fields are valid only on SUCCESS.
struct IndexResult {
    TuneStatus status{TuneStatus::SUCCESS};
    /// Loaded selected index, ready for queries.
    IndexPtr index;
    std::string index_name;
    /// Concrete JSON parameters required to recreate and deserialize the selected index.
    std::string create_parameters;
    /// Concrete JSON parameters recommended for queries.
    std::string search_parameters;
    /// Validated metric values as a JSON object.
    JsonType metrics = JsonType::object();
    /// Selected serialized artifact; the caller removes it when it is no longer needed.
    std::string artifact_path;
    /// Complete report as JSON.
    JsonType report = JsonType::object();
    /// Closest evaluated candidate when status is NO_FEASIBLE_CANDIDATE.
    JsonType best_effort = nullptr;
};

/// Completed search-tuning result. Recommendation fields are valid only on SUCCESS.
struct SearchResult {
    TuneStatus status{TuneStatus::SUCCESS};
    /// Concrete JSON parameters recommended for the existing index.
    std::string parameters;
    /// Validated metric values as a JSON object.
    JsonType metrics = JsonType::object();
    /// Complete report as JSON.
    JsonType report = JsonType::object();
    /// Closest evaluated candidate when status is NO_FEASIBLE_CANDIDATE.
    JsonType best_effort = nullptr;
};

/**
 * Experimental, synchronous build-tree tool API. It is not installed as part of the VSAG SDK.
 *
 * Returns an error for invalid requests or execution failures. A completed request with no
 * feasible candidate returns TuneStatus::NO_FEASIBLE_CANDIDATE and a structured best_effort. On
 * success, only the selected artifact is retained unless Config::keep_intermediate is true.
 */
tl::expected<IndexResult, Error>
TuneIndex(const IndexRequest& request);

/**
 * Synchronously evaluates search candidates against SearchRequest::index without rebuilding it.
 *
 * Returns an error for invalid requests or execution failures. A completed request with no
 * feasible candidate returns TuneStatus::NO_FEASIBLE_CANDIDATE and a structured best_effort.
 */
tl::expected<SearchResult, Error>
TuneSearch(const SearchRequest& request);

/// Offline JSON adapter used by the CLI. Parsed requests persist a full report; validation
/// failures are returned as structured JSON.
JsonType
RunAutoTune(const JsonType& request);

/// Returns the compact, recommendation-first JSON shown by the CLI.
std::string
FormatResultSummaryForCli(const JsonType& report);

}  // namespace vsag::autotune
