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

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "autotune.h"

namespace vsag::eval {
class EvalDataset;
}

namespace vsag::autotune::internal {

namespace eval = vsag::eval;

using MetricMap = std::map<std::string, double>;

struct IndexInput {
    std::string name;
    JsonType create_params = JsonType::object();
    JsonType search_params = JsonType::object();
};

struct RequestContext {
    JsonType effective_request = JsonType::object();
    std::shared_ptr<eval::EvalDataset> dataset;
    std::string workspace_path{"/tmp/vsag_autotune"};
    std::string result_path;
    std::string objective;
    MetricMap constraints;
    uint64_t top_k{0};
    uint64_t concurrency{1};
    uint64_t max_trials{1000};
    uint64_t base_count{0};
    uint64_t query_count{0};
    uint64_t ground_truth_k{0};
    bool enable_recall{false};
    bool keep_intermediate{false};
    bool include_raw_eval{false};
    bool allow_quantization_tune{false};
};

struct IndexTuningRequest {
    RequestContext context;
    std::vector<IndexInput> indexes;
};

struct SearchTuningRequest {
    RequestContext context;
    IndexPtr index;
    IndexInput index_input;
};

using ParsedRequest = std::variant<IndexTuningRequest, SearchTuningRequest>;

struct HGraphEfSearchRange {
    int64_t start;
    int64_t stop;
};

struct Candidate {
    std::string index_name;
    JsonType create_params;
    JsonType search_params;
    std::optional<HGraphEfSearchRange> ef_search_range;
};

struct Evaluation {
    std::vector<JsonType> builds;
    std::vector<JsonType> trials;
};

ParsedRequest
ParseRequest(const JsonType& input);

IndexTuningRequest
ParseRequest(const IndexRequest& input);

SearchTuningRequest
ParseRequest(const SearchRequest& input);

std::vector<Candidate>
GenerateCandidates(const IndexTuningRequest& request);

std::vector<Candidate>
GenerateCandidates(const SearchTuningRequest& request);

void
EvaluateEfSearchRange(const HGraphEfSearchRange& range,
                      double recall_target,
                      const std::function<std::optional<double>(int64_t)>& evaluate);

Evaluation
EvaluateCandidates(const IndexTuningRequest& request,
                   const std::vector<Candidate>& candidates,
                   const std::string& run_path);

Evaluation
EvaluateCandidates(const SearchTuningRequest& request, const std::vector<Candidate>& candidates);

JsonType
SelectResult(const RequestContext& request, const Evaluation& evaluation);

JsonType
Failure(const std::string& stage, const std::string& code, const std::string& message);

void
WriteJson(const std::string& path, const JsonType& value);

}  // namespace vsag::autotune::internal
