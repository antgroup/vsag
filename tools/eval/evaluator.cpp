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

#include "evaluator.h"

#include <omp.h>

#include <limits>
#include <stdexcept>

#include "case/build_eval_case.h"
#include "case/search_eval_case.h"

namespace vsag::eval {

namespace {

class ScopedOpenMpThreads {
public:
    ScopedOpenMpThreads() : original_(omp_get_max_threads()) {
    }

    ~ScopedOpenMpThreads() {
        omp_set_num_threads(original_);
    }

private:
    int original_;
};

void
validate(const IndexPtr& index, const EvalDatasetPtr& dataset) {
    if (index == nullptr) {
        throw std::invalid_argument("evaluation index must not be null");
    }
    if (dataset == nullptr) {
        throw std::invalid_argument("evaluation dataset must not be null");
    }
}

void
validate_search(const EvalDatasetPtr& dataset, const EvalConfig& config) {
    if (config.search_mode != "knn") {
        throw std::invalid_argument("in-memory evaluation supports only knn search mode");
    }
    if (config.top_k <= 0) {
        throw std::invalid_argument("evaluation top_k must be positive");
    }
    if (dataset->GetNumberOfQuery() <= 0) {
        throw std::invalid_argument("evaluation dataset must contain at least one query");
    }
    if (config.num_threads_searching <= 0) {
        throw std::invalid_argument("evaluation search thread count must be positive");
    }
    if (config.search_query_count == 0) {
        throw std::invalid_argument("evaluation search query count must be positive");
    }
    if (config.search_query_count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("evaluation search query count exceeds the supported range");
    }
    if (config.enable_recall or config.enable_percent_recall) {
        const auto requested_top_k = static_cast<uint64_t>(config.top_k);
        if (dataset->GetGroundTruthK() < requested_top_k || dataset->GetNeighbors(0) == nullptr) {
            throw std::invalid_argument(
                "evaluation ground truth must contain at least top_k neighbors per query");
        }
        if (!config.use_id_based_recall && dataset->GetNumberOfBase() <= 0) {
            throw std::invalid_argument("distance-based recall evaluation requires base vectors");
        }
    }
}

}  // namespace

JsonType
EvaluateBuild(const IndexPtr& index, const EvalDatasetPtr& dataset, const EvalConfig& config) {
    validate(index, dataset);
    BuildEvalCase eval_case("", "", index, config, dataset);
    return eval_case.RunInMemory();
}

JsonType
EvaluateSearch(const IndexPtr& index, const EvalDatasetPtr& dataset, const EvalConfig& config) {
    validate(index, dataset);
    validate_search(dataset, config);
    ScopedOpenMpThreads openmp_threads;
    SearchEvalCase eval_case("", "", index, config, dataset);
    return eval_case.RunInMemory();
}

}  // namespace vsag::eval
