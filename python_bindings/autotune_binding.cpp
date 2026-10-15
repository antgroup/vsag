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

#include "autotune_binding.h"

#include <map>
#include <stdexcept>

#include "autotune.h"

namespace vsag::python {

namespace py = pybind11;

namespace {

autotune::Metric
parse_metric(const std::string& name) {
    using Metric = autotune::Metric;
    static const std::map<std::string, Metric> metrics{
        {"recall_at_k", Metric::RECALL_AT_K},
        {"latency_avg_ms", Metric::LATENCY_AVG_MS},
        {"latency_p99_ms", Metric::LATENCY_P99_MS},
        {"qps", Metric::QPS},
        {"index_memory_mb", Metric::INDEX_MEMORY_MB},
        {"index_size_mb", Metric::INDEX_SIZE_MB},
        {"build_seconds", Metric::BUILD_SECONDS},
        {"search_seconds", Metric::SEARCH_SECONDS},
        {"build_and_search_seconds", Metric::BUILD_AND_SEARCH_SECONDS},
    };
    const auto found = metrics.find(name);
    if (found == metrics.end()) {
        throw std::invalid_argument("unsupported AutoTune metric: " + name);
    }
    return found->second;
}

void
validate_matrix(const py::array& array, const py::dtype& dtype, const std::string& name) {
    if (!array.dtype().is(dtype)) {
        throw std::invalid_argument(name + " must have dtype " +
                                    py::str(dtype.attr("name")).cast<std::string>());
    }
    if (array.ndim() != 2 || array.shape(0) == 0 || array.shape(1) == 0) {
        throw std::invalid_argument(name + " must be a non-empty 2-dimensional matrix");
    }
    if ((array.flags() & py::array::c_style) == 0) {
        throw std::invalid_argument(name + " must be C-contiguous");
    }
    if (!array.attr("flags").attr("aligned").cast<bool>()) {
        throw std::invalid_argument(name + " must be aligned");
    }
}

}  // namespace

py::dict
autotune_search(const IndexPtr& index,
                const py::array& queries,
                const py::array& ground_truth,
                uint64_t top_k,
                const py::dict& parameter_space,
                const py::dict& constraints,
                const std::string& objective,
                uint64_t concurrency,
                uint64_t max_trials,
                bool include_raw_evaluation) {
    validate_matrix(queries, py::dtype::of<float>(), "queries");
    validate_matrix(ground_truth, py::dtype::of<int64_t>(), "ground_truth");
    if (queries.shape(0) != ground_truth.shape(0)) {
        throw std::invalid_argument("ground_truth must contain one row for every query");
    }
    if (top_k == 0 || top_k > static_cast<uint64_t>(ground_truth.shape(1))) {
        throw std::invalid_argument(
            "top_k must be positive and no greater than ground_truth width");
    }

    const auto json = py::module_::import("json");
    autotune::SearchRequest request;
    request.index = index;
    request.workload.queries = Dataset::Make()
                                   ->NumElements(queries.shape(0))
                                   ->Dim(queries.shape(1))
                                   ->Float32Vectors(static_cast<const float*>(queries.data()))
                                   ->Owner(false);
    request.workload.ground_truth = Dataset::Make()
                                        ->NumElements(ground_truth.shape(0))
                                        ->Dim(ground_truth.shape(1))
                                        ->Ids(static_cast<const int64_t*>(ground_truth.data()))
                                        ->Owner(false);
    request.workload.top_k = top_k;
    request.workload.concurrency = concurrency;
    request.parameter_space =
        json.attr("dumps")(parameter_space, py::arg("allow_nan") = false).cast<std::string>();
    const auto constraint_json = autotune::JsonType::parse(
        json.attr("dumps")(constraints, py::arg("allow_nan") = false).cast<std::string>());
    for (const auto& item : constraint_json.items()) {
        if (!item.value().is_number()) {
            throw std::invalid_argument("constraint " + item.key() + " must be a number");
        }
        request.constraints.push_back({parse_metric(item.key()), item.value().get<double>()});
    }
    request.objective = parse_metric(objective);
    request.config.max_trials = max_trials;
    request.config.include_raw_evaluation = include_raw_evaluation;

    // Keep the GIL and the argument references alive while the evaluator borrows their buffers.
    // Evaluator workers use only C++ data; concurrency controls their thread count.
    const auto tuned = autotune::TuneSearch(request);
    if (!tuned.has_value()) {
        const auto message = "autotune_search failed: " + tuned.error().message;
        if (tuned.error().type == ErrorType::INVALID_ARGUMENT) {
            throw std::invalid_argument(message);
        }
        throw std::runtime_error(message);
    }
    const auto& result = tuned.value();
    const bool success = result.status == autotune::TuneStatus::SUCCESS;
    py::dict output;
    output["status"] = success ? "success" : "no_feasible_candidate";
    output["search_parameters"] = success ? py::cast(result.parameters) : py::none();
    output["metrics"] = json.attr("loads")(result.metrics.dump());
    output["best_effort"] = json.attr("loads")(result.best_effort.dump());
    output["report"] = json.attr("loads")(result.report.dump());
    return output;
}

}  // namespace vsag::python
