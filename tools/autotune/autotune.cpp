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

#include "autotune.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "autotune_internal.h"
#include "eval_dataset.h"
#include "vsag/constants.h"
#include "vsag/factory.h"

namespace vsag::autotune::internal {

namespace {

double
elapsed(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void
require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

void
known_keys(const JsonType& object,
           std::initializer_list<const char*> keys,
           const std::string& path) {
    require(object.is_object(), path + " must be an object");
    std::set<std::string> known(keys.begin(), keys.end());
    for (const auto& item : object.items()) {
        require(known.find(item.key()) != known.end(), path + "." + item.key() + " is unsupported");
    }
}

std::string
required_string(const JsonType& object, const std::string& key, const std::string& path) {
    require(object.contains(key) && object[key].is_string(), path + "." + key + " is required");
    auto value = object[key].get<std::string>();
    require(!value.empty(), path + "." + key + " must not be empty");
    return value;
}

uint64_t
positive_integer(const JsonType& object, const std::string& key, const std::string& path) {
    require(object.contains(key) && object[key].is_number_integer(),
            path + "." + key + " must be a positive integer");
    if (object[key].is_number_unsigned()) {
        const auto value = object[key].get<uint64_t>();
        require(value > 0, path + "." + key + " must be positive");
        return value;
    }
    const auto value = object[key].get<int64_t>();
    require(value > 0, path + "." + key + " must be positive");
    return static_cast<uint64_t>(value);
}

std::string
normalize(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

std::string
metric_name(Metric metric) {
    switch (metric) {
        case Metric::RECALL_AT_K:
            return "recall_at_k";
        case Metric::LATENCY_AVG_MS:
            return "latency_avg_ms";
        case Metric::LATENCY_P99_MS:
            return "latency_p99_ms";
        case Metric::QPS:
            return "qps";
        case Metric::INDEX_MEMORY_MB:
            return "index_memory_mb";
        case Metric::INDEX_SIZE_MB:
            return "index_size_mb";
        case Metric::BUILD_SECONDS:
            return "build_seconds";
        case Metric::SEARCH_SECONDS:
            return "search_seconds";
        case Metric::BUILD_AND_SEARCH_SECONDS:
            return "build_and_search_seconds";
        case Metric::UNSPECIFIED:
            break;
    }
    throw std::invalid_argument("AutoTune metric must be specified");
}

Metric
parse_metric(const std::string& metric) {
    if (metric == "recall_at_k") {
        return Metric::RECALL_AT_K;
    }
    if (metric == "latency_avg_ms") {
        return Metric::LATENCY_AVG_MS;
    }
    if (metric == "latency_p99_ms") {
        return Metric::LATENCY_P99_MS;
    }
    if (metric == "qps") {
        return Metric::QPS;
    }
    if (metric == "index_memory_mb") {
        return Metric::INDEX_MEMORY_MB;
    }
    if (metric == "index_size_mb") {
        return Metric::INDEX_SIZE_MB;
    }
    if (metric == "build_seconds") {
        return Metric::BUILD_SECONDS;
    }
    if (metric == "search_seconds") {
        return Metric::SEARCH_SECONDS;
    }
    if (metric == "build_and_search_seconds") {
        return Metric::BUILD_AND_SEARCH_SECONDS;
    }
    throw std::invalid_argument("unsupported metric: " + metric);
}

bool
supported_metric(const std::string& metric) {
    static const std::set<std::string> metrics{"build_seconds",
                                               "index_size_mb",
                                               "index_memory_mb",
                                               "recall_at_k",
                                               "latency_avg_ms",
                                               "latency_p99_ms",
                                               "qps",
                                               "search_seconds",
                                               "build_and_search_seconds"};
    return metrics.find(metric) != metrics.end();
}

bool
higher_is_better(const std::string& metric) {
    return metric == "recall_at_k" || metric == "qps";
}

bool
available_for_existing_index(const std::string& metric) {
    return metric != "build_seconds" && metric != "build_and_search_seconds" &&
           metric != "index_size_mb";
}

bool
useful_existing_index_objective(const std::string& metric) {
    return available_for_existing_index(metric) && metric != "index_memory_mb";
}

bool
same_path(const std::string& left, const std::string& right) {
    if (left.empty() || right.empty()) {
        return false;
    }

    std::error_code equivalent_error;
    if (std::filesystem::equivalent(left, right, equivalent_error)) {
        return true;
    }

    const auto normalized = [](const std::string& value) {
        std::error_code error;
        const auto canonical = std::filesystem::weakly_canonical(value, error);
        return error ? std::filesystem::absolute(value).lexically_normal() : canonical;
    };
    return normalized(left) == normalized(right);
}

std::string
metric_type(const std::string& dataset_metric) {
    if (dataset_metric == "euclidean") {
        return "l2";
    }
    if (dataset_metric == "ip") {
        return "ip";
    }
    if (dataset_metric == "angular") {
        return "cosine";
    }
    throw std::invalid_argument("unsupported dataset distance: " + dataset_metric);
}

struct offline_dataset_owner {
    eval::EvalDatasetPtr source;
    std::vector<int64_t> identity_ids;
    DatasetPtr base;
    DatasetPtr queries;
    DatasetPtr ground_truth;
};

void
attach_offline_dataset(IndexRequest& request,
                       const eval::EvalDatasetPtr& dataset,
                       bool include_base) {
    require(dataset->GetVectorType() == "dense_vectors",
            "AutoTune V1 supports only dense vector datasets");
    require(dataset->GetTrainDataType() == vsag::DATATYPE_FLOAT32 &&
                dataset->GetTestDataType() == vsag::DATATYPE_FLOAT32,
            "AutoTune V1 supports only float32 datasets");

    auto owner = std::make_shared<offline_dataset_owner>();
    owner->source = dataset;
    const auto base_count = dataset->GetNumberOfBase();
    const auto query_count = dataset->GetNumberOfQuery();
    const auto ground_truth_k = static_cast<int64_t>(dataset->GetGroundTruthK());
    if (include_base) {
        const auto* train_ids = dataset->GetTrainIds();
        if (train_ids == nullptr) {
            owner->identity_ids.resize(static_cast<uint64_t>(base_count));
            for (int64_t i = 0; i < base_count; ++i) {
                owner->identity_ids[static_cast<uint64_t>(i)] = i;
            }
            train_ids = owner->identity_ids.data();
        }
        owner->base = Dataset::Make()
                          ->NumElements(base_count)
                          ->Dim(dataset->GetDim())
                          ->Ids(train_ids)
                          ->Float32Vectors(static_cast<const float*>(dataset->GetTrain()))
                          ->Owner(false);
    }
    owner->queries = Dataset::Make()
                         ->NumElements(query_count)
                         ->Dim(dataset->GetDim())
                         ->Float32Vectors(static_cast<const float*>(dataset->GetTest()))
                         ->Owner(false);
    if (ground_truth_k > 0) {
        owner->ground_truth = Dataset::Make()
                                  ->NumElements(query_count)
                                  ->Dim(ground_truth_k)
                                  ->Ids(dataset->GetNeighbors(0))
                                  ->Distances(dataset->GetDistances(0))
                                  ->Owner(false);
    }

    if (owner->base != nullptr) {
        request.base = DatasetPtr(owner, owner->base.get());
    }
    request.workload.queries = DatasetPtr(owner, owner->queries.get());
    if (owner->ground_truth != nullptr) {
        request.workload.ground_truth = DatasetPtr(owner, owner->ground_truth.get());
    }
    request.metric_type = metric_type(dataset->GetMetric());
}

void
check_file(const std::string& path, const std::string& field) {
    std::error_code error;
    require(std::filesystem::is_regular_file(path, error) && !error,
            field + " must name a readable regular file: " + path);
    std::ifstream input(path, std::ios::binary);
    require(input.good(), field + " must name a readable regular file: " + path);
}

void
merge_dataset_field(JsonType& create_params,
                    const std::string& name,
                    const JsonType& value,
                    const std::string& index_name) {
    if (create_params.contains(name)) {
        require(create_params[name] == value,
                index_name + " create_params." + name + " must match the dataset");
    }
    create_params[name] = value;
}

MetricMap
read_metrics(const JsonType& value) {
    MetricMap metrics;
    if (!value.is_object()) {
        return metrics;
    }
    for (const auto& item : value.items()) {
        if (item.value().is_number()) {
            metrics[item.key()] = item.value().get<double>();
        }
    }
    return metrics;
}

JsonType
constraint_evaluation(const RequestContext& request, const JsonType& trial) {
    JsonType violations = JsonType::array();
    const auto metrics = read_metrics(trial.value("metrics", JsonType::object()));
    for (const auto& [name, expected] : request.constraints) {
        const auto actual = metrics.find(name);
        const bool present = actual != metrics.end() && std::isfinite(actual->second);
        const bool satisfied = present && (higher_is_better(name) ? actual->second >= expected
                                                                  : actual->second <= expected);
        if (!satisfied) {
            violations.push_back(
                {{"metric", name},
                 {"comparison", higher_is_better(name) ? "at_least" : "at_most"},
                 {"expected", expected},
                 {"actual", present ? JsonType(actual->second) : JsonType(nullptr)}});
        }
    }
    return {{"satisfied", violations.empty()}, {"violations", std::move(violations)}};
}

double
objective_value(const RequestContext& request, const JsonType& trial) {
    const auto metrics = read_metrics(trial.value("metrics", JsonType::object()));
    const auto value = metrics.find(request.objective);
    if (value == metrics.end() || !std::isfinite(value->second)) {
        return higher_is_better(request.objective) ? -std::numeric_limits<double>::infinity()
                                                   : std::numeric_limits<double>::infinity();
    }
    return value->second;
}

bool
has_objective(const RequestContext& request, const JsonType& trial) {
    const auto metrics = read_metrics(trial.value("metrics", JsonType::object()));
    const auto value = metrics.find(request.objective);
    return value != metrics.end() && std::isfinite(value->second);
}

JsonType
recommendation(const RequestContext& request, const JsonType& trial) {
    JsonType result{{"index_name", trial["index_name"]},
                    {"search_params", trial["search_params"]},
                    {"workload", {{"top_k", request.top_k}, {"concurrency", request.concurrency}}},
                    {"metrics", trial["metrics"]},
                    {"evidence", {{"trial_id", trial["trial_id"]}}}};
    if (trial.contains("create_params")) {
        result["create_params"] = trial["create_params"];
    }
    if (trial.contains("artifacts")) {
        result["artifacts"] = trial["artifacts"];
    }
    if (trial.contains("build_id")) {
        result["evidence"]["build_id"] = trial["build_id"];
    }
    return result;
}

std::mutex&
run_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::string
new_run_name() {
    static std::atomic<uint64_t> serial{0};
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return "run-" + std::to_string(now) + "-" + std::to_string(serial++);
}

}  // namespace

namespace {

JsonType
parse_parameters(const std::string& value, const std::string& path) {
    try {
        auto parsed = JsonType::parse(value);
        require(parsed.is_object(), path + " must encode a JSON object");
        return parsed;
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument(path + " is invalid JSON: " + error.what());
    }
}

RequestContext
make_context(eval::EvalDatasetPtr dataset,
             uint64_t base_count,
             const std::string& metric_type,
             const Workload& workload,
             const std::vector<Constraint>& constraints,
             Metric objective,
             const Config& config,
             bool search_only) {
    RequestContext request;
    request.workspace_path =
        config.workspace_path.empty() ? "/tmp/vsag_autotune" : config.workspace_path;
    request.keep_intermediate = config.keep_intermediate;
    request.include_raw_eval = config.include_raw_evaluation;
    request.max_trials = config.max_trials;
    request.top_k = workload.top_k;
    request.concurrency = workload.concurrency;

    request.dataset = std::move(dataset);
    request.base_count = base_count;
    request.query_count = static_cast<uint64_t>(request.dataset->GetNumberOfQuery());
    request.ground_truth_k = request.dataset->GetGroundTruthK();
    require(request.top_k > 0, "request.workload.top_k must be positive");
    require(request.top_k <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
            "request.workload.top_k is too large");
    require(request.concurrency > 0 && request.concurrency <= 200,
            "request.workload.concurrency must be in [1, 200]");
    require(request.max_trials > 0 && request.max_trials <= 100000,
            "request.config.max_trials must be in [1, 100000]");

    request.objective = metric_name(objective);
    request.enable_recall = objective == Metric::RECALL_AT_K;
    require(!constraints.empty(), "request.constraints must not be empty");
    for (const auto& constraint : constraints) {
        const auto name = metric_name(constraint.metric);
        require(std::isfinite(constraint.value) && constraint.value >= 0.0,
                "request.constraints." + name + " must be finite and non-negative");
        require(name != "recall_at_k" || constraint.value <= 1.0,
                "request.constraints.recall_at_k must be in [0, 1]");
        require(!search_only || available_for_existing_index(name),
                "request.constraints." + name + " is unavailable for search tuning");
        require(request.constraints.emplace(name, constraint.value).second,
                "request.constraints contains duplicate metric: " + name);
        request.enable_recall = request.enable_recall || constraint.metric == Metric::RECALL_AT_K;
    }
    require(!search_only || useful_existing_index_objective(request.objective),
            "request.objective cannot rank search candidates: " + request.objective);

    require(request.top_k <= request.base_count, "request.workload.top_k exceeds the index size");
    if (request.enable_recall) {
        require(workload.ground_truth != nullptr,
                "request.workload.ground_truth is required for recall_at_k");
        require(request.ground_truth_k >= request.top_k,
                "request.workload.top_k exceeds ground truth k");
    }

    JsonType dataset_info{{"base_count", request.base_count},
                          {"query_count", request.query_count},
                          {"ground_truth_k", request.ground_truth_k},
                          {"dim", request.dataset->GetDim()},
                          {"dtype", vsag::DATATYPE_FLOAT32}};
    if (!metric_type.empty()) {
        dataset_info["metric_type"] = metric_type;
    }
    request.effective_request = {
        {"version", 1},
        {"dataset", std::move(dataset_info)},
        {"workload", {{"top_k", request.top_k}, {"concurrency", request.concurrency}}},
        {"constraints", JsonType::object()},
        {"objective", {{"metric", request.objective}}},
        {"config",
         {{"workspace_path", request.workspace_path},
          {"keep_intermediate", request.keep_intermediate},
          {"max_trials", request.max_trials}}},
        {"output", {{"include_raw_evaluation", request.include_raw_eval}}}};
    for (const auto& [name, value] : request.constraints) {
        request.effective_request["constraints"][name] = value;
    }
    return request;
}

std::string
index_name(const IndexPtr& index) {
    switch (index->GetIndexType()) {
        case IndexType::HGRAPH:
            return INDEX_HGRAPH;
        case IndexType::IVF:
            return INDEX_IVF;
        case IndexType::PYRAMID:
            return INDEX_PYRAMID;
        default:
            throw std::invalid_argument("request.index type is unsupported by AutoTune");
    }
}

IndexInput
parse_index_space(const IndexSpace& value, uint64_t position, bool search_only) {
    const auto path = "request.index_spaces[" + std::to_string(position) + "]";
    IndexInput index;
    index.name = normalize(value.name);
    require(
        index.name == "hgraph" || index.name == "ivf" || (search_only && index.name == "pyramid"),
        "unsupported index: " + index.name);
    index.create_params = search_only ? JsonType::object()
                                      : parse_parameters(value.create_parameter_space,
                                                         path + ".create_parameter_space");
    const auto search =
        parse_parameters(value.search_parameter_space, path + ".search_parameter_space");
    for (const auto& item : search.items()) {
        require(normalize(item.key()) == index.name,
                path + ".search_parameter_space." + item.key() + " is unsupported");
        require(!index.search_params.contains(index.name),
                path + ".search_parameter_space contains duplicate index namespace");
        index.search_params[index.name] = item.value();
    }
    return index;
}

}  // namespace

IndexTuningRequest
ParseRequest(const IndexRequest& input) {
    require(input.base != nullptr, "request.base is required");
    require(input.base->GetIds() != nullptr, "request.base IDs are required");
    const auto metric = normalize(input.metric_type);
    require(metric == "l2" || metric == "ip" || metric == "cosine",
            "request.metric_type must be l2, ip, or cosine");
    auto dataset = eval::EvalDataset::FromDatasets(
        input.base, input.workload.queries, input.workload.ground_truth, metric);
    IndexTuningRequest request;
    request.context = make_context(std::move(dataset),
                                   static_cast<uint64_t>(input.base->GetNumElements()),
                                   metric,
                                   input.workload,
                                   input.constraints,
                                   input.objective,
                                   input.config,
                                   false);
    if (input.index_spaces.empty()) {
        request.indexes = {{"hgraph"}, {"ivf"}};
    } else {
        for (uint64_t i = 0; i < input.index_spaces.size(); ++i) {
            request.indexes.emplace_back(parse_index_space(input.index_spaces[i], i, false));
        }
    }
    const auto dim = static_cast<uint64_t>(request.context.dataset->GetDim());
    for (auto& index : request.indexes) {
        merge_dataset_field(index.create_params, "dim", dim, index.name);
        merge_dataset_field(index.create_params, "dtype", vsag::DATATYPE_FLOAT32, index.name);
        merge_dataset_field(index.create_params, "metric_type", metric, index.name);
        request.context.effective_request["index_spaces"].push_back(
            {{"name", index.name},
             {"create_parameter_space", index.create_params},
             {"search_parameter_space", index.search_params}});
    }
    return request;
}

SearchTuningRequest
ParseRequest(const SearchRequest& input) {
    require(input.index != nullptr, "request.index is required");
    const auto element_count = input.index->GetNumElements();
    require(element_count > 0, "request.index must not be empty");
    auto dataset =
        eval::EvalDataset::FromSearchDatasets(input.workload.queries, input.workload.ground_truth);
    SearchTuningRequest request;
    request.context = make_context(std::move(dataset),
                                   static_cast<uint64_t>(element_count),
                                   "",
                                   input.workload,
                                   input.constraints,
                                   input.objective,
                                   input.config,
                                   true);
    IndexSpace space;
    space.name = index_name(input.index);
    space.search_parameter_space = input.parameter_space;
    request.index_input = parse_index_space(space, 0, true);
    request.index = input.index;
    request.context.effective_request["index_name"] = request.index_input.name;
    request.context.effective_request["parameter_space"] = request.index_input.search_params;
    return request;
}

ParsedRequest
ParseRequest(const JsonType& input) {
    known_keys(input,
               {"version",
                "data_path",
                "index_path",
                "indexes",
                "workload",
                "constraints",
                "objective",
                "tuning_config",
                "output"},
               "request");
    require(input.contains("version") && input["version"] == 1, "request.version must be 1");
    const auto data_path = required_string(input, "data_path", "request");
    check_file(data_path, "data_path");

    IndexRequest typed;
    std::string report_path;
    attach_offline_dataset(
        typed, eval::EvalDataset::Load(data_path), !input.contains("index_path"));
    if (input.contains("indexes")) {
        require(input["indexes"].is_array() && !input["indexes"].empty(),
                "request.indexes must be a non-empty array");
        for (uint64_t i = 0; i < input["indexes"].size(); ++i) {
            const auto& value = input["indexes"][i];
            const auto path = "request.indexes[" + std::to_string(i) + "]";
            known_keys(value, {"name", "create_params", "search_params"}, path);
            IndexSpace index;
            index.name = required_string(value, "name", path);
            if (value.contains("create_params")) {
                require(value["create_params"].is_object(),
                        path + ".create_params must be an object");
                index.create_parameter_space = value["create_params"].dump();
            }
            if (value.contains("search_params")) {
                require(value["search_params"].is_object(),
                        path + ".search_params must be an object");
                index.search_parameter_space = value["search_params"].dump();
            }
            typed.index_spaces.emplace_back(std::move(index));
        }
    }
    require(input.contains("workload"), "request.workload is required");
    known_keys(input["workload"], {"top_k", "concurrency"}, "request.workload");
    typed.workload.top_k = positive_integer(input["workload"], "top_k", "request.workload");
    if (input["workload"].contains("concurrency")) {
        typed.workload.concurrency =
            positive_integer(input["workload"], "concurrency", "request.workload");
    }
    require(input.contains("constraints") && input["constraints"].is_object() &&
                !input["constraints"].empty(),
            "request.constraints must be a non-empty object");
    for (const auto& item : input["constraints"].items()) {
        require(supported_metric(item.key()), "unsupported metric: " + item.key());
        require(item.value().is_number(),
                "request.constraints." + item.key() + " must be a number");
        typed.constraints.push_back({parse_metric(item.key()), item.value().get<double>()});
    }
    require(input.contains("objective"), "request.objective is required");
    known_keys(input["objective"], {"metric"}, "request.objective");
    typed.objective =
        parse_metric(required_string(input["objective"], "metric", "request.objective"));
    if (input.contains("tuning_config")) {
        const auto& config = input["tuning_config"];
        known_keys(
            config, {"workspace_path", "keep_intermediate", "max_trials"}, "request.tuning_config");
        if (config.contains("workspace_path")) {
            typed.config.workspace_path =
                required_string(config, "workspace_path", "request.tuning_config");
        }
        if (config.contains("keep_intermediate")) {
            require(config["keep_intermediate"].is_boolean(),
                    "request.tuning_config.keep_intermediate must be a boolean");
            typed.config.keep_intermediate = config["keep_intermediate"].get<bool>();
        }
        if (config.contains("max_trials")) {
            typed.config.max_trials =
                positive_integer(config, "max_trials", "request.tuning_config");
        }
    }
    if (input.contains("output")) {
        const auto& output = input["output"];
        known_keys(output, {"result_path", "include_raw_eval"}, "request.output");
        if (output.contains("result_path")) {
            report_path = required_string(output, "result_path", "request.output");
            require(!same_path(report_path, data_path),
                    "request.output.result_path must not alias data_path");
        }
        if (output.contains("include_raw_eval")) {
            require(output["include_raw_eval"].is_boolean(),
                    "request.output.include_raw_eval must be a boolean");
            typed.config.include_raw_evaluation = output["include_raw_eval"].get<bool>();
        }
    }

    if (!input.contains("index_path")) {
        auto parsed = ParseRequest(typed);
        parsed.context.result_path = report_path;
        parsed.context.effective_request["data_path"] = data_path;
        if (!report_path.empty()) {
            parsed.context.effective_request["output"]["report_path"] = report_path;
        }
        return parsed;
    }

    const auto index_path = required_string(input, "index_path", "request");
    check_file(index_path, "index_path");
    require(!same_path(report_path, index_path),
            "request.output.result_path must not alias index_path");
    require(typed.index_spaces.size() == 1,
            "index_path requires exactly one indexes[] specification");
    auto& space = typed.index_spaces.front();
    auto create_params =
        parse_parameters(space.create_parameter_space, "request.indexes[0].create_params");
    require(create_params.contains("index_param") && create_params["index_param"].is_object(),
            "request.indexes[0].create_params.index_param is required");
    merge_dataset_field(
        create_params, "dim", static_cast<uint64_t>(typed.workload.queries->GetDim()), space.name);
    merge_dataset_field(create_params, "dtype", vsag::DATATYPE_FLOAT32, space.name);
    merge_dataset_field(create_params, "metric_type", normalize(typed.metric_type), space.name);
    auto created = Factory::CreateIndex(normalize(space.name), create_params.dump());
    if (!created.has_value()) {
        throw std::invalid_argument(created.error().message);
    }
    std::ifstream serialized(index_path, std::ios::binary);
    auto loaded = created.value()->Deserialize(serialized);
    if (!loaded.has_value()) {
        throw std::invalid_argument(loaded.error().message);
    }

    SearchRequest search;
    search.index = created.value();
    search.workload = typed.workload;
    search.parameter_space = space.search_parameter_space;
    search.constraints = typed.constraints;
    search.objective = typed.objective;
    search.config = typed.config;
    auto parsed = ParseRequest(search);
    parsed.context.result_path = report_path;
    parsed.context.effective_request["data_path"] = data_path;
    parsed.context.effective_request["index_path"] = index_path;
    parsed.context.effective_request["create_params"] = create_params;
    if (!report_path.empty()) {
        parsed.context.effective_request["output"]["report_path"] = report_path;
    }
    return parsed;
}

JsonType
SelectResult(const RequestContext& request, const Evaluation& evaluation) {
    JsonType trials = JsonType::array();
    int64_t best = -1;
    int64_t best_effort = -1;
    int64_t best_pruned = -1;
    uint64_t best_violations = std::numeric_limits<uint64_t>::max();
    uint64_t best_pruned_violations = std::numeric_limits<uint64_t>::max();
    double best_violation_score = std::numeric_limits<double>::infinity();
    double best_pruned_violation_score = std::numeric_limits<double>::infinity();
    bool has_successful_trial = false;
    bool has_successful_objective = false;

    const auto violation_rank = [](const JsonType& trial) {
        const auto& violations = trial["constraint_evaluation"]["violations"];
        double score = 0.0;
        for (const auto& violation : violations) {
            if (violation["actual"].is_null()) {
                score += 1.0;
            } else {
                score += std::abs(violation["actual"].get<double>() -
                                  violation["expected"].get<double>()) /
                         std::max(violation["expected"].get<double>(), 1e-12);
            }
        }
        return std::make_pair(static_cast<uint64_t>(violations.size()), score);
    };

    for (const auto& source : evaluation.trials) {
        auto trial = source;
        trial["constraint_evaluation"] = constraint_evaluation(request, trial);
        trials.push_back(std::move(trial));
        const auto stored_index = static_cast<int64_t>(trials.size() - 1);
        const auto& stored = trials.back();
        if (stored["status"] == "pruned") {
            const auto [violation_count, score] = violation_rank(stored);
            if (violation_count > 0 &&
                (best_pruned < 0 || violation_count < best_pruned_violations ||
                 (violation_count == best_pruned_violations &&
                  score < best_pruned_violation_score))) {
                best_pruned = stored_index;
                best_pruned_violations = violation_count;
                best_pruned_violation_score = score;
            }
            continue;
        }
        if (stored["status"] != "success") {
            continue;
        }
        has_successful_trial = true;
        if (!has_objective(request, stored)) {
            continue;
        }
        has_successful_objective = true;

        const auto [violation_count, score] = violation_rank(stored);
        if (violation_count == 0) {
            if (best < 0 ||
                (higher_is_better(request.objective)
                     ? objective_value(request, stored) > objective_value(request, trials[best])
                     : objective_value(request, stored) < objective_value(request, trials[best]))) {
                best = stored_index;
            }
            continue;
        }

        if (best_effort < 0 || violation_count < best_violations ||
            (violation_count == best_violations && score < best_violation_score)) {
            best_effort = stored_index;
            best_violations = violation_count;
            best_violation_score = score;
        }
    }

    JsonType result{{"status", best >= 0 ? "success" : "no_feasible_candidate"},
                    {"recommendation", nullptr},
                    {"best_effort", nullptr},
                    {"builds", evaluation.builds},
                    {"trials", std::move(trials)}};
    if (best >= 0) {
        result["recommendation"] = recommendation(request, result["trials"][best]);
    } else if (best_effort >= 0) {
        result["best_effort"] = recommendation(request, result["trials"][best_effort]);
        result["best_effort"]["constraint_evaluation"] =
            result["trials"][best_effort]["constraint_evaluation"];
    } else if (best_pruned >= 0 && !has_successful_trial) {
        result["best_effort"] = recommendation(request, result["trials"][best_pruned]);
        result["best_effort"]["constraint_evaluation"] =
            result["trials"][best_pruned]["constraint_evaluation"];
    } else {
        result["status"] = "failed";
        result["failure"] =
            has_successful_trial && !has_successful_objective
                ? Failure("selection",
                          "objective_metric_unavailable",
                          "objective metric is unavailable: " + request.objective)
                : Failure("evaluation", "all_trials_failed", "all candidate evaluations failed");
    }
    return result;
}

JsonType
Failure(const std::string& stage, const std::string& code, const std::string& message) {
    return {{"stage", stage}, {"code", code}, {"message", message}};
}

void
WriteJson(const std::string& path, const JsonType& value) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream output(path);
    if (!output.good()) {
        throw std::runtime_error("failed to open report path: " + path);
    }
    output << value.dump(2) << std::endl;
    if (!output.good()) {
        throw std::runtime_error("failed to write report: " + path);
    }
}

}  // namespace vsag::autotune::internal

namespace vsag::autotune {

namespace {

void
prepare_report_path(const std::string& path) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream output(path, std::ios::app);
    if (!output.good()) {
        throw std::runtime_error("failed to open report path: " + path);
    }
}

void
finalize_artifacts(JsonType& report, bool keep_all, const std::optional<std::string>& selected);

std::optional<std::string>
selected_artifact(const JsonType& report);

template <typename Parser>
JsonType
run_tuning_locked(Parser parser, bool persist_report, std::chrono::steady_clock::time_point start) {
    std::string stage = "validation";
    std::string run_path;
    std::string report_path;
    bool keep_intermediate = false;
    const auto failure = [&](const std::string& message) {
        if (!keep_intermediate && !run_path.empty()) {
            std::error_code cleanup_error;
            std::filesystem::remove_all(run_path, cleanup_error);
        }
        const char* const code = stage == "validation" || stage == "candidate_generation"
                                     ? "invalid_request"
                                     : "execution_failed";
        JsonType report{{"version", 1},
                        {"status", "failed"},
                        {"recommendation", nullptr},
                        {"best_effort", nullptr},
                        {"elapsed_seconds", internal::elapsed(start)},
                        {"failure", internal::Failure(stage, code, message)}};
        if (!report_path.empty()) {
            report["report_path"] = report_path;
            try {
                internal::WriteJson(report_path, report);
            } catch (...) {
            }
        }
        return report;
    };
    try {
        auto request = parser();
        auto& context = request.context;
        keep_intermediate = context.keep_intermediate;
        const auto run_name = internal::new_run_name();
        if (persist_report) {
            report_path = context.result_path.empty()
                              ? context.workspace_path + "/" + run_name + ".json"
                              : context.result_path;
            stage = "report";
            prepare_report_path(report_path);
        }
        stage = "candidate_generation";
        const auto candidates = internal::GenerateCandidates(request);
        stage = "evaluation";
        internal::Evaluation evaluation;
        if constexpr (std::is_same_v<decltype(request), internal::IndexTuningRequest>) {
            run_path = context.workspace_path + "/runs/" + run_name;
            std::filesystem::create_directories(run_path + "/artifacts");
            evaluation = internal::EvaluateCandidates(request, candidates, run_path);
        } else {
            evaluation = internal::EvaluateCandidates(request, candidates);
        }
        auto report = internal::SelectResult(context, evaluation);
        report["version"] = 1;
        report["request"] = context.effective_request;
        if (!report_path.empty()) {
            report["report_path"] = report_path;
            report["request"]["output"]["report_path"] = report_path;
        }
        if constexpr (std::is_same_v<decltype(request), internal::IndexTuningRequest>) {
            const auto selected = selected_artifact(report);
            finalize_artifacts(report, context.keep_intermediate, selected);
            if (!context.keep_intermediate && !selected.has_value() && !run_path.empty()) {
                std::error_code error;
                std::filesystem::remove_all(run_path, error);
            }
        }
        report["elapsed_seconds"] = internal::elapsed(start);
        stage = "report";
        if (!report_path.empty()) {
            internal::WriteJson(report_path, report);
        }
        return report;
    } catch (const std::exception& error) {
        return failure(error.what());
    } catch (...) {
        return failure("unknown AutoTune error");
    }
}

template <typename Parser>
JsonType
run_tuning(Parser parser,
           std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now()) {
    std::lock_guard<std::mutex> lock(internal::run_mutex());
    return run_tuning_locked(std::move(parser), false, start);
}

Error
report_error(const JsonType& report) {
    auto type = ErrorType::INTERNAL_ERROR;
    auto message = std::string("AutoTune failed");
    if (report.contains("failure") && report["failure"].is_object()) {
        const auto& failure = report["failure"];
        if (failure.value("stage", std::string()) == "validation" ||
            failure.value("stage", std::string()) == "candidate_generation") {
            type = ErrorType::INVALID_ARGUMENT;
        }
        message = failure.value("message", message);
    }
    return {type, std::move(message)};
}

void
update_artifact(JsonType& value,
                bool keep_all,
                const std::optional<std::string>& selected,
                std::unordered_set<std::string>& removed) {
    if (!value.is_object() || !value.contains("artifacts") || !value["artifacts"].is_object() ||
        !value["artifacts"].contains("index_path") ||
        !value["artifacts"]["index_path"].is_string()) {
        return;
    }

    auto& artifacts = value["artifacts"];
    const auto path = artifacts["index_path"].get<std::string>();
    std::error_code exists_error;
    const auto exists = std::filesystem::is_regular_file(path, exists_error) && !exists_error;
    const auto should_retain = keep_all || (selected.has_value() && path == *selected);
    if (exists && !should_retain && removed.emplace(path).second) {
        std::error_code remove_error;
        if (std::filesystem::remove(path, remove_error)) {
            std::filesystem::remove(std::filesystem::path(path).parent_path(), remove_error);
            std::filesystem::remove(std::filesystem::path(path).parent_path().parent_path(),
                                    remove_error);
        }
    }
    std::error_code retained_error;
    artifacts["retained"] =
        std::filesystem::is_regular_file(path, retained_error) && !retained_error;
}

void
finalize_artifacts(JsonType& report, bool keep_all, const std::optional<std::string>& selected) {
    std::unordered_set<std::string> removed;
    if (report.contains("builds") && report["builds"].is_array()) {
        for (auto& build : report["builds"]) {
            update_artifact(build, keep_all, selected, removed);
        }
    }
    if (report.contains("trials") && report["trials"].is_array()) {
        for (auto& trial : report["trials"]) {
            update_artifact(trial, keep_all, selected, removed);
        }
    }
    for (const auto* key : {"recommendation", "best_effort"}) {
        if (report.contains(key)) {
            update_artifact(report[key], keep_all, selected, removed);
        }
    }
}

std::optional<std::string>
selected_artifact(const JsonType& report) {
    if (report.value("status", std::string()) != "success" || !report.contains("recommendation") ||
        !report["recommendation"].is_object()) {
        return std::nullopt;
    }
    const auto& recommendation = report["recommendation"];
    if (!recommendation.contains("artifacts") || !recommendation["artifacts"].is_object() ||
        !recommendation["artifacts"].contains("index_path") ||
        !recommendation["artifacts"]["index_path"].is_string()) {
        return std::nullopt;
    }
    return recommendation["artifacts"]["index_path"].get<std::string>();
}

void
finalize_artifacts_noexcept(JsonType& report, bool keep_all) noexcept {
    try {
        finalize_artifacts(report, keep_all, std::nullopt);
    } catch (...) {
    }
}

JsonType
validation_failure(const std::chrono::steady_clock::time_point& start, const std::string& message) {
    return {{"version", 1},
            {"status", "failed"},
            {"recommendation", nullptr},
            {"best_effort", nullptr},
            {"elapsed_seconds", internal::elapsed(start)},
            {"failure", internal::Failure("validation", "invalid_request", message)}};
}

}  // namespace

tl::expected<IndexResult, Error>
TuneIndex(const IndexRequest& request) {
    const auto start = std::chrono::steady_clock::now();
    auto report = run_tuning([&request]() { return internal::ParseRequest(request); }, start);
    const auto status = report.value("status", std::string("failed"));
    if (status == "failed") {
        return tl::unexpected(report_error(report));
    }
    if (status == "no_feasible_candidate") {
        IndexResult result;
        result.status = TuneStatus::NO_FEASIBLE_CANDIDATE;
        result.report = report;
        result.best_effort = report.value("best_effort", JsonType(nullptr));
        return result;
    }

    const auto artifact_failure = [&](Error error) -> tl::expected<IndexResult, Error> {
        finalize_artifacts_noexcept(report, request.config.keep_intermediate);
        return tl::unexpected(std::move(error));
    };

    try {
        const auto& recommendation = report.at("recommendation");
        const auto index_name = recommendation.at("index_name").get<std::string>();
        const auto create_parameters = recommendation.at("create_params").dump();
        const auto search_parameters = recommendation.at("search_params").dump();
        const auto artifact_path =
            recommendation.at("artifacts").at("index_path").get<std::string>();
        auto created = Factory::CreateIndex(index_name, create_parameters);
        if (!created.has_value()) {
            return artifact_failure(created.error());
        }
        std::ifstream input(artifact_path, std::ios::binary);
        if (!input.good()) {
            return artifact_failure(
                Error(ErrorType::MISSING_FILE, "failed to open index artifact: " + artifact_path));
        }
        auto loaded = created.value()->Deserialize(input);
        if (!loaded.has_value()) {
            return artifact_failure(loaded.error());
        }

        IndexResult result;
        result.index = created.value();
        result.index_name = index_name;
        result.create_parameters = create_parameters;
        result.search_parameters = search_parameters;
        result.metrics = recommendation.at("metrics");
        result.artifact_path = artifact_path;
        result.report = report;
        return result;
    } catch (const std::exception& error) {
        return artifact_failure(Error(ErrorType::INTERNAL_ERROR, error.what()));
    }
}

tl::expected<SearchResult, Error>
TuneSearch(const SearchRequest& request) {
    auto report = run_tuning([&request]() { return internal::ParseRequest(request); });
    const auto status = report.value("status", std::string("failed"));
    if (status == "failed") {
        return tl::unexpected(report_error(report));
    }
    if (status == "no_feasible_candidate") {
        SearchResult result;
        result.status = TuneStatus::NO_FEASIBLE_CANDIDATE;
        result.report = report;
        result.best_effort = report.value("best_effort", JsonType(nullptr));
        return result;
    }

    const auto& recommendation = report["recommendation"];
    SearchResult result;
    result.parameters = recommendation["search_params"].dump();
    result.metrics = recommendation["metrics"];
    result.report = report;
    return result;
}

JsonType
RunAutoTune(const JsonType& request) {
    const auto start = std::chrono::steady_clock::now();
    try {
        std::lock_guard<std::mutex> lock(internal::run_mutex());
        return std::visit(
            [start](auto parsed) {
                return run_tuning_locked(
                    [parsed = std::move(parsed)]() { return parsed; }, true, start);
            },
            internal::ParseRequest(request));
    } catch (const H5::Exception& error) {
        return validation_failure(start,
                                  "failed to load evaluation dataset: " + error.getDetailMsg());
    } catch (const std::exception& error) {
        return validation_failure(start, error.what());
    } catch (...) {
        return validation_failure(start, "unknown request validation error");
    }
}

std::string
FormatResultSummaryForCli(const JsonType& report) {
    nlohmann::ordered_json result;
    for (const auto* key : {"recommendation",
                            "best_effort",
                            "failure",
                            "status",
                            "elapsed_seconds",
                            "report_path",
                            "version"}) {
        if (report.contains(key) && !report[key].is_null()) {
            result[key] = report[key];
        }
    }
    return result.dump(2);
}

}  // namespace vsag::autotune
