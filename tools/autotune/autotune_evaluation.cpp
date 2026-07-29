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

#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <system_error>

#include "autotune_internal.h"
#include "eval_config.h"
#include "evaluator.h"
#include "vsag/factory.h"

namespace vsag::autotune::internal {

namespace {

constexpr double BYTES_PER_MEBIBYTE = 1024.0 * 1024.0;

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

double
elapsed(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

std::optional<double>
number(const JsonType& value, const std::string& key) {
    if (!value.is_object() || !value.contains(key) || !value[key].is_number()) {
        return std::nullopt;
    }
    return value[key].get<double>();
}

void
set_metric(MetricMap& metrics, const std::string& name, const std::optional<double>& value) {
    if (value.has_value()) {
        metrics[name] = *value;
    }
}

MetricMap
build_metrics(const JsonType& raw, const std::string& index_path) {
    MetricMap metrics;
    set_metric(metrics, "build_seconds", number(raw, "duration(s)"));
    const auto memory = number(raw, "index_memory(B)");
    if (memory.has_value() && *memory > 0.0) {
        metrics["index_memory_mb"] = *memory / BYTES_PER_MEBIBYTE;
    }
    std::error_code error;
    const auto bytes = std::filesystem::file_size(index_path, error);
    if (!error) {
        metrics["index_size_mb"] = static_cast<double>(bytes) / BYTES_PER_MEBIBYTE;
    }
    return metrics;
}

MetricMap
search_metrics(const JsonType& raw, double seconds) {
    MetricMap metrics;
    set_metric(metrics, "recall_at_k", number(raw, "recall_avg"));
    set_metric(metrics, "latency_avg_ms", number(raw, "latency_avg(ms)"));
    set_metric(metrics, "qps", number(raw, "qps"));
    if (raw.contains("latency_detail(ms)") && raw["latency_detail(ms)"].is_object()) {
        set_metric(metrics, "latency_p99_ms", number(raw["latency_detail(ms)"], "p99"));
    }
    const auto memory = number(raw, "index_memory(B)");
    if (memory.has_value() && *memory > 0.0) {
        metrics["index_memory_mb"] = *memory / BYTES_PER_MEBIBYTE;
    }
    metrics["search_seconds"] = seconds;
    return metrics;
}

JsonType
metrics_json(const MetricMap& metrics) {
    JsonType result = JsonType::object();
    for (const auto& [name, value] : metrics) {
        result[name] = value;
    }
    return result;
}

eval::EvalConfig
build_config(const Candidate& candidate) {
    eval::EvalConfig config;
    config.index_name = candidate.index_name;
    config.build_param = candidate.create_params.dump();
    config.enable_memory = false;
    return config;
}

eval::EvalConfig
search_config(const RequestContext& request, const Candidate& candidate) {
    auto config = build_config(candidate);
    config.search_param = candidate.search_params.dump();
    config.search_mode = "knn";
    config.top_k = static_cast<int>(request.top_k);
    config.search_query_count = request.query_count;
    config.num_threads_searching = static_cast<int32_t>(request.concurrency);
    config.enable_memory = false;
    config.enable_recall = request.enable_recall;
    config.enable_percent_recall = false;
    return config;
}

IndexPtr
create_index(const Candidate& candidate) {
    auto created = Factory::CreateIndex(candidate.index_name, candidate.create_params.dump());
    if (!created.has_value()) {
        throw std::runtime_error(created.error().message);
    }
    return created.value();
}

void
serialize_index(const IndexPtr& index, const std::string& path) {
    const auto parent = std::filesystem::path(path).parent_path();
    std::filesystem::create_directories(parent);
    std::ofstream output(path, std::ios::binary);
    if (!output.good()) {
        throw std::runtime_error("failed to open index artifact: " + path);
    }
    auto serialized = index->Serialize(output);
    if (!serialized.has_value()) {
        throw std::runtime_error(serialized.error().message);
    }
}

}  // namespace

void
EvaluateEfSearchRange(const HGraphEfSearchRange& range,
                      double recall_target,
                      const std::function<std::optional<double>(int64_t)>& evaluate) {
    auto low = range.start;
    const auto low_recall = evaluate(low);
    if (!low_recall.has_value() || *low_recall >= recall_target || low == range.stop) {
        return;
    }

    auto high = low;
    while (low < range.stop) {
        high = low > range.stop / 2 ? range.stop : low * 2;
        const auto high_recall = evaluate(high);
        if (!high_recall.has_value()) {
            return;
        }
        if (*high_recall >= recall_target) {
            break;
        }
        if (high == range.stop) {
            return;
        }
        low = high;
    }

    while (high - low > 1) {
        const auto middle = low + (high - low) / 2;
        const auto middle_recall = evaluate(middle);
        if (!middle_recall.has_value()) {
            return;
        }
        if (*middle_recall >= recall_target) {
            high = middle;
        } else {
            low = middle;
        }
    }
}

Evaluation
EvaluateCandidates(const IndexTuningRequest& tuning_request,
                   const std::vector<Candidate>& candidates,
                   const std::string& run_path) {
    const auto& request = tuning_request.context;
    ScopedOpenMpThreads openmp_threads;
    Evaluation evaluation;
    std::map<std::string, std::vector<uint64_t>> groups;
    for (uint64_t i = 0; i < candidates.size(); ++i) {
        const auto key = candidates[i].index_name + "\n" + candidates[i].create_params.dump();
        groups[key].emplace_back(i);
    }

    uint64_t build_number = 0;
    uint64_t trial_number = 0;
    for (const auto& [unused, indexes] : groups) {
        (void)unused;
        const auto& first = candidates[indexes.front()];
        const auto build_id = "build-" + std::to_string(build_number++);
        const auto index_path =
            (std::filesystem::path(run_path) / "artifacts" / (build_id + ".index")).string();
        JsonType build{{"build_id", build_id},
                       {"index_name", first.index_name},
                       {"create_params", first.create_params},
                       {"status", "failed"},
                       {"metrics", JsonType::object()},
                       {"failure", nullptr},
                       {"artifacts",
                        {{"index_path", index_path},
                         {"source", "generated"},
                         {"use_existing_index", false},
                         {"retained", request.keep_intermediate}}}};

        MetricMap shared_metrics;
        IndexPtr index;
        const auto build_start = std::chrono::steady_clock::now();
        try {
            index = create_index(first);
            auto raw = eval::EvaluateBuild(index, request.dataset, build_config(first));
            serialize_index(index, index_path);
            shared_metrics = build_metrics(raw, index_path);
            if (request.include_raw_eval) {
                build["raw_eval_result"] = std::move(raw);
            }
            build["metrics"] = metrics_json(shared_metrics);
            build["status"] = "success";
        } catch (const std::exception& error) {
            build["failure"] = Failure("build", "build_evaluation_failed", error.what());
            std::error_code cleanup_error;
            std::filesystem::remove(index_path, cleanup_error);
            build["artifacts"]["retained"] = false;
        }
        build["elapsed_seconds"] = elapsed(build_start);
        evaluation.builds.emplace_back(build);

        const auto evaluate = [&](const Candidate& candidate) -> std::optional<double> {
            const auto trial_id = "trial-" + std::to_string(trial_number++);
            JsonType trial{{"trial_id", trial_id},
                           {"build_id", build_id},
                           {"index_name", candidate.index_name},
                           {"create_params", candidate.create_params},
                           {"search_params", candidate.search_params},
                           {"status", "failed"},
                           {"metrics", metrics_json(shared_metrics)},
                           {"failure", nullptr},
                           {"artifacts", build["artifacts"]}};
            std::optional<double> recall;
            const auto search_start = std::chrono::steady_clock::now();
            if (build["status"] != "success") {
                trial["failure"] =
                    Failure("search", "build_failed", "search skipped because build failed");
            } else {
                try {
                    const auto measured_start = std::chrono::steady_clock::now();
                    auto raw = eval::EvaluateSearch(
                        index, request.dataset, search_config(request, candidate));
                    auto metrics = search_metrics(raw, elapsed(measured_start));
                    for (const auto& [name, value] : shared_metrics) {
                        metrics.emplace(name, value);
                    }
                    if (metrics.find("build_seconds") != metrics.end()) {
                        metrics["build_and_search_seconds"] =
                            metrics["build_seconds"] + metrics["search_seconds"];
                    }
                    trial["metrics"] = metrics_json(metrics);
                    trial["status"] = "success";
                    recall = number(trial["metrics"], "recall_at_k");
                    if (request.include_raw_eval) {
                        trial["raw_eval_result"] = std::move(raw);
                    }
                } catch (const std::exception& error) {
                    trial["failure"] = Failure("search", "search_evaluation_failed", error.what());
                }
            }
            trial["elapsed_seconds"] = elapsed(search_start);
            evaluation.trials.emplace_back(std::move(trial));
            return recall;
        };

        for (const auto candidate_index : indexes) {
            const auto& candidate = candidates[candidate_index];
            if (!candidate.ef_search_range.has_value()) {
                evaluate(candidate);
                continue;
            }

            const auto recall_target = request.constraints.at("recall_at_k");
            const auto evaluate_ef_search = [&](int64_t ef_search) {
                auto concrete = candidate;
                concrete.search_params["hgraph"]["ef_search"] = ef_search;
                return evaluate(concrete);
            };
            EvaluateEfSearchRange(*candidate.ef_search_range, recall_target, evaluate_ef_search);
        }

        if (!request.keep_intermediate) {
            std::error_code error;
            std::filesystem::remove(index_path, error);
        }
    }
    return evaluation;
}

Evaluation
EvaluateCandidates(const SearchTuningRequest& tuning_request,
                   const std::vector<Candidate>& candidates) {
    ScopedOpenMpThreads openmp_threads;
    const auto& request = tuning_request.context;
    Evaluation evaluation;
    uint64_t trial_number = 0;

    const auto evaluate = [&](const Candidate& candidate) -> std::optional<double> {
        JsonType trial{{"trial_id", "trial-" + std::to_string(trial_number++)},
                       {"index_name", candidate.index_name},
                       {"search_params", candidate.search_params},
                       {"status", "failed"},
                       {"metrics", JsonType::object()},
                       {"failure", nullptr}};
        std::optional<double> recall;
        const auto start = std::chrono::steady_clock::now();
        try {
            const auto measured_start = std::chrono::steady_clock::now();
            auto raw = eval::EvaluateSearch(
                tuning_request.index, request.dataset, search_config(request, candidate));
            const auto metrics = search_metrics(raw, elapsed(measured_start));
            trial["metrics"] = metrics_json(metrics);
            trial["status"] = "success";
            recall = number(trial["metrics"], "recall_at_k");
            if (request.include_raw_eval) {
                trial["raw_eval_result"] = std::move(raw);
            }
        } catch (const std::exception& error) {
            trial["failure"] = Failure("search", "search_evaluation_failed", error.what());
        }
        trial["elapsed_seconds"] = elapsed(start);
        evaluation.trials.emplace_back(std::move(trial));
        return recall;
    };

    for (const auto& candidate : candidates) {
        if (!candidate.ef_search_range.has_value()) {
            evaluate(candidate);
            continue;
        }
        const auto recall_target = request.constraints.at("recall_at_k");
        const auto evaluate_ef_search = [&](int64_t ef_search) {
            auto concrete = candidate;
            concrete.search_params["hgraph"]["ef_search"] = ef_search;
            return evaluate(concrete);
        };
        EvaluateEfSearchRange(*candidate.ef_search_range, recall_target, evaluate_ef_search);
    }
    return evaluation;
}

}  // namespace vsag::autotune::internal
