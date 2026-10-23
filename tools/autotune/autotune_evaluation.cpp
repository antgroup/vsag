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

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <utility>

#include "autotune_internal.h"
#include "eval_config.h"
#include "evaluator.h"
#include "vsag/factory.h"

namespace vsag::autotune::internal {

namespace {

constexpr double BYTES_PER_MEBIBYTE = 1024.0 * 1024.0;

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
tuned_build_metrics(const JsonType& source_raw,
                    const IndexPtr& index,
                    const std::string& index_path,
                    double transform_seconds) {
    auto metrics = build_metrics(source_raw, index_path);
    const auto build = metrics.find("build_seconds");
    if (build != metrics.end()) {
        build->second += transform_seconds;
    }
    metrics["index_memory_mb"] = static_cast<double>(index->GetMemoryUsage()) / BYTES_PER_MEBIBYTE;
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
    config.use_id_based_recall = true;
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

IndexPtr
deserialize_index(const Candidate& candidate, const std::string& path) {
    auto index = create_index(candidate);
    std::ifstream input(path, std::ios::binary);
    if (!input.good()) {
        throw std::runtime_error("failed to open index artifact: " + path);
    }
    auto deserialized = index->Deserialize(input);
    if (!deserialized.has_value()) {
        throw std::runtime_error(deserialized.error().message);
    }
    return index;
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
    output.flush();
    if (!output.good()) {
        throw std::runtime_error("failed to write index artifact: " + path);
    }
}

class TemporaryArtifact {
public:
    explicit TemporaryArtifact(std::string path) : path_(std::move(path)) {
    }

    ~TemporaryArtifact() {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    TemporaryArtifact(const TemporaryArtifact&) = delete;
    TemporaryArtifact&
    operator=(const TemporaryArtifact&) = delete;

private:
    std::string path_;
};

std::optional<uint64_t>
hgraph_max_degree(const Candidate& candidate) {
    if (candidate.index_name != "hgraph" || !candidate.create_params.is_object() ||
        !candidate.create_params.contains("index_param") ||
        !candidate.create_params["index_param"].is_object()) {
        return std::nullopt;
    }
    const auto& params = candidate.create_params["index_param"];
    if (!params.contains("max_degree") || !params["max_degree"].is_number_integer()) {
        return std::nullopt;
    }
    if (params["max_degree"].is_number_unsigned()) {
        const auto degree = params["max_degree"].get<uint64_t>();
        return degree > 0 ? std::optional<uint64_t>(degree) : std::nullopt;
    }
    const auto degree = params["max_degree"].get<int64_t>();
    return degree > 0 ? std::optional<uint64_t>(static_cast<uint64_t>(degree)) : std::nullopt;
}

std::optional<std::string>
hgraph_reuse_key(const Candidate& candidate, bool allow_quantization_tune) {
    if (!hgraph_max_degree(candidate).has_value()) {
        return std::nullopt;
    }
    auto create_params = candidate.create_params;
    auto& params = create_params["index_param"];
    if (!params.contains("base_quantization_type") ||
        !params["base_quantization_type"].is_string()) {
        return std::nullopt;
    }
    params.erase("max_degree");
    if (allow_quantization_tune) {
        params.erase("base_quantization_type");
    }
    return "hgraph_tune\n" + create_params.dump();
}

Candidate
canonical_candidate(const Candidate& candidate, uint64_t max_degree, bool allow_quantization_tune) {
    auto canonical = candidate;
    auto& params = canonical.create_params["index_param"];
    params["max_degree"] = max_degree;
    if (allow_quantization_tune) {
        params["base_quantization_type"] = "fp32";
        params["store_raw_vector"] = true;
    }
    return canonical;
}

void
tune_index(const IndexPtr& index, const Candidate& candidate, bool disable_future_tuning) {
    auto target = candidate.create_params;
    const auto preserve_tuning = target["index_param"].value("store_raw_vector", false);
    auto tuned = index->Tune(target.dump(), disable_future_tuning && !preserve_tuning);
    if (!tuned.has_value()) {
        throw std::runtime_error(tuned.error().message);
    }
    if (!tuned.value()) {
        throw std::runtime_error("index does not support the requested tuning transformation");
    }
}

bool
uses_build_cost(const RequestContext& request) {
    return request.objective == "build_seconds" ||
           request.objective == "build_and_search_seconds" ||
           request.constraints.find("build_seconds") != request.constraints.end() ||
           request.constraints.find("build_and_search_seconds") != request.constraints.end();
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
    Evaluation evaluation;
    std::map<std::string, std::vector<uint64_t>> groups;
    const auto reuse_hgraph_builds = !uses_build_cost(request);
    for (uint64_t i = 0; i < candidates.size(); ++i) {
        auto key = candidates[i].index_name + "\n" + candidates[i].create_params.dump();
        if (reuse_hgraph_builds) {
            const auto reuse_key = hgraph_reuse_key(candidates[i], request.allow_quantization_tune);
            if (reuse_key.has_value()) {
                key = *reuse_key;
            }
        }
        groups[key].emplace_back(i);
    }

    struct BuildState {
        IndexPtr index;
        JsonType report;
        MetricMap metrics;
        std::string path;
    };

    struct SourceState {
        IndexPtr index;
        Candidate candidate;
        JsonType raw;
        std::string id;
    };

    uint64_t build_number = 0;
    uint64_t source_number = 0;
    uint64_t trial_number = 0;

    const auto artifact_path = [&run_path](const std::string& id) {
        return (std::filesystem::path(run_path) / "artifacts" / (id + ".index")).string();
    };

    const auto new_build_state =
        [&](const Candidate& candidate, const std::string& build_id, const std::string& strategy) {
            const auto index_path = artifact_path(build_id);
            const auto artifact_source = strategy == "full_build" ? "generated" : strategy;
            return BuildState{nullptr,
                              {{"build_id", build_id},
                               {"index_name", candidate.index_name},
                               {"create_params", candidate.create_params},
                               {"strategy", strategy},
                               {"status", "failed"},
                               {"metrics", JsonType::object()},
                               {"failure", nullptr},
                               {"artifacts",
                                {{"index_path", index_path},
                                 {"source", artifact_source},
                                 {"use_existing_index", false},
                                 {"retained", true}}}},
                              {},
                              index_path};
        };

    const auto build_index = [&](const Candidate& candidate, const std::string& build_id) {
        auto state = new_build_state(candidate, build_id, "full_build");
        const auto start = std::chrono::steady_clock::now();
        try {
            state.index = create_index(candidate);
            auto raw = eval::EvaluateBuild(state.index, request.dataset, build_config(candidate));
            serialize_index(state.index, state.path);
            state.metrics = build_metrics(raw, state.path);
            if (request.include_raw_eval) {
                state.report["raw_eval_result"] = std::move(raw);
            }
            state.report["metrics"] = metrics_json(state.metrics);
            state.report["status"] = "success";
        } catch (const std::exception& error) {
            state.report["failure"] = Failure("build", "build_evaluation_failed", error.what());
            std::error_code cleanup_error;
            std::filesystem::remove(state.path, cleanup_error);
            state.report["artifacts"]["retained"] = false;
            state.index.reset();
        }
        state.report["elapsed_seconds"] = elapsed(start);
        return state;
    };

    const auto evaluate_searches = [&](const BuildState& build,
                                       const std::vector<uint64_t>& indexes) {
        const auto evaluate = [&](const Candidate& candidate) -> std::optional<double> {
            const auto trial_id = "trial-" + std::to_string(trial_number++);
            JsonType trial{{"trial_id", trial_id},
                           {"build_id", build.report["build_id"]},
                           {"index_name", candidate.index_name},
                           {"create_params", candidate.create_params},
                           {"search_params", candidate.search_params},
                           {"status", "failed"},
                           {"metrics", metrics_json(build.metrics)},
                           {"failure", nullptr},
                           {"artifacts", build.report["artifacts"]}};
            std::optional<double> recall;
            const auto search_start = std::chrono::steady_clock::now();
            if (build.report["status"] != "success") {
                trial["failure"] =
                    Failure("search", "build_failed", "search skipped because build failed");
            } else {
                try {
                    const auto measured_start = std::chrono::steady_clock::now();
                    auto raw = eval::EvaluateSearch(
                        build.index, request.dataset, search_config(request, candidate));
                    auto metrics = search_metrics(raw, elapsed(measured_start));
                    for (const auto& [name, value] : build.metrics) {
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
    };

    const auto record_and_search = [&](BuildState state, const std::vector<uint64_t>& indexes) {
        evaluation.builds.emplace_back(state.report);
        evaluate_searches(state, indexes);
    };

    const auto evaluate_native = [&](const std::vector<uint64_t>& indexes,
                                     std::optional<std::string> build_id = std::nullopt) {
        const auto id =
            build_id.has_value() ? *build_id : "build-" + std::to_string(build_number++);
        record_and_search(build_index(candidates[indexes.front()], id), indexes);
    };

    for (const auto& [unused, indexes] : groups) {
        (void)unused;
        std::map<std::string, std::vector<uint64_t>> create_groups;
        for (const auto candidate_index : indexes) {
            create_groups[candidates[candidate_index].create_params.dump()].emplace_back(
                candidate_index);
        }
        if (create_groups.size() <= 1) {
            evaluate_native(indexes);
            continue;
        }

        std::map<uint64_t, std::vector<std::string>, std::greater<>> degree_groups;
        bool reusable = true;
        for (const auto& [create_key, candidate_indexes] : create_groups) {
            const auto degree = hgraph_max_degree(candidates[candidate_indexes.front()]);
            if (!degree.has_value()) {
                reusable = false;
                break;
            }
            degree_groups[*degree].emplace_back(create_key);
        }
        if (!reusable) {
            for (const auto& [create_key, candidate_indexes] : create_groups) {
                (void)create_key;
                evaluate_native(candidate_indexes);
            }
            continue;
        }

        const auto source_id = "source-build-" + std::to_string(source_number++);
        const auto max_degree = degree_groups.begin()->first;
        const auto& source_indexes = create_groups.at(degree_groups.begin()->second.front());
        SourceState source{
            nullptr,
            canonical_candidate(
                candidates[source_indexes.front()], max_degree, request.allow_quantization_tune),
            JsonType::object(),
            source_id};
        try {
            source.index = create_index(source.candidate);
            source.raw =
                eval::EvaluateBuild(source.index, request.dataset, build_config(source.candidate));
        } catch (const std::exception&) {
            reusable = false;
        }
        if (!reusable) {
            for (const auto& [create_key, candidate_indexes] : create_groups) {
                (void)create_key;
                evaluate_native(candidate_indexes);
            }
            continue;
        }

        double degree_transform_seconds = 0.0;
        for (auto degree = degree_groups.begin(); degree != degree_groups.end(); ++degree) {
            if (!reusable) {
                for (const auto& create_key : degree->second) {
                    evaluate_native(create_groups.at(create_key));
                }
                continue;
            }

            source.candidate.create_params["index_param"]["max_degree"] = degree->first;
            const auto snapshot_path =
                artifact_path(source.id + "-" + std::to_string(degree->first));
            const TemporaryArtifact snapshot(snapshot_path);
            try {
                serialize_index(source.index, snapshot_path);
            } catch (const std::exception&) {
                reusable = false;
            }

            if (reusable) {
                for (const auto& create_key : degree->second) {
                    const auto& candidate_indexes = create_groups.at(create_key);
                    const auto& candidate = candidates[candidate_indexes.front()];
                    const auto build_id = "build-" + std::to_string(build_number++);
                    auto state = new_build_state(candidate, build_id, "hgraph_tune");
                    bool tuned = false;
                    try {
                        state.index = deserialize_index(source.candidate, snapshot_path);
                        const auto transform_start = std::chrono::steady_clock::now();
                        tune_index(state.index, candidate, true);
                        const auto transform_seconds =
                            degree_transform_seconds + elapsed(transform_start);
                        serialize_index(state.index, state.path);
                        state.metrics = tuned_build_metrics(
                            source.raw, state.index, state.path, transform_seconds);
                        state.report["strategy"] = "hgraph_tune";
                        state.report["source_build_id"] = source.id;
                        state.report["transform_seconds"] = transform_seconds;
                        state.report["metrics"] = metrics_json(state.metrics);
                        state.report["status"] = "success";
                        state.report["elapsed_seconds"] = elapsed(transform_start);
                        if (request.include_raw_eval) {
                            auto raw = source.raw;
                            raw["duration(s)"] = state.metrics["build_seconds"];
                            raw["index_info"] = candidate.create_params;
                            raw["index_memory(B)"] = state.index->GetMemoryUsage();
                            state.report["raw_eval_result"] = std::move(raw);
                        }
                        tuned = true;
                    } catch (const std::exception&) {
                        std::error_code cleanup_error;
                        std::filesystem::remove(state.path, cleanup_error);
                    }

                    if (tuned) {
                        record_and_search(std::move(state), candidate_indexes);
                    } else {
                        evaluate_native(candidate_indexes, build_id);
                    }
                }
            } else {
                for (const auto& create_key : degree->second) {
                    evaluate_native(create_groups.at(create_key));
                }
            }

            const auto next = std::next(degree);
            if (!reusable || next == degree_groups.end()) {
                continue;
            }
            source.candidate.create_params["index_param"]["max_degree"] = next->first;
            try {
                const auto transform_start = std::chrono::steady_clock::now();
                tune_index(source.index, source.candidate, false);
                degree_transform_seconds += elapsed(transform_start);
            } catch (const std::exception&) {
                reusable = false;
            }
        }
    }
    return evaluation;
}

Evaluation
EvaluateCandidates(const SearchTuningRequest& tuning_request,
                   const std::vector<Candidate>& candidates) {
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
