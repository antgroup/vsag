
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
#include <nlohmann/json.hpp>
#include <string>

namespace argparse {
class ArgumentParser;
}

namespace YAML {
class Node;
}

namespace vsag::eval {

struct EvalJob;

class EvalConfig {
public:
    // The loaders below are implemented in eval_config_loader.cpp, which needs argparse and
    // yaml-cpp and is therefore not part of the dependency-free evaluation core.
    static EvalConfig
    Load(argparse::ArgumentParser& parser);

    static EvalConfig
    Load(YAML::Node& yaml_node, const EvalJob& global_options);

    static void
    CheckKeyAndType(YAML::Node& yaml_node);

    // Builds a config from a flat JSON object keyed by the field names below. Unknown keys and
    // values of the wrong type raise std::invalid_argument. `build_param` and `search_param` may be
    // given either as JSON strings or as JSON objects.
    static EvalConfig
    FromJson(const nlohmann::json& json);

public:
    std::string dataset_path;
    std::string action_type;
    std::string index_name;
    std::string build_param;
    std::string index_path{"/tmp/performance/index"};

    std::string search_param;
    std::string search_mode{"knn"};
    int top_k{10};
    float radius{0.5F};
    uint64_t search_query_count{100'000L};
    bool delete_index_after_search{false};

    int32_t num_threads_building{1};
    int32_t num_threads_searching{1};

    bool enable_recall{true};
    bool enable_percent_recall{true};
    bool enable_qps{true};
    bool enable_tps{true};
    bool enable_memory{true};
    bool enable_latency{true};
    bool enable_percent_latency{true};
    bool use_id_based_recall{false};

    EvalConfig() = default;
};

}  // namespace vsag::eval
