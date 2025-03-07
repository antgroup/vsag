
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

#include "argparse/argparse.hpp"
#include "eval_job.h"
#include "yaml-cpp/yaml.h"

namespace vsag::eval {

class EvalConfig {
public:
    static EvalConfig
    Load(argparse::ArgumentParser& parser);

    static EvalConfig
    Load(YAML::Node& yaml_node, const eval_job& global_options);

    static void
    CheckKeyAndType(YAML::Node& yaml_node);

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

    EvalConfig() = default;
};

}  // namespace vsag::eval
