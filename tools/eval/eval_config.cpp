
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

#include "eval_config.h"

namespace vsag::eval {
EvalConfig
EvalConfig::Load(argparse::ArgumentParser& parser) {
    EvalConfig config;
    config.dataset_path = parser.get("--datapath");
    config.action_type = parser.get("--type");
    config.build_param = parser.get("--create_params");
    config.index_name = parser.get("--index_name");
    config.index_path = parser.get("--index_path");

    config.search_param = parser.get("--search_params");
    config.search_mode = parser.get("--search_mode");

    config.top_k = parser.get<int>("--topk");
    config.radius = parser.get<float>("--range");

    if (parser.get<bool>("--disable_recall")) {
        config.enable_recall = false;
    }
    if (parser.get<bool>("--disable_percent_recall")) {
        config.enable_percent_recall = false;
    }
    if (parser.get<bool>("--disable_memory")) {
        config.enable_memory = false;
    }
    if (parser.get<bool>("--disable_latency")) {
        config.enable_latency = false;
    }
    if (parser.get<bool>("--disable_qps")) {
        config.enable_qps = false;
    }
    if (parser.get<bool>("--disable_tps")) {
        config.enable_tps = false;
    }
    if (parser.get<bool>("--disable_percent_latency")) {
        config.enable_percent_latency = false;
    }

    return config;
}
}  // namespace vsag::eval
