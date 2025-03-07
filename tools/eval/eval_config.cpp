
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

#include "./eval_config.h"

#include "./common.h"

namespace vsag::eval {

template <class T = std::string>
void
check_and_get_value(const YAML::Node& node, const std::string& key, T& value) {
    if (node[key].IsDefined()) {
        value = node[key].as<T>();
    }
};

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

    config.delete_index_after_search = parser.get<bool>("--delete-index-after-search");

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

EvalConfig
EvalConfig::Load(YAML::Node& yaml_node, const eval_job& global_options) {
    EvalConfig config;

    // set global options at first
    if (global_options.num_threads_building.has_value()) {
        config.num_threads_building = global_options.num_threads_building.value();
    }
    if (global_options.num_threads_searching.has_value()) {
        config.num_threads_searching = global_options.num_threads_searching.value();
    }

    // set options by case
    config.dataset_path = yaml_node["datapath"].as<std::string>();
    config.action_type = yaml_node["type"].as<std::string>();
    config.build_param = yaml_node["create_params"].as<std::string>();
    config.index_name = yaml_node["index_name"].as<std::string>();
    check_and_get_value<>(yaml_node, "search_mode", config.search_mode);
    check_and_get_value<>(yaml_node, "search_params", config.search_param);

    check_and_get_value<>(yaml_node, "index_path", config.index_path);
    check_and_get_value<int>(yaml_node, "topk", config.top_k);
    check_and_get_value<float>(yaml_node, "range", config.radius);

    check_and_get_value<bool>(
        yaml_node, "delete_index_after_search", config.delete_index_after_search);

    check_and_get_value<int>(yaml_node, "num_threads_building", config.num_threads_building);
    check_and_get_value<int>(yaml_node, "num_threads_searching", config.num_threads_searching);

    bool disable = false;
    check_and_get_value<bool>(yaml_node, "disable_recall", disable);
    if (disable == true) {
        config.enable_recall = false;
        disable = false;
    }
    check_and_get_value<bool>(yaml_node, "disable_recall", disable);
    if (disable == true) {
        config.enable_recall = false;
        disable = false;
    }
    check_and_get_value<bool>(yaml_node, "disable_percent_recall", disable);
    if (disable == true) {
        config.enable_percent_recall = false;
        disable = false;
    }
    check_and_get_value<bool>(yaml_node, "disable_qps", disable);
    if (disable == true) {
        config.enable_qps = false;
        disable = false;
    }
    check_and_get_value<bool>(yaml_node, "disable_tps", disable);
    if (disable == true) {
        config.enable_tps = false;
        disable = false;
    }
    check_and_get_value<bool>(yaml_node, "disable_memory", disable);
    if (disable == true) {
        config.enable_memory = false;
        disable = false;
    }
    check_and_get_value<bool>(yaml_node, "disable_latency", disable);
    if (disable == true) {
        config.enable_latency = false;
        disable = false;
    }
    check_and_get_value<bool>(yaml_node, "disable_percent_latency", disable);
    if (disable == true) {
        config.enable_percent_latency = false;
        disable = false;
    }

    return config;
}

void
EvalConfig::CheckKeyAndType(YAML::Node& yaml_node) {
    check_exist_and_get_value<>(yaml_node, "datapath");
    check_exist_and_get_value<>(yaml_node, "index_name");
    check_exist_and_get_value<>(yaml_node, "create_params");
    auto action = check_exist_and_get_value<>(yaml_node, "type");
    if (action == "search") {
        check_exist_and_get_value<>(yaml_node, "search_params");
    }
    check_and_get_value<>(yaml_node, "search_mode");
    check_and_get_value<>(yaml_node, "index_path");
    check_and_get_value<int>(yaml_node, "topk");
    check_and_get_value<float>(yaml_node, "range");
    check_and_get_value<bool>(yaml_node, "disable_recall");
    check_and_get_value<bool>(yaml_node, "disable_percent_recall");
    check_and_get_value<bool>(yaml_node, "disable_qps");
    check_and_get_value<bool>(yaml_node, "disable_tps");
    check_and_get_value<bool>(yaml_node, "disable_memory");
    check_and_get_value<bool>(yaml_node, "disable_latency");
    check_and_get_value<bool>(yaml_node, "disable_percent_latency");
}

}  // namespace vsag::eval
