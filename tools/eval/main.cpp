
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

#include <argparse/argparse.hpp>
#include <iostream>
#include <string>

#include "eval_case.h"
#include "eval_config.h"

void
CheckArgs(argparse::ArgumentParser& parser) {
    auto mode = parser.get<std::string>("--type");
    if (mode == "search") {
        auto search_mode = parser.get<std::string>("--search_params");
        if (search_mode.empty()) {
            throw std::runtime_error(R"(When "--type" is "search", "--search_params" is required)");
        }
    }
}

void
ParseArgs(argparse::ArgumentParser& parser, int argc, char** argv) {
    parser.add_argument<std::string>("--datapath", "-d")
        .required()
        .help("The hdf5 file path for eval");
    parser.add_argument<std::string>("--type", "-t")
        .required()
        .choices("build", "search")
        .help(R"(The eval method to select, choose from {"build", "search"})");
    parser.add_argument<std::string>("--index_name", "-n")
        .required()
        .help("The name of index fot create index");
    parser.add_argument<std::string>("--create_params", "-c")
        .required()
        .help("The param for create index");
    parser.add_argument<std::string>("--index_path", "-i")
        .default_value("/tmp/performance/index")
        .help("The index path for load or save");
    parser.add_argument<std::string>("--search_params", "-s")
        .default_value("")
        .help("The param for search");
    parser.add_argument<std::string>("--search_mode")
        .default_value("knn")
        .choices("knn", "range", "knn_filter", "range_filter")
        .help(
            "The mode supported while use 'search' type,"
            " choose from {\"knn\", \"range\", \"knn_filter\", \"range_filter\"}");
    parser.add_argument("--topk")
        .default_value(10)
        .help("The topk value for knn search or knn_filter search")
        .scan<'i', int>();
    parser.add_argument("--range")
        .default_value(0.5f)
        .help("The range value for range search or range_filter search")
        .scan<'f', float>();
    parser.add_argument("--disable_recall")
        .default_value(false)
        .help("Disable average recall eval");
    parser.add_argument("--disable_percent_recall")
        .default_value(false)
        .help("Disable percent recall eval, include p0, p10, p30, p50, p70, p90");
    parser.add_argument("--disable_qps").default_value(false).help("Disable qps eval");
    parser.add_argument("--disable_tps").default_value(false).help("Disable tps eval");
    parser.add_argument("--disable_memory").default_value(false).help("Disable memory eval");
    parser.add_argument("--disable_latency")
        .default_value(false)
        .help("Disable average latency eval");
    parser.add_argument("--disable_percent_latency")
        .default_value(false)
        .help("Disable percent latency eval, include p50, p80, p90, p95, p99");

    try {
        parser.parse_args(argc, argv);
        CheckArgs(parser);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
    }
}
template <class T = std::string>
T
check_exist_and_get_value(const YAML::Node& node, const std::string& key) {
    if (not node[key].IsDefined()) {
        throw std::invalid_argument(key + " is not in config file");
    }
    return node[key].as<T>();
};

template <class T = std::string>
T
check_and_get_value(const YAML::Node& node, const std::string& key) {
    if (node[key].IsDefined()) {
        return node[key].as<T>();
    } else {
        return T();
    }
};

std::vector<YAML::Node>
ParseYamlFile(const std::string& yaml_file) {
    using Node = YAML::Node;
    Node config_all = YAML::LoadFile(yaml_file);
    std::vector<YAML::Node> nodes;
    for (auto it = config_all.begin(); it != config_all.end(); ++it) {
        auto config = it->second;
        try {
            if (config.IsMap()) {
                check_exist_and_get_value<>(config, "datapath");
                auto action = check_exist_and_get_value<>(config, "type");
                check_exist_and_get_value<>(config, "index_name");
                check_exist_and_get_value<>(config, "create_params");
                if (action == "search") {
                    check_exist_and_get_value<>(config, "search_params");
                }
                check_and_get_value<>(config, "search_mode");
                check_and_get_value<>(config, "index_path");
                check_and_get_value<int>(config, "topk");
                check_and_get_value<float>(config, "range");
                check_and_get_value<bool>(config, "disable_recall");
                check_and_get_value<bool>(config, "disable_percent_recall");
                check_and_get_value<bool>(config, "disable_qps");
                check_and_get_value<bool>(config, "disable_tps");
                check_and_get_value<bool>(config, "disable_memory");
                check_and_get_value<bool>(config, "disable_latency");
                check_and_get_value<bool>(config, "disable_percent_latency");

            } else {
                std::cout << "The root node is not a map!" << std::endl;
                exit(-1);
            }
        } catch (YAML::Exception& e) {
            std::cerr << "Error parsing YAML: " << e.what() << std::endl;
            exit(-1);
        }
        nodes.emplace_back(config);
    }
    return nodes;
}

int
main(int argc, char** argv) {
    vsag::eval::EvalConfig config;
    if (argc == 2) {
        std::string yaml_file = argv[1];
        auto nodes = ParseYamlFile(yaml_file);
        for (auto& node : nodes) {
            config = vsag::eval::EvalConfig::Load(node);
            auto eval_case = vsag::eval::EvalCase::MakeInstance(config);
            if (eval_case != nullptr) {
                eval_case->Run();
            }
        }
    } else {
        argparse::ArgumentParser program("eval_performance");
        ParseArgs(program, argc, argv);
        config = vsag::eval::EvalConfig::Load(program);
        auto eval_case = vsag::eval::EvalCase::MakeInstance(config);
        if (eval_case != nullptr) {
            eval_case->Run();
        }
    }
}
