
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

#include <vsag/vsag.h>

#include <argparse/argparse.hpp>
#include <iostream>
#include <string>
#include <tabulate/table.hpp>

#include "case/eval_case.h"
#include "eval_config.h"
#include "logger.h"
#include "typing.h"

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
    parser.add_argument("--delete-index-after-search")
        .default_value(false)
        .help("Delete index after search");
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

using name2config = std::pair<std::string, YAML::Node>;
std::vector<name2config>
ParseYamlFile(const std::string& yaml_file) {
    using Node = YAML::Node;
    Node config_all = YAML::LoadFile(yaml_file);
    std::vector<name2config> configs;
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
        configs.emplace_back(std::make_pair<>(it->first.as<std::string>(), config));
    }
    return configs;
}

#define JSON_GET(varname, jsonobj, defaultvalue) \
    std::string varname;                         \
    try {                                        \
        varname = jsonobj;                       \
    } catch (...) {                              \
        varname = defaultvalue;                  \
    }

void
print_as_table(vsag::eval::JsonType results) {
    using namespace tabulate;
    Table table;
    table.add_row({"Name",
                   "NumVectors",
                   "Dim",
                   "DataType",
                   "MetricType",
                   "IndexParam",
                   "BuildTime",
                   "TPS",
                   "SearchParam",
                   "QPS",
                   "LatencyAvg(ms)",
                   "RecallAvg"});
    for (const auto& [key, value] : results.items()) {
        JSON_GET(
            num_vectors, std::to_string(value["dataset_info"]["base_count"].get<int>()), "N/A");
        JSON_GET(dim, std::to_string(value["dataset_info"]["dim"].get<int>()), "N/A");
        JSON_GET(data_type, value["dataset_info"]["data_type"], "N/A");
        JSON_GET(metric_type, value["index_info"]["metric_type"], "N/A");
        JSON_GET(index_param, value["index_info"]["index_param"].dump(), "N/A");
        JSON_GET(build_time, std::to_string(value["duration(s)"].get<float>()), "N/A");
        JSON_GET(tps, std::to_string(value["tps"].get<float>()), "N/A");
        JSON_GET(search_param, value["search_param"], "N/A");
        JSON_GET(qps, std::to_string(value["qps"].get<float>()), "N/A");
        JSON_GET(latency_avg, std::to_string(value["latency_avg(ms)"].get<float>()), "N/A");
        JSON_GET(recall_avg, std::to_string(value["recall_avg"].get<float>()), "N/A");

        table.add_row({key,
                       num_vectors,
                       dim,
                       data_type,
                       metric_type,
                       index_param,
                       build_time,
                       tps,
                       search_param,
                       qps,
                       latency_avg,
                       recall_avg});
    }

    table.column(5).format().width(40);

    std::cout << table << std::endl;
}

int
main(int argc, char** argv) {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
    vsag::eval::EvalConfig config;
    if (argc == 2) {
        std::string yaml_file = argv[1];
        auto nodes = ParseYamlFile(yaml_file);

        vsag::eval::JsonType results;
        for (auto& node : nodes) {
            config = vsag::eval::EvalConfig::Load(node.second);
            auto eval_case = vsag::eval::EvalCase::MakeInstance(config);
            if (eval_case != nullptr) {
                results[node.first] = eval_case->Run();
            }
        }

        print_as_table(results);
    } else {
        argparse::ArgumentParser program("eval_performance");
        ParseArgs(program, argc, argv);
        config = vsag::eval::EvalConfig::Load(program);
        auto eval_case = vsag::eval::EvalCase::MakeInstance(config);
        if (eval_case != nullptr) {
            std::cout << eval_case->Run() << std::endl;
        }
    }
}
