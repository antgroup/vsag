
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
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <tabulate/markdown_exporter.hpp>
#include <tabulate/table.hpp>

#include "./eval_config.h"
#include "./eval_job.h"
#include "case/eval_case.h"
#include "common.h"
#include "exporter/exporter.h"
#include "exporter/formatter.h"
#include "impl/logger/logger.h"
#include "typing.h"
#include "vsag/options.h"

void
check_args(argparse::ArgumentParser& parser) {
    auto mode = parser.get<std::string>("--type");
    if (mode == "search") {
        auto search_mode = parser.get<std::string>("--search_params");
        if (search_mode.empty()) {
            throw std::runtime_error(R"(When "--type" is "search", "--search_params" is required)");
        }
    }
}

void
parse_args(argparse::ArgumentParser& parser, int argc, char** argv) {
    // index
    parser.add_argument<std::string>("--datapath", "-d")
        .required()
        .help("The hdf5 file path for eval");
    parser.add_argument<std::string>("--type", "-t")
        .required()
        .choices("build", "search")
        .help(R"(The eval method to select, choose from {"build", "search"})");
    parser.add_argument<std::string>("--index_name", "-n")
        .required()
        .help("The name of index for create index");
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

    // metrics
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
        check_args(parser);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
    }
}

vsag::eval::eval_job
parse_yaml_file(const std::string& yaml_file) {
    using Node = YAML::Node;
    Node config_all = YAML::LoadFile(yaml_file);

    vsag::eval::eval_job cac;
    try {
        if (config_all["global"]) {
            if (config_all["global"]["exporters"]) {
                auto exporters_root = config_all["global"]["exporters"];
                if (exporters_root.IsMap()) {
                    for (auto it = exporters_root.begin(); it != exporters_root.end(); ++it) {
                        auto exporter = vsag::eval::exporter::Load(it->second);
                        cac.exporters.emplace_back(exporter);
                    }
                }
            }

            if (config_all["global"]["num_threads_building"]) {
                cac.num_threads_building = vsag::eval::check_and_get_value<int32_t>(
                    config_all["global"], "num_threads_building");
            }

            if (config_all["global"]["num_threads_searching"]) {
                cac.num_threads_searching = vsag::eval::check_and_get_value<int32_t>(
                    config_all["global"], "num_threads_searching");
            }
        }
    } catch (YAML::Exception& e) {
        std::cerr << "Error parsing YAML(global): " << e.what() << std::endl;
        exit(-1);
    }

    for (auto it = config_all.begin(); it != config_all.end(); ++it) {
        auto config = it->second;
        try {
            // `global` is a reserved section, process otherwhere
            if (it->first.as<std::string>() == "global") {
                continue;
            }

            if (config.IsMap()) {
                vsag::eval::EvalConfig::CheckKeyAndType(config);
            } else {
                std::cerr << "The root node is not a map!" << std::endl;
                exit(-1);
            }
        } catch (YAML::Exception& e) {
            std::cerr << "Error parsing YAML: " << e.what() << std::endl;
            exit(-1);
        }
        // just separate YAML nodes by name
        cac.cases.emplace_back(std::make_pair<>(it->first.as<std::string>(), config));
    }
    return cac;
}

int
main(int argc, char** argv) {
    using vsag::eval::EvalCase;
    using vsag::eval::EvalConfig;
    using vsag::eval::Exporter;
    using vsag::eval::Formatter;

    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
    vsag::eval::EvalConfig config;
    if (argc == 2) {
        std::string yaml_file = argv[1];
        auto job = parse_yaml_file(yaml_file);

        vsag::eval::JsonType results;
        for (auto& [name, case_yaml_node] : job.cases) {
            config = EvalConfig::Load(case_yaml_node, job);
            vsag::Options::Instance().set_num_threads_building(config.num_threads_building);

            auto eval_case = EvalCase::MakeInstance(config);
            if (eval_case != nullptr) {
                results[name] = eval_case->Run();
            }
        }

        // <format, formatted_results>
        std::unordered_map<std::string, std::string> cached_strings;
        for (const auto& exporter : job.exporters) {
            // std::cout << "export to " << exporter.to << " in " << exporter.format << std::endl;

            // convert at first time
            if (cached_strings.find(exporter.format) == cached_strings.end()) {
                cached_strings[exporter.format] =
                    Formatter::Create(exporter.format)->Format(results);
            }
            std::string formatted_string = cached_strings[exporter.format];

            Exporter::Create(exporter.to, exporter.vars)->Export(formatted_string);
        }

        // by default, eval output as table/text format
        if (job.exporters.empty()) {
            std::cout << Formatter::Create("table")->Format(results) << std::endl;
        }
    } else {
        argparse::ArgumentParser program("eval_performance");
        parse_args(program, argc, argv);
        config = vsag::eval::EvalConfig::Load(program);
        auto eval_case = vsag::eval::EvalCase::MakeInstance(config);
        if (eval_case != nullptr) {
            std::cout << eval_case->Run() << std::endl;
        }
    }
}
