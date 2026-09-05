
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
#include "monitor/http_server_monitor.h"
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
    vsag::eval::EvalConfig::AddCliArguments(parser);

    try {
        parser.parse_args(argc, argv);
        check_args(parser);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
    }
}

vsag::eval::EvalJob
parse_yaml_file(const std::string& yaml_file) {
    using Node = YAML::Node;
    Node config_all = YAML::LoadFile(yaml_file);

    vsag::eval::EvalJob cac;
    try {
        if (config_all["global"]) {
            if (config_all["global"]["exporters"]) {
                auto exporters_root = config_all["global"]["exporters"];
                if (exporters_root.IsMap()) {
                    for (auto it = exporters_root.begin(); it != exporters_root.end(); ++it) {
                        auto exporter = vsag::eval::ExporterConfig::Load(it->second);
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

            // Parse HTTP server config
            if (config_all["global"]["http_server"]) {
                vsag::eval::HttpServer http_cfg;
                auto http_node = config_all["global"]["http_server"];
                if (http_node["enabled"]) {
                    http_cfg.enabled = http_node["enabled"].as<bool>();
                }
                if (http_node["port"]) {
                    http_cfg.port = http_node["port"].as<int>();
                }
                cac.http_server = http_cfg;
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
    using vsag::eval::HttpServerMonitor;

    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
    vsag::eval::EvalConfig config;
    if (argc == 2) {
        std::string yaml_file = argv[1];
        auto job = parse_yaml_file(yaml_file);

        // Start HTTP monitor if configured
        std::unique_ptr<HttpServerMonitor> http_monitor;
        if (job.http_server.has_value() && job.http_server->enabled) {
            http_monitor = std::make_unique<HttpServerMonitor>(job.http_server->port);
            http_monitor->Start();
            http_monitor->SetTotalCases(static_cast<int>(job.cases.size()));
            EvalCase::SetHttpMonitor(http_monitor.get());
            std::cout << "[Eval] HTTP Monitor started on http://0.0.0.0:" << job.http_server->port
                      << std::endl;
        }

        vsag::eval::JsonType results;
        for (auto& [name, case_yaml_node] : job.cases) {
            EvalCase::SetCurrentCaseName(name);

            config = EvalConfig::Load(case_yaml_node, job);
            vsag::Options::Instance().set_num_threads_building(config.num_threads_building);

            auto eval_case = EvalCase::MakeInstance(config);
            if (eval_case != nullptr) {
                results[name] = eval_case->Run();
                EvalCase::MarkCaseCompleted();
            }
        }

        // Stop HTTP monitor
        if (http_monitor) {
            http_monitor->Stop();
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
