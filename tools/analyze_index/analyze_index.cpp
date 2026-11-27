
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

#include <unistd.h>
#include <vsag/vsag.h>

#include <argparse/argparse.hpp>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

namespace {

constexpr int EXIT_ANALYSIS_FAILED = 1;
constexpr int EXIT_USAGE = 2;
// Kept for compatibility with scripts written for earlier versions of this tool.
constexpr const char* LEGACY_EMPTY_QUERY_PATH = "empty";

const char*
error_code(vsag::ErrorType type) {
    switch (type) {
        case vsag::ErrorType::INTERNAL_ERROR:
            return "internal_error";
        case vsag::ErrorType::INVALID_ARGUMENT:
            return "invalid_argument";
        case vsag::ErrorType::WRONG_STATUS:
            return "wrong_status";
        case vsag::ErrorType::BUILD_TWICE:
            return "build_twice";
        case vsag::ErrorType::INDEX_NOT_EMPTY:
            return "index_not_empty";
        case vsag::ErrorType::UNSUPPORTED_INDEX:
            return "unsupported_index";
        case vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION:
            return "unsupported_index_operation";
        case vsag::ErrorType::DIMENSION_NOT_EQUAL:
            return "dimension_not_equal";
        case vsag::ErrorType::INDEX_EMPTY:
            return "index_empty";
        case vsag::ErrorType::NO_ENOUGH_MEMORY:
            return "no_enough_memory";
        case vsag::ErrorType::READ_ERROR:
            return "read_error";
        case vsag::ErrorType::MISSING_FILE:
            return "missing_file";
        case vsag::ErrorType::INVALID_BINARY:
            return "invalid_binary";
        case vsag::ErrorType::UNKNOWN_ERROR:
            break;
    }
    return "unknown_error";
}

void
add_arguments(argparse::ArgumentParser& parser) {
    parser.add_description(
        "Analyze a serialized VSAG index and print a JSON report to stdout. Diagnostics and "
        "errors go to stderr. Exit status: 0 on success, 1 when the analysis fails, 2 for "
        "invalid command lines.");
    parser.add_argument("--index_path", "-i", "--index-path")
        .required()
        .help("path of the serialized index file to analyze");
    parser.add_argument("--build_parameter", "-bp", "--build-parameter")
        .default_value(std::string("default"))
        .help(
            "build parameters (JSON) used to reload the index; \"default\" uses the "
            "parameters recorded in the file");
    parser.add_argument("--query_path", "-qp", "--query-path")
        .default_value(std::string())
        .help("query dataset for dynamic analysis; omit it for static analysis only");
    parser.add_argument("--query_data_type", "--query-data-type")
        .default_value(std::string("auto"))
        .help("query file layout: auto, dense or sparse");
    parser.add_argument("--base_path", "--base-path")
        .default_value(std::string())
        .help("sindi only: sparse CSR base dataset for exact ground truth and pruning metrics");
    parser.add_argument("--groundtruth_path", "--groundtruth-path")
        .default_value(std::string())
        .help("sindi only: existing ground-truth file for dynamic analysis");
    parser.add_argument("--save_groundtruth_path", "--save-groundtruth-path")
        .default_value(std::string())
        .help("sindi only: where to save the ground truth generated during dynamic analysis");
    parser.add_argument("--search_parameter", "-sp", "--search-parameter")
        .default_value(std::string("default"))
        .help("search parameters (JSON) for dynamic analysis; \"default\" uses index defaults");
    parser.add_argument("--topk", "-k")
        .default_value(int64_t{100})
        .help("number of neighbors evaluated for each query")
        .scan<'i', int64_t>();
    parser.add_argument("--compact")
        .default_value(false)
        .implicit_value(true)
        .help("print the report on a single line");
}

void
flush_stdout() {
    std::cout.flush();
    std::fflush(stdout);
}

#if defined(__ELF__)
// libvsag logs its start-up banner to stdout while it is loaded, before main() runs. Functions in
// .preinit_array run before the initializers of every shared library, so stdout is pointed at
// stderr until main() takes it back.
int deferred_stdout = -1;

void
defer_stdout(int /*argc*/, char** /*argv*/, char** /*envp*/) {
    deferred_stdout = dup(STDOUT_FILENO);
    if (deferred_stdout >= 0) {
        dup2(STDERR_FILENO, STDOUT_FILENO);
    }
}

__attribute__((section(".preinit_array"),
               used)) void (*const DEFER_STDOUT)(int, char**, char**) = &defer_stdout;
#endif

void
restore_deferred_stdout() {
#if defined(__ELF__)
    if (deferred_stdout >= 0) {
        flush_stdout();
        dup2(deferred_stdout, STDOUT_FILENO);
        close(deferred_stdout);
        deferred_stdout = -1;
    }
#endif
}

// Native code writes diagnostics to stdout through the default logger; send every write made
// while analyzing to stderr so that stdout carries only the JSON report.
class StdoutToStderr {
public:
    StdoutToStderr() : saved_stdout_(dup(STDOUT_FILENO)) {
        if (saved_stdout_ >= 0) {
            flush_stdout();
            dup2(STDERR_FILENO, STDOUT_FILENO);
        }
    }

    StdoutToStderr(const StdoutToStderr&) = delete;
    StdoutToStderr&
    operator=(const StdoutToStderr&) = delete;

    ~StdoutToStderr() {
        if (saved_stdout_ >= 0) {
            flush_stdout();
            dup2(saved_stdout_, STDOUT_FILENO);
            close(saved_stdout_);
        }
    }

private:
    int saved_stdout_;
};

}  // namespace

int
main(int argc, char** argv) {
    restore_deferred_stdout();
    argparse::ArgumentParser parser("analyze_index", vsag::version());
    add_arguments(parser);
    vsag::AnalyzeIndexOptions options;
    bool compact = false;
    try {
        parser.parse_args(argc, argv);
        options.index_path = parser.get<std::string>("--index_path");
        options.build_parameter = parser.get<std::string>("--build_parameter");
        options.query_path = parser.get<std::string>("--query_path");
        if (options.query_path == LEGACY_EMPTY_QUERY_PATH) {
            options.query_path.clear();
        }
        options.query_data_type = parser.get<std::string>("--query_data_type");
        options.base_path = parser.get<std::string>("--base_path");
        options.groundtruth_path = parser.get<std::string>("--groundtruth_path");
        options.save_groundtruth_path = parser.get<std::string>("--save_groundtruth_path");
        options.search_parameter = parser.get<std::string>("--search_parameter");
        options.topk = parser.get<int64_t>("--topk");
        compact = parser.get<bool>("--compact");
    } catch (const std::exception& error) {
        std::cerr << "analyze_index: error: " << error.what() << "\n\n" << parser;
        return EXIT_USAGE;
    }

    tl::expected<std::string, vsag::Error> report;
    {
        StdoutToStderr redirect;
        report = vsag::analyze_index(options);
    }
    if (not report.has_value()) {
        std::cerr << "analyze_index: error [" << error_code(report.error().type)
                  << "]: " << report.error().message << std::endl;
        return EXIT_ANALYSIS_FAILED;
    }
    std::cout << nlohmann::ordered_json::parse(report.value()).dump(compact ? -1 : 2) << std::endl;
    return 0;
}
