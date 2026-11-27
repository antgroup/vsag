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
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "analyzer/index_analyzer.h"

namespace {

constexpr const char* DEFAULT_BUILD_PARAM = "default";
constexpr const char* DEFAULT_SEARCH_PARAM = "default";
constexpr const char* EMPTY_QUERY_PATH = "";
constexpr const char* QUERY_DATA_TYPE_AUTO = "auto";
constexpr const char* EMPTY_OPTIONAL_PATH = "";

// The library reads this variable during static initialization, before main can adjust logging.
class InitBannerSuppressor {
public:
    InitBannerSuppressor() {
#if defined(_WIN32)
        suppressed_ = _putenv_s("VSAG_SUPPRESS_INIT_BANNER", "1") == 0;
#else
        suppressed_ = ::setenv("VSAG_SUPPRESS_INIT_BANNER", "1", 1) == 0;
#endif
    }

    [[nodiscard]] bool
    Suppressed() const {
        return suppressed_;
    }

private:
    bool suppressed_{false};
};

// Ordering is guaranteed only by MSVC init_seg, or by GCC/Clang init_priority on ELF targets
// linked with GNU ld/lld. Other toolchains (including Mach-O/Darwin) and the fallback below have
// no such guarantee, so the library banner may still precede the JSON report on those targets.
#if defined(_MSC_VER)
#pragma init_seg(lib)
InitBannerSuppressor init_banner_suppressor;
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__ELF__)
InitBannerSuppressor init_banner_suppressor __attribute__((init_priority(101)));
#else
InitBannerSuppressor init_banner_suppressor;
#endif

void
parse_args(argparse::ArgumentParser& parser, int argc, char** argv) {
    parser.add_argument<std::string>("--index_path", "-i")
        .required()
        .help("The index path for load or save");

    parser.add_argument<std::string>("--build_parameter", "-bp")
        .default_value(DEFAULT_BUILD_PARAM)
        .help(
            "The parameter for build index, "
            "if not set, will use default parameter in index file");

    parser.add_argument<std::string>("--query_path", "-qp")
        .default_value(EMPTY_QUERY_PATH)
        .help("The query dataset path, if not set, will not do query analysis");

    parser.add_argument<std::string>("--query_data_type")
        .default_value(std::string(QUERY_DATA_TYPE_AUTO))
        .help("Query dataset type: auto, dense, sparse");

    parser.add_argument<std::string>("--base_path")
        .default_value(std::string(EMPTY_OPTIONAL_PATH))
        .help("Optional sparse base dataset path for SINDI analyze");

    parser.add_argument<std::string>("--groundtruth_path")
        .default_value(std::string(EMPTY_OPTIONAL_PATH))
        .help("Optional ground truth path for analyze");

    parser.add_argument<std::string>("--save_groundtruth_path")
        .default_value(std::string(EMPTY_OPTIONAL_PATH))
        .help("Optional output path for generated ground truth");

    parser.add_argument<std::string>("--search_parameter", "-sp")
        .default_value(DEFAULT_SEARCH_PARAM)
        .help("The parameter for search, if not set, will use default parameter");

    parser.add_argument("--topk", "-k")
        .default_value(100)
        .help("The topk for search")
        .scan<'i', int>();

    parser.parse_args(argc, argv);
}

}  // namespace

int
main(int argc, char** argv) {
    argparse::ArgumentParser parser("analyze_index");
    try {
        parse_args(parser, argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "analyze_index: " << error.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    if (not init_banner_suppressor.Suppressed()) {
        std::cerr << "analyze_index: failed to suppress the initialization banner" << std::endl;
        return 1;
    }

    // Keep library diagnostics on stderr so stdout only carries the JSON report.
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kWARN);

    vsag::IndexAnalysisOptions options;
    options.index_path = parser.get<std::string>("--index_path");
    options.build_parameter = parser.get<std::string>("--build_parameter");
    options.query_path = parser.get<std::string>("--query_path");
    options.query_data_type = parser.get<std::string>("--query_data_type");
    options.base_path = parser.get<std::string>("--base_path");
    options.groundtruth_path = parser.get<std::string>("--groundtruth_path");
    options.save_groundtruth_path = parser.get<std::string>("--save_groundtruth_path");
    options.search_parameter = parser.get<std::string>("--search_parameter");
    options.topk = parser.get<int>("--topk");

    try {
        auto result = vsag::AnalyzeIndex(options);
        std::cout << result.Dump(4) << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "analyze_index: error: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
