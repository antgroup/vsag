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

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>

#include "binding.h"
#include "vsag/analyze_index.h"

namespace {

// Stable names of vsag::ErrorType, exposed as pyvsag.IndexAnalysisError.code.
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

// Returns (None, report) on success and (error_code, message) on failure; both payloads are
// UTF-8 bytes. pyvsag.analyze_index() turns this into a dict or an IndexAnalysisError.
py::tuple
analyze_index(const std::string& index_path,
              const std::string& build_parameter,
              const std::string& query_path,
              const std::string& query_data_type,
              const std::string& base_path,
              const std::string& groundtruth_path,
              const std::string& save_groundtruth_path,
              const std::string& search_parameter,
              int64_t topk) {
    vsag::AnalyzeIndexOptions options;
    options.index_path = index_path;
    options.build_parameter = build_parameter;
    options.query_path = query_path;
    options.query_data_type = query_data_type;
    options.base_path = base_path;
    options.groundtruth_path = groundtruth_path;
    options.save_groundtruth_path = save_groundtruth_path;
    options.search_parameter = search_parameter;
    options.topk = topk;

    tl::expected<std::string, vsag::Error> result;
    {
        // Analysis can take long; let other Python threads run meanwhile.
        py::gil_scoped_release release;
        result = vsag::analyze_index(options);
        // The default logger writes to std::cout without flushing. Flush here so that a caller
        // which redirects the stdout file descriptor around this call receives every diagnostic.
        std::cout.flush();
        std::fflush(stdout);
    }
    if (result.has_value()) {
        return py::make_tuple(py::none(), py::bytes(result.value()));
    }
    return py::make_tuple(error_code(result.error().type), py::bytes(result.error().message));
}

}  // namespace

void
bind_analysis(py::module_& module) {
    module.attr("ANALYZE_INDEX_SCHEMA_VERSION") = vsag::ANALYZE_INDEX_SCHEMA_VERSION;
    module.def("_analyze_index",
               &analyze_index,
               py::arg("index_path"),
               py::arg("build_parameter"),
               py::arg("query_path"),
               py::arg("query_data_type"),
               py::arg("base_path"),
               py::arg("groundtruth_path"),
               py::arg("save_groundtruth_path"),
               py::arg("search_parameter"),
               py::arg("topk"),
               "Native entry point of pyvsag.analyze_index(); use that function instead.");
}
