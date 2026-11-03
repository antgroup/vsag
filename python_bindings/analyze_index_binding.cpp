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
#include <string>

#include "analyzer/index_analyzer.h"
#include "binding.h"

namespace {

std::string
AnalyzeIndexJson(const std::string& index_path,
                 const std::string& build_parameter,
                 const std::string& query_path,
                 const std::string& query_data_type,
                 const std::string& base_path,
                 const std::string& groundtruth_path,
                 const std::string& save_groundtruth_path,
                 const std::string& search_parameter,
                 int64_t topk) {
    vsag::IndexAnalysisOptions options;
    options.index_path = index_path;
    options.build_parameter = build_parameter;
    options.query_path = query_path;
    options.query_data_type = query_data_type;
    options.base_path = base_path;
    options.groundtruth_path = groundtruth_path;
    options.save_groundtruth_path = save_groundtruth_path;
    options.search_parameter = search_parameter;
    options.topk = topk;
    return vsag::AnalyzeIndex(options).Dump();
}

}  // namespace

void
bind_analyze_index(py::module_& module) {
    module.def("analyze_index_json",
               &AnalyzeIndexJson,
               py::arg("index_path"),
               py::arg("build_parameter") = "default",
               py::arg("query_path") = "",
               py::arg("query_data_type") = "auto",
               py::arg("base_path") = "",
               py::arg("groundtruth_path") = "",
               py::arg("save_groundtruth_path") = "",
               py::arg("search_parameter") = "default",
               py::arg("topk") = 100,
               R"pbdoc(
        Analyze a serialized VSAG index and return the report as a JSON string.

        This is the low-level binding used by :func:`pyvsag.analyze_index`. Prefer the
        Python-level wrapper, which returns a dictionary and validates arguments.

        Args:
            index_path (str): Path to a serialized VSAG index file.
            build_parameter (str): Build parameter JSON, or ``"default"`` to reuse the
                parameters embedded in the index file.
            query_path (str): Query dataset path. Empty performs static analysis only.
            query_data_type (str): ``"auto"``, ``"dense"`` or ``"sparse"``.
            base_path (str): Optional sparse CSR base dataset for SINDI analysis.
            groundtruth_path (str): Optional ground-truth file for dynamic analysis.
            save_groundtruth_path (str): Optional output path for generated ground truth.
            search_parameter (str): Search parameter JSON, or ``"default"``.
            topk (int): Top-K used by dynamic analysis.

        Returns:
            str: JSON report containing ``schema_version``, ``index``, ``static_analysis``,
            ``dynamic_analysis``, ``warnings`` and ``compatibility``.

        Raises:
            ValueError: Invalid or unsupported arguments.
            RuntimeError: The index or dataset cannot be loaded or analyzed.
        )pbdoc");
}
