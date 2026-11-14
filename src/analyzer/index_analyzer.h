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

#pragma once

#include <cstdint>
#include <string>

#include "json_types.h"

namespace vsag {

/**
 * @brief Schema version of the JSON document returned by AnalyzeIndex().
 *
 * Consumers must use this value to detect incompatible result layouts.
 */
constexpr const char* INDEX_ANALYSIS_SCHEMA_VERSION = "1.0";

/**
 * @brief Options accepted by the reusable index analysis component.
 *
 * The component performs no command-line parsing and no output formatting. It is shared by the
 * C++ `analyze_index` executable, the C ABI used by `pyvsag`, and the installed
 * `pyvsag-analyze-index` command.
 */
struct IndexAnalysisOptions {
    /// Path to a serialized VSAG index file. Required.
    std::string index_path;

    /// Build parameter JSON used to reload the index. "default" reuses the parameters embedded in
    /// the serialized metadata.
    std::string build_parameter = "default";

    /// Binary query dataset path. Empty (or the legacy "empty" sentinel) performs static analysis
    /// only.
    std::string query_path;

    /// Query dataset type: "auto", "dense" or "sparse".
    std::string query_data_type = "auto";

    /// Optional sparse CSR base dataset used by SINDI analysis and ground-truth generation.
    std::string base_path;

    /// Optional ground-truth file reused by dynamic analysis.
    std::string groundtruth_path;

    /// Optional output path for generated ground truth.
    std::string save_groundtruth_path;

    /// Search parameter JSON used by dynamic analysis (and by SINDI base-path static analysis).
    std::string search_parameter = "default";

    /// Top-K used by dynamic analysis.
    int64_t topk = 100;
};

/**
 * @brief Run static and, when a query path is supplied, dynamic index analysis.
 *
 * @param options Analysis inputs.
 * @return A JSON-serializable object separating schema version, index metadata, static
 * statistics, optional dynamic results, warnings and compatibility information.
 *
 * @throws std::invalid_argument when an option is invalid or unsupported.
 * @throws std::runtime_error when the index or dataset cannot be loaded or analyzed.
 */
JsonType
AnalyzeIndex(const IndexAnalysisOptions& options);

}  // namespace vsag
