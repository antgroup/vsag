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

#include "vsag/errors.h"
#include "vsag/expected.hpp"

namespace vsag {

/// Version of the JSON report returned by analyze_index().
constexpr int64_t ANALYZE_INDEX_SCHEMA_VERSION = 1;

/**
 * @brief Inputs of analyze_index().
 *
 * Only index_path is required. An empty string means "not provided". For compatibility with
 * the analyze_index command-line tool, build_parameter and search_parameter also accept the
 * literal "default".
 */
struct AnalyzeIndexOptions {
    /// Path of a serialized index file.
    std::string index_path;

    /// Build parameters (a JSON object) used to reload the index. Empty or "default" uses the
    /// parameters embedded in the file. Files without embedded parameters (legacy files and
    /// lazy_hgraph files) require them; the index type is taken from "type" or
    /// "index_param.type" when the file does not record it.
    std::string build_parameter;

    /// Query dataset for dynamic analysis. Empty runs static analysis only. Dense files hold
    /// `uint32 rows, uint32 dim, float32 data[rows * dim]`; sparse files hold a CSR matrix
    /// `int64 rows, int64 cols, int64 nnz, int64 indptr[rows + 1], int32 indices[nnz],
    /// float32 values[nnz]`.
    std::string query_path;

    /// Query file layout: "auto" (sparse for sparse indexes, dense otherwise), "dense" or
    /// "sparse".
    std::string query_data_type{"auto"};

    /// SINDI only: sparse CSR base dataset used to compute exact ground truth and the metrics
    /// that need original vectors.
    std::string base_path;

    /// SINDI only: existing ground truth (`uint32 rows, uint32 topk, int32 ids[],
    /// float32 distances[]`) for dynamic analysis.
    std::string groundtruth_path;

    /// SINDI only: output path for the ground truth generated from base_path.
    std::string save_groundtruth_path;

    /// Search parameters (a JSON object) used by dynamic analysis. Empty or "default" uses the
    /// defaults of the index.
    std::string search_parameter;

    /// Number of neighbors evaluated for each query. Must be positive when query_path is set.
    int64_t topk{100};
};

/**
 * @brief Analyze a serialized index file without building or modifying it.
 *
 * Detects the serialization format, reloads the index, reports its metadata and the
 * GetStats() statistics, and runs AnalyzeIndexBySearch() when a query file is given. The
 * whole index is loaded into memory, with memory IO even when the recorded parameters name
 * file-backed IO, so no recorded file path is opened. Failures are returned as an Error; the
 * library code that loads and analyzes the index may also log through the vsag logger.
 *
 * Supported files: footer files written by Serialize(std::ostream&), uncompressed chunked
 * files written by Serialize(SerializeWriter&, ...), and streaming files written by
 * SerializeStreaming(). Files without a footer can be loaded only with build_parameter.
 *
 * @param options what to analyze; see AnalyzeIndexOptions.
 * @return A JSON object string with the members "schema_version" (equal to
 *         ANALYZE_INDEX_SCHEMA_VERSION), "metadata", "static_analysis", "dynamic_analysis"
 *         (null without a query file), "warnings" and "compatibility". On failure, an Error
 *         whose type identifies the cause: INVALID_ARGUMENT for invalid or unsupported option
 *         combinations, MISSING_FILE or READ_ERROR for unreadable inputs, INVALID_BINARY for
 *         damaged or unrecognized files, UNSUPPORTED_INDEX_OPERATION for formats or analyses
 *         that cannot be performed, and the error reported by the index otherwise.
 */
[[nodiscard]] tl::expected<std::string, Error>
analyze_index(const AnalyzeIndexOptions& options);

}  // namespace vsag
