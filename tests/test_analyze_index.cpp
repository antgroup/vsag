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

#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "analyzer/index_analyzer.h"
#include "functest.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"

namespace {

constexpr const char* kIndexParam = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 4,
    "index_param": {
        "base_quantization_type": "fp32",
        "precise_quantization_type": "fp32"
    }
})";

constexpr const char* kCosineIndexParam = R"({
    "type": "brute_force",
    "dtype": "float32",
    "metric_type": "cosine",
    "dim": 4,
    "index_param": {
        "base_quantization_type": "fp32",
        "precise_quantization_type": "fp32"
    }
})";

constexpr const char* kWarpIndexParam = R"({
    "type": "warp",
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 4,
    "index_param": {
        "base_quantization_type": "fp32"
    }
})";

constexpr const char* kSindiIndexParam = R"({
    "dtype": "sparse",
    "metric_type": "ip",
    "dim": 128,
    "index_param": {
        "term_id_limit": 1000,
        "window_size": 10000,
        "doc_prune_ratio": 0.0,
        "use_quantization": false,
        "use_reorder": false
    }
})";

constexpr const char* kSindiV2IndexParam = R"({
    "dtype": "sparse",
    "metric_type": "ip",
    "dim": 128,
    "index_param": {
        "term_id_limit": 1000,
        "window_size": 10000,
        "doc_prune_ratio": 0.0,
        "use_quantization": false,
        "use_reorder": false,
        "term_io": {"type": "memory_io"}
    }
})";

constexpr const char* kPyramidHalfIndexParam = R"({
    "dtype": "float16",
    "metric_type": "l2",
    "dim": 4,
    "extra_info_size": 8,
    "index_param": {
        "max_degree": 16,
        "ef_construction": 16,
        "build_thread_count": 1,
        "alpha": 1.2,
        "graph_type": "nsw",
        "root_graph_type": "multi_layer",
        "no_build_levels": [1, 2],
        "base_quantization_type": "fp16",
        "precise_quantization_type": "fp32",
        "use_reorder": false,
        "index_min_size": 8,
        "persist_source_id": true,
        "store_paths": true
    }
})";

constexpr int64_t kDim = 4;
constexpr int64_t kNumElements = 8;

std::vector<float>
MakeVectors() {
    std::vector<float> vectors(kDim * kNumElements);
    for (int64_t i = 0; i < kNumElements; ++i) {
        vectors[i * kDim] = static_cast<float>(i) * 0.1F;
        vectors[i * kDim + 1] = static_cast<float>(i) * 0.2F;
        vectors[i * kDim + 2] = 0.5F - static_cast<float>(i) * 0.01F;
        vectors[i * kDim + 3] = 1.0F;
    }
    return vectors;
}

std::string
BuildAndSaveIndex(const std::string& index_path, const std::string& index_param = kIndexParam) {
    auto index = vsag::Factory::CreateIndex("brute_force", index_param);
    REQUIRE(index.has_value());

    auto vectors = MakeVectors();
    std::vector<int64_t> ids(kNumElements);
    for (int64_t i = 0; i < kNumElements; ++i) {
        ids[i] = i;
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(kNumElements)->Dim(kDim)->Float32Vectors(vectors.data())->Ids(ids.data());
    REQUIRE(index.value()->Build(base).has_value());

    std::ofstream stream(index_path, std::ios::binary | std::ios::out);
    REQUIRE(stream.is_open());
    REQUIRE(index.value()->Serialize(stream).has_value());
    stream.close();
    return index_path;
}

std::string
SaveEmptyIndex(const std::string& index_path,
               const std::string& index_name,
               const std::string& index_param) {
    auto index = vsag::Factory::CreateIndex(index_name, index_param);
    REQUIRE(index.has_value());

    std::ofstream stream(index_path, std::ios::binary | std::ios::out);
    REQUIRE(stream.is_open());
    REQUIRE(index.value()->Serialize(stream).has_value());
    stream.close();
    return index_path;
}

std::string
ReadBinaryFile(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::in);
    REQUIRE(stream.is_open());
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

void
UpdateBasicInfoInFooter(const std::string& path,
                        const std::function<void(vsag::JsonType&)>& update) {
    const auto bytes = ReadBinaryFile(path);
    std::stringstream input(bytes);
    vsag::IOStreamReader reader(input);
    auto footer = vsag::Footer::Parse(reader);
    REQUIRE(footer != nullptr);
    REQUIRE(bytes.size() >= footer->Length());

    auto metadata = footer->GetMetadata();
    auto basic_info = metadata->Get("basic_info");
    REQUIRE(basic_info.IsObject());
    update(basic_info);
    metadata->Set("basic_info", basic_info);

    std::ofstream output(path, std::ios::binary | std::ios::out | std::ios::trunc);
    REQUIRE(output.is_open());
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size() - footer->Length()));
    vsag::IOStreamWriter writer(output);
    footer->Write(writer);
    output.close();
    REQUIRE(output.good());
}

void
RemoveMetricFromFooter(const std::string& path) {
    UpdateBasicInfoInFooter(path, [](vsag::JsonType& basic_info) {
        REQUIRE(basic_info.Contains("metric"));
        basic_info.Erase("metric");
    });
}

void
AddUnknownKeyToFooter(const std::string& path) {
    UpdateBasicInfoInFooter(path, [](vsag::JsonType& basic_info) {
        REQUIRE(basic_info.Contains("index_param"));
        auto index_param = vsag::JsonType::Parse(basic_info["index_param"].GetString());
        index_param["unknown_future_key"].SetBool(true);
        basic_info["index_param"].SetString(index_param.Dump());
    });
}

void
SetInvalidMetricInFooter(const std::string& path) {
    UpdateBasicInfoInFooter(path, [](vsag::JsonType& basic_info) {
        REQUIRE(basic_info.Contains("metric"));
        basic_info["metric"].SetInt(999);
    });
}

void
WriteDenseQueryFile(const std::string& query_path, int64_t rows, int64_t cols) {
    std::ofstream stream(query_path, std::ios::binary | std::ios::out);
    REQUIRE(stream.is_open());
    const auto uint32_rows = static_cast<uint32_t>(rows);
    const auto uint32_cols = static_cast<uint32_t>(cols);
    stream.write(reinterpret_cast<const char*>(&uint32_rows), sizeof(uint32_rows));
    stream.write(reinterpret_cast<const char*>(&uint32_cols), sizeof(uint32_cols));
    std::vector<float> queries(static_cast<uint64_t>(rows) * cols, 0.25F);
    stream.write(reinterpret_cast<const char*>(queries.data()),
                 static_cast<std::streamsize>(queries.size() * sizeof(float)));
    stream.close();
}

void
WriteSparseQueryFile(const std::string& query_path, int64_t rows) {
    std::ofstream stream(query_path, std::ios::binary | std::ios::out);
    REQUIRE(stream.is_open());
    const std::vector<int64_t> header{rows, 128, rows};
    std::vector<int64_t> indptr(static_cast<uint64_t>(rows) + 1);
    std::vector<int32_t> indices(static_cast<uint64_t>(rows));
    const std::vector<float> values(static_cast<uint64_t>(rows), 1.0F);
    for (int64_t row = 0; row < rows; ++row) {
        indptr[static_cast<uint64_t>(row)] = row;
        indices[static_cast<uint64_t>(row)] = static_cast<int32_t>(row);
    }
    indptr[static_cast<uint64_t>(rows)] = rows;
    stream.write(reinterpret_cast<const char*>(header.data()),
                 static_cast<std::streamsize>(header.size() * sizeof(int64_t)));
    stream.write(reinterpret_cast<const char*>(indptr.data()),
                 static_cast<std::streamsize>(indptr.size() * sizeof(int64_t)));
    stream.write(reinterpret_cast<const char*>(indices.data()),
                 static_cast<std::streamsize>(indices.size() * sizeof(int32_t)));
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    stream.close();
}

}  // namespace

TEST_CASE("AnalyzeIndex returns a structured static report", "[analyze_index]") {
    fixtures::TempDir temp_dir("analyze_index_static");
    const std::string index_path = BuildAndSaveIndex(temp_dir.path + "index.vsag");

    vsag::IndexAnalysisOptions options;
    options.index_path = index_path;

    auto report = vsag::JsonType::Parse(vsag::AnalyzeIndex(options).Dump());

    REQUIRE(report["schema_version"].GetString() == vsag::INDEX_ANALYSIS_SCHEMA_VERSION);
    REQUIRE(report["index"]["name"].GetString() == "brute_force");
    REQUIRE(report["index"]["dim"].GetInt() == kDim);
    REQUIRE(report["index"]["data_type"].GetString() == "float32");
    REQUIRE(report["index"]["metric"].GetString() == "l2");
    REQUIRE(report["static_analysis"]["supported"].GetBool());
    REQUIRE(report["static_analysis"]["statistics"].IsObject());
    REQUIRE(report.Contains("dynamic_analysis"));
    REQUIRE(report["warnings"].IsArray());
    REQUIRE(report["compatibility"]["schema_version"].GetString() ==
            vsag::INDEX_ANALYSIS_SCHEMA_VERSION);
}

TEST_CASE("AnalyzeIndex loads legacy brute-force metadata without metric", "[analyze_index]") {
    fixtures::TempDir temp_dir("analyze_index_legacy_metric");
    const std::string index_path = BuildAndSaveIndex(temp_dir.path + "index.vsag");
    RemoveMetricFromFooter(index_path);

    vsag::IndexAnalysisOptions options;
    options.index_path = index_path;

    auto report = vsag::JsonType::Parse(vsag::AnalyzeIndex(options).Dump());

    REQUIRE(report["index"]["metric"].GetString() == "l2");
    REQUIRE_FALSE(report["index"]["metadata_complete"].GetBool());
    REQUIRE(report["warnings"].Dump().find(
                "serialized index metadata is missing 'metric'; using analyzer default 'l2'") !=
            std::string::npos);
}

TEST_CASE("AnalyzeIndex tolerates unknown serialized brute-force parameters", "[analyze_index]") {
    fixtures::TempDir temp_dir("analyze_index_unknown_serialized_param");
    const std::string index_path = BuildAndSaveIndex(temp_dir.path + "index.vsag");
    AddUnknownKeyToFooter(index_path);

    vsag::IndexAnalysisOptions options;
    options.index_path = index_path;

    auto report = vsag::JsonType::Parse(vsag::AnalyzeIndex(options).Dump());

    REQUIRE(report["index"]["name"].GetString() == "brute_force");
    REQUIRE(report["static_analysis"]["supported"].GetBool());
    REQUIRE(report["static_analysis"]["statistics"].IsObject());
    REQUIRE(report["warnings"].Dump().find(
                "ignored serialized BruteForce parameter 'unknown_future_key'") !=
            std::string::npos);
}

TEST_CASE("AnalyzeIndex wraps invalid serialized metric errors", "[analyze_index]") {
    fixtures::TempDir temp_dir("analyze_index_invalid_serialized_metric");
    const std::string index_path = BuildAndSaveIndex(temp_dir.path + "index.vsag");
    SetInvalidMetricInFooter(index_path);

    vsag::IndexAnalysisOptions options;
    options.index_path = index_path;

    const std::string expected_context =
        "failed to load index '" + index_path + "' from serialized metadata";
    REQUIRE_THROWS_WITH(vsag::AnalyzeIndex(options),
                        Catch::Matchers::ContainsSubstring(expected_context));
}

TEST_CASE("AnalyzeIndex does not modify brute-force backing files", "[analyze_index]") {
    fixtures::TempDir temp_dir("analyze_index_isolated_io");
    const std::string backing_path = temp_dir.path + "base-codes.bin";

    vsag::JsonType build_param = vsag::JsonType::Parse(kIndexParam);
    build_param["index_param"]["base_io_type"].SetString("buffer_io");
    build_param["index_param"]["base_file_path"].SetString(backing_path);
    const std::string index_path =
        BuildAndSaveIndex(temp_dir.path + "index.vsag", build_param.Dump());
    const std::string backing_before = ReadBinaryFile(backing_path);
    REQUIRE_FALSE(backing_before.empty());

    vsag::IndexAnalysisOptions options;
    options.index_path = index_path;
    REQUIRE_NOTHROW(vsag::AnalyzeIndex(options));

    REQUIRE(ReadBinaryFile(backing_path) == backing_before);
}

TEST_CASE("Brute-force public parameters reject internal reload bypass", "[analyze_index]") {
    const std::string int8_param = R"({
        "dtype": "int8",
        "metric_type": "l2",
        "dim": 4,
        "index_param": {
            "_analyzer_deserialize": true,
            "base_codes": {
                "codes_type": "flatten_codes",
                "io_params": {"type": "memory_io"},
                "quantization_params": {"type": "int8"}
            }
        }
    })";
    REQUIRE_FALSE(vsag::Factory::CreateIndex("brute_force", int8_param).has_value());

    vsag::JsonType unknown_key_param = vsag::JsonType::Parse(int8_param);
    unknown_key_param["dtype"].SetString("float32");
    unknown_key_param["index_param"]["unknown_key"].SetBool(true);
    REQUIRE_FALSE(vsag::Factory::CreateIndex("brute_force", unknown_key_param.Dump()).has_value());

    unknown_key_param["index_param"].Erase("_analyzer_deserialize");
    unknown_key_param["index_param"].Erase("unknown_key");
    REQUIRE_FALSE(vsag::Factory::CreateIndex("brute_force", unknown_key_param.Dump()).has_value());

    vsag::JsonType unsupported_io_param = vsag::JsonType::Parse(int8_param);
    unsupported_io_param["dtype"].SetString("float32");
    unsupported_io_param["index_param"]["base_codes"]["io_params"]["type"].SetString("unsupported");
    REQUIRE_FALSE(
        vsag::Factory::CreateIndex("brute_force", unsupported_io_param.Dump()).has_value());
}

TEST_CASE("AnalyzeIndex reports the metric from explicit build parameters", "[analyze_index]") {
    fixtures::TempDir temp_dir("analyze_index_explicit_metric");
    const std::string index_path =
        BuildAndSaveIndex(temp_dir.path + "index.vsag", kCosineIndexParam);

    vsag::IndexAnalysisOptions options;
    options.index_path = index_path;
    options.build_parameter = kCosineIndexParam;

    auto report = vsag::JsonType::Parse(vsag::AnalyzeIndex(options).Dump());

    REQUIRE(report["index"]["name"].GetString() == "brute_force");
    REQUIRE(report["index"]["metric"].GetString() == "cosine");
    REQUIRE(report["index"]["metadata_source"].GetString() == "build_parameter");
}

TEST_CASE("AnalyzeIndex backfills Pyramid report metadata from explicit build parameters",
          "[analyze_index]") {
    fixtures::TempDir temp_dir("analyze_index_pyramid_explicit_params");
    const std::string index_path =
        SaveEmptyIndex(temp_dir.path + "index.vsag", "pyramid", kPyramidHalfIndexParam);

    vsag::IndexAnalysisOptions options;
    options.index_path = index_path;
    options.build_parameter = kPyramidHalfIndexParam;

    auto report = vsag::JsonType::Parse(vsag::AnalyzeIndex(options).Dump());

    REQUIRE(report["index"]["name"].GetString() == "pyramid");
    REQUIRE(report["index"]["metadata_source"].GetString() == "build_parameter");
    // The Pyramid footer carries neither data_type nor extra_info_size, so the report must
    // reflect the explicit build parameters instead of the float32 / 0 analyzer defaults.
    REQUIRE(report["index"]["data_type"].GetString() == "float16");
    REQUIRE(report["index"]["extra_info_size"].GetInt() == 8);
    REQUIRE(report["index"]["dim"].GetInt() == kDim);
    REQUIRE(report["index"]["metric"].GetString() == "l2");
}

TEST_CASE("AnalyzeIndex restores cosine for raw-vector brute-force reload", "[analyze_index]") {
    fixtures::TempDir temp_dir("analyze_index_cosine_raw_vector");
    vsag::JsonType build_param = vsag::JsonType::Parse(kCosineIndexParam);
    build_param["index_param"]["store_raw_vector"].SetBool(true);
    const std::string index_path =
        BuildAndSaveIndex(temp_dir.path + "index.vsag", build_param.Dump());
    const std::string query_path = temp_dir.path + "queries.bin";
    WriteDenseQueryFile(query_path, 1, kDim);

    vsag::IndexAnalysisOptions options;
    options.index_path = index_path;
    options.query_path = query_path;
    options.topk = 2;

    auto report = vsag::JsonType::Parse(vsag::AnalyzeIndex(options).Dump());

    REQUIRE(report["index"]["metric"].GetString() == "cosine");
    REQUIRE(report["dynamic_analysis"]["supported"].GetBool());
    REQUIRE(report["dynamic_analysis"]["results"].IsObject());
}

TEST_CASE("AnalyzeIndex does not inject precise IO parameters into WARP", "[analyze_index]") {
    fixtures::TempDir temp_dir("analyze_index_warp_external_param");
    const std::string index_path =
        SaveEmptyIndex(temp_dir.path + "index.vsag", "warp", kWarpIndexParam);

    vsag::IndexAnalysisOptions options;
    options.index_path = index_path;
    options.build_parameter = kWarpIndexParam;

    auto report = vsag::JsonType::Parse(vsag::AnalyzeIndex(options).Dump());

    REQUIRE(report["index"]["name"].GetString() == "warp");
    REQUIRE(report["static_analysis"]["supported"].GetBool());
}

TEST_CASE("AnalyzeIndex loads an empty SINDI with explicit parameters", "[analyze_index]") {
    fixtures::TempDir temp_dir("analyze_index_empty_sindi");
    const std::string index_path =
        SaveEmptyIndex(temp_dir.path + "index.vsag", "sindi", kSindiIndexParam);

    vsag::IndexAnalysisOptions options;
    options.index_path = index_path;
    options.build_parameter = kSindiIndexParam;

    auto report = vsag::JsonType::Parse(vsag::AnalyzeIndex(options).Dump());

    REQUIRE(report["index"]["name"].GetString() == "sindi");
    REQUIRE(report["index"]["data_type"].GetString() == "sparse");
    REQUIRE(report["static_analysis"]["supported"].GetBool());
    REQUIRE(report["static_analysis"]["statistics"]["total_count"].GetInt() == 0);
}

TEST_CASE("AnalyzeIndex keeps SINDI V2 base-path static analysis on GetStats", "[analyze_index]") {
    fixtures::TempDir temp_dir("analyze_index_sindi_v2_base_path");
    const std::string index_path =
        SaveEmptyIndex(temp_dir.path + "index.vsag", "sindi_v2", kSindiV2IndexParam);

    vsag::IndexAnalysisOptions options;
    options.index_path = index_path;
    options.build_parameter = kSindiV2IndexParam;
    options.base_path = temp_dir.path + "unused-base.bin";

    auto report = vsag::JsonType::Parse(vsag::AnalyzeIndex(options).Dump());

    REQUIRE(report["index"]["name"].GetString() == "sindi_v2");
    REQUIRE(report["static_analysis"]["supported"].GetBool());
    REQUIRE(report["static_analysis"]["statistics"]["_analysis"]["analysis_type"].GetString() ==
            "stats");
    REQUIRE(report["warnings"].Dump().find(
                "base_path is ignored for static-only SINDI_V2 analysis") != std::string::npos);
}

TEST_CASE("AnalyzeIndex runs dynamic analysis when a query file is provided", "[analyze_index]") {
    fixtures::TempDir temp_dir("analyze_index_dynamic");
    const std::string index_path = BuildAndSaveIndex(temp_dir.path + "index.vsag");
    const std::string query_path = temp_dir.path + "queries.bin";
    WriteDenseQueryFile(query_path, 1, kDim);

    vsag::IndexAnalysisOptions options;
    options.index_path = index_path;
    options.query_path = query_path;
    options.query_data_type = "dense";
    options.topk = 2;

    auto report = vsag::JsonType::Parse(vsag::AnalyzeIndex(options).Dump());

    REQUIRE(report["dynamic_analysis"]["supported"].GetBool());
    REQUIRE(report["dynamic_analysis"]["topk"].GetInt() == 2);
    REQUIRE(report["dynamic_analysis"]["query_data_type"].GetString() == "dense");
    REQUIRE(report["dynamic_analysis"]["results"].IsObject());
}

TEST_CASE("AnalyzeIndex rejects multi-row dense query files", "[analyze_index]") {
    fixtures::TempDir temp_dir("analyze_index_multi_row_query");
    const std::string index_path = BuildAndSaveIndex(temp_dir.path + "index.vsag");
    const std::string query_path = temp_dir.path + "queries.bin";
    WriteDenseQueryFile(query_path, 2, kDim);

    vsag::IndexAnalysisOptions options;
    options.index_path = index_path;
    options.query_path = query_path;
    options.query_data_type = "dense";

    REQUIRE_THROWS_WITH(
        vsag::AnalyzeIndex(options),
        Catch::Matchers::ContainsSubstring("must contain exactly one query vector (rows=1)"));
}

TEST_CASE("AnalyzeIndex rejects multi-row sparse query files", "[analyze_index]") {
    fixtures::TempDir temp_dir("analyze_index_multi_row_sparse_query");
    const std::string index_path =
        SaveEmptyIndex(temp_dir.path + "index.vsag", "sindi", kSindiIndexParam);
    const std::string query_path = temp_dir.path + "queries.bin";
    WriteSparseQueryFile(query_path, 2);

    vsag::IndexAnalysisOptions options;
    options.index_path = index_path;
    options.build_parameter = kSindiIndexParam;
    options.query_path = query_path;
    options.query_data_type = "sparse";
    options.topk = 2;

    REQUIRE_THROWS_WITH(
        vsag::AnalyzeIndex(options),
        Catch::Matchers::ContainsSubstring("must contain exactly one query row (nrow=1)"));
}

TEST_CASE("AnalyzeIndex explicitly skips search for an empty SINDI", "[analyze_index]") {
    fixtures::TempDir temp_dir("analyze_index_empty_sindi_search");
    const std::string index_path =
        SaveEmptyIndex(temp_dir.path + "index.vsag", "sindi", kSindiIndexParam);
    const std::string query_path = temp_dir.path + "queries.bin";
    WriteSparseQueryFile(query_path, 1);

    vsag::IndexAnalysisOptions options;
    options.index_path = index_path;
    options.build_parameter = kSindiIndexParam;
    options.query_path = query_path;
    options.query_data_type = "sparse";
    options.topk = 2;

    auto report = vsag::JsonType::Parse(vsag::AnalyzeIndex(options).Dump());

    REQUIRE(report["dynamic_analysis"]["supported"].GetBool());
    REQUIRE(report["dynamic_analysis"]["results"]["_analysis"]["status"].GetString() ==
            "not_applicable");
    REQUIRE(report["dynamic_analysis"]["results"]["_analysis"]["skipped"]["search"].GetString() ==
            "index is empty");
}

TEST_CASE("AnalyzeIndex rejects invalid options", "[analyze_index]") {
    fixtures::TempDir temp_dir("analyze_index_invalid");
    const std::string index_path = BuildAndSaveIndex(temp_dir.path + "index.vsag");

    vsag::IndexAnalysisOptions options;
    options.index_path = "";
    REQUIRE_THROWS_AS(vsag::AnalyzeIndex(options), std::invalid_argument);

    options.index_path = index_path;
    options.topk = 0;
    REQUIRE_THROWS_AS(vsag::AnalyzeIndex(options), std::invalid_argument);

    options.topk = 10;
    options.query_data_type = "fp64";
    REQUIRE_THROWS_AS(vsag::AnalyzeIndex(options), std::invalid_argument);

    options.query_data_type = "auto";
    options.groundtruth_path = temp_dir.path + "groundtruth.bin";
    REQUIRE_THROWS_AS(vsag::AnalyzeIndex(options), std::invalid_argument);

    options.groundtruth_path = "";
    options.build_parameter = "not-json";
    REQUIRE_THROWS_AS(vsag::AnalyzeIndex(options), std::invalid_argument);

    options.build_parameter = "default";
    options.search_parameter = "5";
    REQUIRE_THROWS_AS(vsag::AnalyzeIndex(options), std::invalid_argument);
}

TEST_CASE("AnalyzeIndex reports clear errors for missing and damaged files", "[analyze_index]") {
    fixtures::TempDir temp_dir("analyze_index_errors");

    vsag::IndexAnalysisOptions options;
    options.index_path = temp_dir.path + "missing.vsag";
    REQUIRE_THROWS_AS(vsag::AnalyzeIndex(options), std::runtime_error);

    const std::string damaged_path = temp_dir.path + "damaged.vsag";
    {
        std::ofstream stream(damaged_path, std::ios::binary | std::ios::out);
        REQUIRE(stream.is_open());
        stream << "not a vsag index";
    }
    options.index_path = damaged_path;
    REQUIRE_THROWS_AS(vsag::AnalyzeIndex(options), std::runtime_error);
}
