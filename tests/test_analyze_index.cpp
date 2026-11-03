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
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "analyzer/index_analyzer.h"
#include "functest.h"

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
BuildAndSaveIndex(const std::string& index_path) {
    auto index = vsag::Factory::CreateIndex("brute_force", kIndexParam);
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

TEST_CASE("AnalyzeIndex runs dynamic analysis when a query file is provided", "[analyze_index]") {
    fixtures::TempDir temp_dir("analyze_index_dynamic");
    const std::string index_path = BuildAndSaveIndex(temp_dir.path + "index.vsag");
    const std::string query_path = temp_dir.path + "queries.bin";
    WriteDenseQueryFile(query_path, 2, kDim);

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
