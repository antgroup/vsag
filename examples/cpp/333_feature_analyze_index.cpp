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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

int
main(int argc, char** argv) {
    vsag::init();

    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 1000;
    int64_t dim = 64;
    std::vector<int64_t> ids(num_vectors);
    std::vector<float> datas(num_vectors * dim);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        datas[i] = distrib_real(rng);
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(datas.data())
        ->Owner(false);

    /******************* Build and Save an HGraph Index *****************/
    std::string hgraph_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 64,
        "index_param": {
            "base_quantization_type": "sq8",
            "max_degree": 16,
            "ef_construction": 100
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("hgraph", hgraph_build_parameters).value();
    if (not index->Build(base).has_value()) {
        std::cerr << "build failed" << std::endl;
        return 1;
    }
    const std::string index_path = "/tmp/vsag-analyze-index-example.index";
    std::ofstream index_file(index_path, std::ios::binary);
    if (not index->Serialize(index_file).has_value()) {
        std::cerr << "serialize failed" << std::endl;
        return 1;
    }
    index_file.close();

    /******************* Write a Query File: uint32 rows, uint32 dim, float32 data *****************/
    const std::string query_path = "/tmp/vsag-analyze-index-example.queries";
    const uint32_t query_count = 10;
    const auto query_dim = static_cast<uint32_t>(dim);
    std::ofstream query_file(query_path, std::ios::binary);
    query_file.write(reinterpret_cast<const char*>(&query_count), sizeof(query_count));
    query_file.write(reinterpret_cast<const char*>(&query_dim), sizeof(query_dim));
    query_file.write(reinterpret_cast<const char*>(datas.data()),
                     static_cast<std::streamsize>(query_count * query_dim * sizeof(float)));
    query_file.close();

    /******************* Analyze the Index File *****************/
    // The index type and its parameters are read from the file; only index_path is required.
    vsag::AnalyzeIndexOptions options;
    options.index_path = index_path;
    options.query_path = query_path;
    options.search_parameter = R"({"hgraph": {"ef_search": 100}})";
    options.topk = 10;
    auto report = vsag::analyze_index(options);
    if (not report.has_value()) {
        std::cerr << "analysis failed: " << report.error().message << std::endl;
        return 1;
    }
    std::cout << "analysis report: " << report.value() << std::endl;
    return 0;
}
