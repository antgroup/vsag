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

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "vsag/options.h"

namespace {

constexpr int64_t K_DIM = 128;
constexpr int64_t K_NUM_VECTORS = 500;
constexpr int64_t K_TOP_K = 10;
constexpr const char* K_STREAMING_INDEX_PATH = "/tmp/vsag-streaming-load-hybrid.index";
constexpr const char* K_PRECISE_FILE_PATH = "/tmp/vsag-streaming-load-hybrid-precise";

const char* hybrid_hgraph_build_parameters = R"(
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "rabitq",
        "precise_quantization_type": "sq8",
        "use_reorder": true,
        "rabitq_bits_per_dim_base": 3,
        "rabitq_bits_per_dim_query": 32,
        "rabitq_error_rate": 1.9,
        "max_degree": 32,
        "ef_construction": 200,
        "graph_storage_type": "compressed"
    }
}
)";

const char* hybrid_hgraph_load_parameters = R"(
{
    "base_io_type": "block_memory_io",
    "precise_io_type": "buffer_io",
    "precise_file_path": "/tmp/vsag-streaming-load-hybrid-precise"
}
)";

const char* hgraph_search_parameters = R"(
{
    "hgraph": {
        "ef_search": 200,
        "rabitq_one_bit_search": true
    }
}
)";

void
cleanup_hybrid_files() {
    std::remove(K_STREAMING_INDEX_PATH);
    std::remove(K_PRECISE_FILE_PATH);
}

vsag::DatasetPtr
make_base_dataset(std::vector<int64_t>& ids, std::vector<float>& vectors) {
    ids.resize(K_NUM_VECTORS);
    vectors.resize(K_NUM_VECTORS * K_DIM);

    std::mt19937 rng(47);
    std::uniform_real_distribution<float> dist;
    for (int64_t i = 0; i < K_NUM_VECTORS; ++i) {
        ids[i] = i;
    }
    for (auto& value : vectors) {
        value = dist(rng);
    }

    return vsag::Dataset::Make()
        ->NumElements(K_NUM_VECTORS)
        ->Dim(K_DIM)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
}

vsag::DatasetPtr
make_query_dataset(std::vector<float>& query_vector) {
    query_vector.resize(K_DIM);
    std::mt19937 rng(101);
    std::uniform_real_distribution<float> dist;
    for (auto& value : query_vector) {
        value = dist(rng);
    }

    return vsag::Dataset::Make()
        ->NumElements(1)
        ->Dim(K_DIM)
        ->Float32Vectors(query_vector.data())
        ->Owner(false);
}

template <typename T>
void
check_result(const tl::expected<T, vsag::Error>& result, const std::string& action) {
    if (!result.has_value()) {
        std::cerr << action << " failed: " << result.error().message << std::endl;
        std::abort();
    }
}

void
print_results(const vsag::DatasetPtr& result) {
    std::cout << "results:" << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << "  " << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }
}

}  // namespace

int
main() {
    vsag::Options::Instance().set_block_size_limit(2UL * 1024 * 1024);
    cleanup_hybrid_files();

    std::vector<int64_t> ids;
    std::vector<float> vectors;
    auto base = make_base_dataset(ids, vectors);

    auto index = vsag::Factory::CreateIndex("hgraph", hybrid_hgraph_build_parameters).value();
    check_result(index->Build(base), "Build");

    {
        std::ofstream out(K_STREAMING_INDEX_PATH, std::ios::binary);
        check_result(index->SerializeStreaming(out), "SerializeStreaming");
    }

    vsag::IndexPtr loaded_index;
    {
        std::ifstream in(K_STREAMING_INDEX_PATH, std::ios::binary);
        auto load_result = vsag::Index::Load(in, hybrid_hgraph_load_parameters);
        check_result(load_result, "Load");
        loaded_index = load_result.value();
    }

    std::vector<float> query_vector;
    auto query = make_query_dataset(query_vector);
    auto result = loaded_index->KnnSearch(query, K_TOP_K, hgraph_search_parameters);
    check_result(result, "KnnSearch");

    std::cout << "After streaming Load(), hybrid HGraph contains: "
              << loaded_index->GetNumElements() << std::endl;
    std::cout << "Streaming index file: " << K_STREAMING_INDEX_PATH << std::endl;
    std::cout << "Disk reorder file: " << K_PRECISE_FILE_PATH << std::endl;
    print_results(result.value());
    return 0;
}
