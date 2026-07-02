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

//
// 325_feature_uring_io.cpp
//
// Demonstrates using io_uring as the disk IO backend for HGraph.
//
// When liburing is available, uring_io uses the Linux io_uring subsystem
// for efficient asynchronous direct IO. When liburing is unavailable,
// uring_io transparently falls back to buffer_io.
//
// The example builds an HGraph index in memory, serializes it to disk,
// then reloads it with uring_io as the base storage backend so that
// graph traversal reads go through io_uring (or its fallback).
//
// Build:
//
//     cmake -B build-release -DVSAG_BUILD_TYPE=Release -DENABLE_EXAMPLES=ON -DENABLE_LIBURING=ON
//     cmake --build build-release --target 325_feature_uring_io -j
//     # binary: build-release/examples/cpp/325_feature_uring_io
//
// Run:
//
//     ./build-release/examples/cpp/325_feature_uring_io
//

#include <stdlib.h>
#include <unistd.h>
#include <vsag/vsag.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int64_t K_NUM_VECTORS = 1000;
constexpr int64_t K_DIM = 64;
constexpr int64_t K_TOP_K = 10;

}  // namespace

int
main() {
    vsag::init();

    auto tmp_dir =
        std::filesystem::temp_directory_path() / ("vsag-uring-" + std::to_string(::getpid()));
    std::filesystem::create_directories(tmp_dir);
    auto index_path = (tmp_dir / "hgraph.index").string();
    auto base_path = (tmp_dir / "hgraph.base").string();

    /******************* Prepare Dataset *****************/
    std::vector<int64_t> ids(K_NUM_VECTORS);
    std::vector<float> vectors(K_NUM_VECTORS * K_DIM);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distrib;
    for (int64_t i = 0; i < K_NUM_VECTORS; ++i) {
        ids[i] = i;
    }
    for (auto& v : vectors) {
        v = distrib(rng);
    }
    auto base = vsag::Dataset::Make()
                    ->NumElements(K_NUM_VECTORS)
                    ->Dim(K_DIM)
                    ->Ids(ids.data())
                    ->Float32Vectors(vectors.data())
                    ->Owner(false);

    /******************* Build HGraph (in-memory) *****************/
    auto build_params = R"(
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
    auto index = vsag::Factory::CreateIndex("hgraph", build_params).value();
    auto build_result = index->Build(base);
    if (not build_result.has_value()) {
        std::cerr << "Build failed: " << build_result.error().message << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Built HGraph with " << index->GetNumElements() << " vectors" << std::endl;

    /******************* Serialize to Disk *****************/
    {
        std::ofstream out(index_path.c_str(), std::ios::binary);
        auto ser = index->SerializeStreaming(out);
        if (not ser.has_value()) {
            std::cerr << "Serialize failed: " << ser.error().message << std::endl;
            return EXIT_FAILURE;
        }
    }
    std::cout << "Serialized to " << index_path.c_str() << std::endl;

    /******************* Load with uring_io Backend *****************/
    // Set base_io_type to "uring_io" so graph/base data is read via io_uring.
    // Optional: set base_direct_read to true for O_DIRECT (bypass page cache).
    auto load_params = std::string(R"(
    {
        "base_io_type": "uring_io",
        "base_file_path": ")") +
                       base_path + R"("
    }
    )";
    vsag::IndexPtr uring_index;
    {
        std::ifstream in(index_path.c_str(), std::ios::binary);
        auto load_result = vsag::Index::Load(in, load_params);
        if (not load_result.has_value()) {
            std::cerr << "Load with uring_io failed: " << load_result.error().message << std::endl;
            return EXIT_FAILURE;
        }
        uring_index = load_result.value();
    }
    std::cout << "Loaded with uring_io backend" << std::endl;

    /******************* Search *****************/
    std::vector<float> query(K_DIM);
    for (auto& v : query) {
        v = distrib(rng);
    }
    auto query_ds = vsag::Dataset::Make()
                        ->NumElements(1)
                        ->Dim(K_DIM)
                        ->Float32Vectors(query.data())
                        ->Owner(false);

    auto search_params = R"(
    {
        "hgraph": {
            "ef_search": 100
        }
    }
    )";
    auto result = uring_index->KnnSearch(query_ds, K_TOP_K, search_params);
    if (not result.has_value()) {
        std::cerr << "Search failed: " << result.error().message << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Top-" << result.value()->GetDim() << " results:" << std::endl;
    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
        std::cout << "  id=" << result.value()->GetIds()[i]
                  << "  dist=" << result.value()->GetDistances()[i] << std::endl;
    }

    std::filesystem::remove_all(tmp_dir);

    return EXIT_SUCCESS;
}
