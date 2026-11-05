
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
// 323_index_hgraph_rabitq_split.cpp
//
// End-to-end demo for the RaBitQ "x+y" split code layout on HGraph. The split
// layout is selected by setting both base_quantization_type and
// precise_quantization_type to "rabitq", then choosing:
//   * rabitq_bits_per_dim_base    = x filter bits;
//   * rabitq_bits_per_dim_precise = y supplement bits.
//
// layout stores each base vector as TWO independent storages:
//   * filter storage     - the leading x-bit traversal codes, accessed on
//                          graph hops;
//   * supplement storage - the trailing y-bit reorder codes plus
//                          metadata, accessed only during full-distance /
//                          reorder evaluation.
//
// This split lets you place the two storages on different IO backends. We
// build the index once with the in-memory configuration and then load it
// three times to demonstrate the three deployment patterns:
//
//   Mode 1. Pure in-memory  : both storages on block_memory_io.
//   Mode 2. Pure on-disk    : both storages on async_io (falls back to
//                             buffer_io if libaio is unavailable).
//   Mode 3. Hybrid          : filter codes in memory + supplement on disk.
//                             This is the recommended large-index setup -
//                             x-bit traversal stays hot in RAM while the
//                             heavier y-bit supplement bytes are paged in
//                             on demand during reorder.
//
// Build:
//
//     make release
//     # the binary lands at build-release/examples/cpp/323_index_hgraph_rabitq_split
//
// Run:
//
//     ./build-release/examples/cpp/323_index_hgraph_rabitq_split
//
// All temporary files / saved indexes are placed under /tmp.
//

#include <vsag/vsag.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int64_t kDim = 128;
constexpr int64_t kBaseCount = 5000;
constexpr int64_t kTopK = 10;

// File paths used by the three demos.
constexpr const char* kSerializedIndex = "/tmp/vsag_rabitq_split_demo.index";
constexpr const char* kDiskBaseFilePath = "/tmp/vsag_rabitq_split_disk";
constexpr const char* kHybridBaseFilePath = "/tmp/vsag_rabitq_split_hybrid";

vsag::DatasetPtr
generate_base_dataset(std::vector<int64_t>& ids, std::vector<float>& vectors) {
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    ids.resize(kBaseCount);
    vectors.resize(kDim * kBaseCount);
    for (int64_t i = 0; i < kBaseCount; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < kDim * kBaseCount; ++i) {
        vectors[i] = distrib_real(rng);
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(kBaseCount)
        ->Dim(kDim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    return base;
}

vsag::DatasetPtr
generate_query_dataset(std::vector<float>& query_vector) {
    std::mt19937 rng(101);
    std::uniform_real_distribution<float> distrib_real;
    query_vector.resize(kDim);
    for (int64_t i = 0; i < kDim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(kDim)->Float32Vectors(query_vector.data())->Owner(false);
    return query;
}

// Build the index once in memory and serialize it. Later modes load this
// same byte stream into a freshly-created index whose JSON differs only by
// IO-type / file-path knobs. The split RaBitQ wire format is identical
// across the three IO combinations covered here, so a single saved file is
// enough.
void
build_and_save(vsag::Engine& engine, const vsag::DatasetPtr& base) {
    const std::string build_params = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "rabitq",
            "precise_quantization_type": "rabitq",
            "base_io_type": "block_memory_io",
            "use_reorder": true,
            "rabitq_bits_per_dim_base": 3,
            "rabitq_bits_per_dim_precise": 5,
            "rabitq_error_rate": 1.9,
            "max_degree": 32,
            "ef_construction": 200,
            "graph_storage_type": "compressed"
        }
    }
    )";
    auto index = engine.CreateIndex("hgraph", build_params).value();

    if (auto built = index->Build(base); not built.has_value()) {
        std::cerr << "build failed: " << built.error().message << std::endl;
        std::exit(1);
    }
    std::cout << "[build] indexed " << index->GetNumElements() << " vectors" << std::endl;

    std::ofstream out(kSerializedIndex, std::ios::binary);
    if (auto ok = index->Serialize(out); not ok.has_value()) {
        std::cerr << "serialize failed: " << ok.error().message << std::endl;
        std::exit(1);
    }
    std::cout << "[build] saved to " << kSerializedIndex << std::endl;
}

vsag::IndexPtr
load_index(vsag::Engine& engine, const std::string& index_params) {
    auto index = engine.CreateIndex("hgraph", index_params).value();
    std::ifstream in(kSerializedIndex, std::ios::binary);
    if (auto ok = index->Deserialize(in); not ok.has_value()) {
        std::cerr << "deserialize failed: " << ok.error().message << std::endl;
        std::exit(1);
    }
    return index;
}

void
run_search(const std::string& tag, const vsag::IndexPtr& index, const vsag::DatasetPtr& query) {
    const std::string search_params = R"(
    {
        "hgraph": {
            "ef_search": 200,
            "rabitq_one_bit_search": true
        }
    }
    )";
    auto result = index->KnnSearch(query, kTopK, search_params).value();
    std::cout << "[" << tag << "] top-" << kTopK << " results:" << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << "    rank " << i << ": id=" << result->GetIds()[i]
                  << ", dist=" << result->GetDistances()[i] << std::endl;
    }
}

// Mode 1: both storages live in RAM. This is the default and matches the
// behaviour you get when no file_path / IO override is supplied.
void
demo_memory(vsag::Engine& engine, const vsag::DatasetPtr& query) {
    const std::string params = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "rabitq",
            "precise_quantization_type": "rabitq",
            "base_io_type": "block_memory_io",
            "use_reorder": true,
            "rabitq_bits_per_dim_base": 3,
            "rabitq_bits_per_dim_precise": 5,
            "rabitq_error_rate": 1.9,
            "max_degree": 32,
            "ef_construction": 200,
            "graph_storage_type": "compressed"
        }
    }
    )";
    auto index = load_index(engine, params);
    run_search("mode1-memory", index, query);
}

// Mode 2: both storages backed by async_io. The library will append
// "_onebit" / "_supplement" suffixes to base_file_path internally so the
// two storages never collide on disk. async_io transparently falls back to
// buffer_io when libaio is unavailable.
void
demo_disk(vsag::Engine& engine, const vsag::DatasetPtr& query) {
    const std::string params = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "rabitq",
            "precise_quantization_type": "rabitq",
            "base_io_type": "async_io",
            "base_file_path": "/tmp/vsag_rabitq_split_disk",
            "use_reorder": true,
            "rabitq_bits_per_dim_base": 3,
            "rabitq_bits_per_dim_precise": 5,
            "rabitq_error_rate": 1.9,
            "max_degree": 32,
            "ef_construction": 200,
            "graph_storage_type": "compressed"
        }
    }
    )";
    auto index = load_index(engine, params);
    run_search("mode2-disk", index, query);
}

// Mode 3: hybrid - x-bit filter codes stay in block_memory_io while
// the colder y-bit supplement codes are pushed to async_io. The hgraph
// entry to flip the supplement-only backend is "base_supplement_io_type".
// Only the (block_memory_io, async_io) combination is currently allowed.
// The supplement file path is automatically derived by appending the
// "_supplement" suffix to the base file path, so a single
// "base_file_path" knob is enough to lay out both backing files.
void
demo_hybrid(vsag::Engine& engine, const vsag::DatasetPtr& query) {
    const std::string params = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "rabitq",
            "precise_quantization_type": "rabitq",
            "base_io_type": "block_memory_io",
            "base_supplement_io_type": "async_io",
            "base_file_path": "/tmp/vsag_rabitq_split_hybrid",
            "use_reorder": true,
            "rabitq_bits_per_dim_base": 3,
            "rabitq_bits_per_dim_precise": 5,
            "rabitq_error_rate": 1.9,
            "max_degree": 32,
            "ef_construction": 200,
            "graph_storage_type": "compressed"
        }
    }
    )";
    auto index = load_index(engine, params);
    run_search("mode3-hybrid", index, query);
}

}  // namespace

int
main(int /*argc*/, char** /*argv*/) {
    vsag::init();
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);

    std::vector<int64_t> ids;
    std::vector<float> base_vectors;
    auto base = generate_base_dataset(ids, base_vectors);

    std::vector<float> query_vector;
    auto query = generate_query_dataset(query_vector);

    build_and_save(engine, base);

    demo_memory(engine, query);
    demo_disk(engine, query);
    demo_hybrid(engine, query);

    engine.Shutdown();
    return 0;
}
