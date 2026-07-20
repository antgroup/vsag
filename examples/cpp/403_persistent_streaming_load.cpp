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

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "vsag/options.h"

namespace {

enum class ExampleIndex {
    K_BRUTE_FORCE,
    K_H_GRAPH,
    K_IVF,
    K_PYRAMID,
    K_SINDI,
};

constexpr ExampleIndex K_INDEX_KIND = ExampleIndex::K_H_GRAPH;
constexpr int64_t K_DENSE_DIM = 16;
constexpr int64_t K_SPARSE_DIM = 128;
constexpr int64_t K_NUM_VECTORS = 100;
constexpr int64_t K_TOP_K = 5;

struct ExampleConfig {
    const char* index_name;
    const char* build_parameters;
    const char* load_parameters;
    const char* search_parameters;
    const char* file_path;
    int64_t dim;
    bool use_paths;
    bool use_sparse;
};

ExampleConfig
get_example_config() {
    switch (K_INDEX_KIND) {
        case ExampleIndex::K_BRUTE_FORCE:
            return {
                "brute_force",
                R"(
                {
                    "dtype": "float32",
                    "metric_type": "l2",
                    "dim": 16,
                    "index_param": {
                        "base_quantization_type": "fp32",
                        "base_io_type": "memory_io"
                    }
                }
                )",
                R"(
                {
                    "base_io_type": "memory_io"
                }
                )",
                "{}",
                "/tmp/vsag-streaming-load-bruteforce.index",
                K_DENSE_DIM,
                false,
                false,
            };
        case ExampleIndex::K_H_GRAPH:
            return {
                "hgraph",
                R"(
                {
                    "dtype": "float32",
                    "metric_type": "l2",
                    "dim": 16,
                    "index_param": {
                        "base_quantization_type": "sq8",
                        "max_degree": 16,
                        "ef_construction": 100,
                        "alpha": 1.2
                    }
                }
                )",
                "{}",
                R"(
                {
                    "hgraph": {
                        "ef_search": 100
                    }
                }
                )",
                "/tmp/vsag-streaming-load-hgraph.index",
                K_DENSE_DIM,
                false,
                false,
            };
        case ExampleIndex::K_IVF:
            return {
                "ivf",
                R"(
                {
                    "dtype": "float32",
                    "metric_type": "l2",
                    "dim": 16,
                    "index_param": {
                        "buckets_count": 10,
                        "base_quantization_type": "fp32",
                        "partition_strategy_type": "ivf",
                        "ivf_train_type": "kmeans",
                        "train_sample_count": 512
                    }
                }
                )",
                "{}",
                R"(
                {
                    "ivf": {
                        "scan_buckets_count": 5
                    }
                }
                )",
                "/tmp/vsag-streaming-load-ivf.index",
                K_DENSE_DIM,
                false,
                false,
            };
        case ExampleIndex::K_PYRAMID:
            return {
                "pyramid",
                R"(
                {
                    "dtype": "float32",
                    "metric_type": "l2",
                    "dim": 16,
                    "index_param": {
                        "base_quantization_type": "sq8",
                        "max_degree": 24,
                        "alpha": 1.2,
                        "graph_iter_turn": 15,
                        "neighbor_sample_rate": 0.2,
                        "no_build_levels": [0, 1],
                        "use_reorder": true,
                        "graph_type": "odescent",
                        "build_thread_count": 1
                    }
                }
                )",
                "{}",
                R"(
                {
                    "pyramid": {
                        "ef_search": 100
                    }
                }
                )",
                "/tmp/vsag-streaming-load-pyramid.index",
                K_DENSE_DIM,
                true,
                false,
            };
        case ExampleIndex::K_SINDI:
            return {
                "sindi",
                R"(
                {
                    "dtype": "sparse",
                    "metric_type": "ip",
                    "dim": 128,
                    "index_param": {
                        "use_reorder": true,
                        "use_quantization": false,
                        "doc_prune_ratio": 0.0,
                        "term_prune_ratio": 0.0,
                        "window_size": 10000,
                        "term_id_limit": 30001,
                        "avg_doc_term_length": 100
                    }
                }
                )",
                "{}",
                R"(
                {
                    "sindi": {
                        "query_prune_ratio": 0.0,
                        "term_prune_ratio": 0.0,
                        "n_candidate": 20
                    }
                }
                )",
                "/tmp/vsag-streaming-load-sindi.index",
                K_SPARSE_DIM,
                false,
                true,
            };
    }
    std::abort();
}

vsag::DatasetPtr
make_dense_dataset(std::vector<int64_t>& ids,
                   std::vector<float>& vectors,
                   std::vector<std::string>& paths,
                   int64_t dim,
                   bool use_paths) {
    auto dataset = vsag::Dataset::Make()
                       ->NumElements(static_cast<int64_t>(ids.size()))
                       ->Dim(dim)
                       ->Ids(ids.data())
                       ->Float32Vectors(vectors.data())
                       ->Owner(false);
    if (use_paths) {
        dataset->Paths(paths.data());
    }
    return dataset;
}

void
fill_sparse_vector(vsag::SparseVector& vector,
                   std::mt19937& rng,
                   std::uniform_real_distribution<float>& value_dist,
                   std::uniform_int_distribution<uint32_t>& term_dist,
                   int64_t nnz) {
    vector.len_ = nnz;
    vector.ids_ = new uint32_t[vector.len_];
    vector.vals_ = new float[vector.len_];
    std::unordered_set<uint32_t> terms;
    std::vector<std::pair<uint32_t, float>> entries;
    entries.reserve(static_cast<size_t>(nnz));
    for (int64_t i = 0; i < nnz; ++i) {
        auto term = term_dist(rng);
        while (terms.count(term) != 0) {
            term = term_dist(rng);
        }
        terms.insert(term);
        entries.emplace_back(term, value_dist(rng));
    }
    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });
    for (int64_t i = 0; i < nnz; ++i) {
        vector.ids_[i] = entries[static_cast<size_t>(i)].first;
        vector.vals_[i] = entries[static_cast<size_t>(i)].second;
    }
}

void
release_sparse_vectors(std::vector<vsag::SparseVector>& vectors) {
    for (auto& vector : vectors) {
        delete[] vector.ids_;
        delete[] vector.vals_;
        vector.ids_ = nullptr;
        vector.vals_ = nullptr;
        vector.len_ = 0;
    }
}

vsag::DatasetPtr
make_sparse_dataset(std::vector<int64_t>& ids, std::vector<vsag::SparseVector>& vectors) {
    return vsag::Dataset::Make()
        ->NumElements(static_cast<int64_t>(ids.size()))
        ->Ids(ids.data())
        ->SparseVectors(vectors.data())
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

}  // namespace

int
main() {
    vsag::init();
    vsag::Options::Instance().set_block_size_limit(2UL * 1024 * 1024);
    const auto config = get_example_config();

    std::vector<int64_t> ids(K_NUM_VECTORS);
    std::vector<float> vectors(K_NUM_VECTORS * config.dim);
    std::vector<std::string> paths(K_NUM_VECTORS);
    std::vector<vsag::SparseVector> sparse_vectors(K_NUM_VECTORS);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib;
    std::uniform_real_distribution<float> sparse_value_dist(0.0F, 10.0F);
    std::uniform_int_distribution<uint32_t> sparse_term_dist(0, 30000);
    for (int64_t i = 0; i < K_NUM_VECTORS; ++i) {
        ids[i] = i;
        paths[i] = "path/" + std::to_string(i);
        if (config.use_sparse) {
            fill_sparse_vector(sparse_vectors[i], rng, sparse_value_dist, sparse_term_dist, 64);
        }
    }
    for (auto& value : vectors) {
        value = distrib(rng);
    }

    auto index = vsag::Factory::CreateIndex(config.index_name, config.build_parameters).value();
    auto base = config.use_sparse
                    ? make_sparse_dataset(ids, sparse_vectors)
                    : make_dense_dataset(ids, vectors, paths, config.dim, config.use_paths);
    auto build_result = index->Build(base);
    if (!build_result.has_value()) {
        std::cerr << "build failed: " << build_result.error().message << std::endl;
        return -1;
    }

    {
        std::ofstream out(config.file_path, std::ios::binary);
        check_result(index->SerializeStreaming(out), "SerializeStreaming");
    }

    auto full_index =
        vsag::Factory::CreateIndex(config.index_name, config.build_parameters).value();
    {
        std::ifstream in(config.file_path, std::ios::binary);
        check_result(full_index->DeserializeStreaming(in), "DeserializeStreaming");
    }

    vsag::IndexPtr loaded_index;
    {
        std::ifstream in(config.file_path, std::ios::binary);
        auto load_result = vsag::Index::Load(in, config.load_parameters);
        check_result(load_result, "Load");
        loaded_index = load_result.value();
    }

    std::vector<float> query_vector(config.dim);
    std::vector<std::string> query_paths{"path/0"};
    std::vector<vsag::SparseVector> query_sparse_vectors(1);
    for (auto& value : query_vector) {
        value = distrib(rng);
    }
    if (config.use_sparse) {
        fill_sparse_vector(query_sparse_vectors[0], rng, sparse_value_dist, sparse_term_dist, 64);
    }
    auto query = vsag::Dataset::Make()->NumElements(1)->Owner(false);
    if (config.use_sparse) {
        query->SparseVectors(query_sparse_vectors.data());
    } else {
        query->Dim(config.dim)->Float32Vectors(query_vector.data());
    }
    if (config.use_paths) {
        query->Paths(query_paths.data());
    }
    auto result = loaded_index->KnnSearch(query, K_TOP_K, config.search_parameters).value();

    std::cout << "After Build(), Index " << config.index_name
              << " contains: " << index->GetNumElements() << std::endl;
    std::cout << "Streaming index file: " << config.file_path << std::endl;
    std::cout << "results: " << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }
    release_sparse_vectors(sparse_vectors);
    release_sparse_vectors(query_sparse_vectors);
    return 0;
}
