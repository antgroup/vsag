
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
#include <iostream>
#include <random>
#include <vector>

int
main() {
    vsag::init();

    /******************* Prepare Base Dataset *****************/
    int64_t num_docs = 200;           // number of documents
    int64_t sub_dim = 32;             // dimension of each sub-vector (token)
    int64_t max_tokens_per_doc = 20;  // upper bound on tokens per document
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib(0.0f, 1.0f);
    std::uniform_int_distribution<int64_t> tok_distrib(1, max_tokens_per_doc);

    // Allocate all sub-vectors in one flat buffer so MultiVector::vectors_
    // can point into it (Owner(false) on the dataset).
    std::vector<int64_t> ids(num_docs);
    std::vector<uint32_t> token_counts(num_docs);
    int64_t total_tokens = 0;
    for (int64_t i = 0; i < num_docs; ++i) {
        ids[i] = i;
        token_counts[i] = static_cast<uint32_t>(tok_distrib(rng));
        total_tokens += token_counts[i];
    }

    std::vector<float> base_data(static_cast<size_t>(total_tokens) * sub_dim);
    for (auto& v : base_data) {
        v = distrib(rng);
    }

    // Build the MultiVector array: each entry points to a slice of base_data.
    std::vector<vsag::MultiVector> base_mvs(num_docs);
    float* cursor = base_data.data();
    for (int64_t i = 0; i < num_docs; ++i) {
        base_mvs[i].len_ = token_counts[i];
        base_mvs[i].vectors_ = cursor;
        cursor += token_counts[i] * sub_dim;
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(num_docs)
        ->Dim(sub_dim)
        ->Ids(ids.data())
        ->MultiVectors(base_mvs.data())
        ->MultiVectorDim(sub_dim)
        ->Owner(false);

    /******************* Create SIMQ Index *****************/
    /*
     * build_params is the configuration for building a SIMQ (multi-vector) index.
     *
     * - dtype: Must be "float32" for multi-vector data.
     * - dim: Sub-vector dimension (same as MultiVectorDim on the dataset).
     * - metric_type: Distance metric. SIMQ uses inner product ("ip") over
     *   sub-vectors and aggregates across a document via MaxSim.
     * - index_param: Parameters for the SIMQ index:
     *   - base_io_type: Storage backend for token vectors. "memory_io" keeps
     *     everything in RAM; "disk_io" offloads to a file for large datasets.
     *   - base_file_path: Path used by the storage backend (only meaningful for
     *     disk_io).
     *   - init_cluster_ratio: Fraction of base tokens used as initial cluster
     *     seeds. Larger values yield more clusters.
     *   - max_cluster_size: Split threshold — a cluster is split once it grows
     *     beyond this many tokens.
     *   - split_start_idx: Position within a sorted cluster at which a new
     *     sub-cluster begins during a split.
     *   - coarse_k: Default number of clusters probed per query token during
     *     search (overridable per-query).
     *   - rerank_k: Number of candidate documents reranked with exact MaxSim.
     *   - build_thread_count: Number of threads used during Build().
     *   - split_delay_seconds: Seconds to wait after the first cluster overflow
     *     before triggering a split (0 = immediate).
     *   - quantization_type: Token storage format — "fp32" (default), "fp16",
     *     "bf16", "sq8_uniform", or "int8". Smaller formats trade some MaxSim
     *     precision for reduced memory.
     */
    auto simq_build_params = R"({
        "dtype": "float32",
        "dim": 32,
        "metric_type": "ip",
        "index_param": {
            "base_io_type": "memory_io",
            "base_file_path": "/tmp/simq_example_base.bin",
            "init_cluster_ratio": 0.2,
            "max_cluster_size": 64,
            "split_start_idx": 32,
            "coarse_k": 8,
            "rerank_k": 100,
            "build_thread_count": 1,
            "split_delay_seconds": 0.0,
            "quantization_type": "fp32"
        }
    })";

    auto create_res = vsag::Factory::CreateIndex("simq", simq_build_params);
    if (not create_res.has_value()) {
        std::cerr << "CreateIndex(simq) failed" << std::endl;
        return -1;
    }
    auto index = create_res.value();

    /******************* Build SIMQ Index *****************/
    if (auto build_res = index->Build(base); build_res.has_value()) {
        std::cout << "After Build(), Index SIMQ contains: " << index->GetNumElements() << std::endl;
    } else {
        std::cerr << "Failed to build index: " << build_res.error().message << std::endl;
        return -1;
    }

    /******************* Prepare Query Dataset *****************/
    // A query is itself a multi-vector (a small document).
    int64_t query_tokens = 5;
    std::vector<float> query_data(static_cast<size_t>(query_tokens) * sub_dim);
    for (auto& v : query_data) {
        v = distrib(rng);
    }

    vsag::MultiVector query_mv;
    query_mv.len_ = static_cast<uint32_t>(query_tokens);
    query_mv.vectors_ = query_data.data();

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(sub_dim)->MultiVectors(&query_mv)->MultiVectorDim(sub_dim)->Owner(
        false);

    /******************* KnnSearch For SIMQ Index *****************/
    /*
     * search_params is the configuration for SIMQ search:
     *
     * - simq: Parameters specific to the SIMQ index:
     *   - coarse_k: Number of clusters to probe per query token. Higher values
     *     improve recall at the cost of latency. Overrides the build-time
     *     default.
     *   - rerank_k: Number of candidate documents reranked with exact MaxSim.
     *     Must be >= topK. Overrides the build-time default.
     */
    auto simq_search_params = R"({
        "simq": {
            "coarse_k": 16,
            "rerank_k": 50
        }
    })";

    int64_t topk = 10;
    auto search_res = index->KnnSearch(query, topk, simq_search_params);
    if (not search_res.has_value()) {
        std::cerr << "KnnSearch failed: " << search_res.error().message << std::endl;
        return -1;
    }
    auto result = search_res.value();

    /******************* Print Search Result *****************/
    std::cout << "results:" << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }

    return 0;
}
