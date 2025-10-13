
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

#include <fstream>
#include <iostream>
#include <unordered_set>

#include "vsag/binaryset.h"

float
compute_recall(vsag::DatasetPtr global, vsag::DatasetPtr local, int k) {
    std::unordered_set<int64_t> global_set;
    std::unordered_set<int64_t> local_set;
    float recall = 0;

    for (auto i = 0; i < global->GetDim(); ++i) {
        global_set.insert(global->GetIds()[i]);
    }

    for (auto i = 0; i < local->GetDim(); ++i) {
        auto id = local->GetIds()[i];
        if (global_set.count(id) != 0) {
            recall += 1;
        }
    }
    return recall / (float)k;
}

std::vector<vsag::SparseVector>
GenerateSparseVectors(
    uint32_t count, uint32_t max_dim, uint32_t max_id, float min_val, float max_val, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distrib_real(min_val, max_val);
    std::uniform_int_distribution<int> distrib_dim(max_dim / 2, max_dim);
    std::uniform_int_distribution<int> distrib_id(0, max_id);

    std::vector<vsag::SparseVector> sparse_vectors(count);
    if (max_dim > max_id) {
        throw std::runtime_error("generate sparse vectors failed, max_dim > max_id");
    }

    for (int i = 0; i < count; i++) {
        sparse_vectors[i].len_ = distrib_dim(rng);
        sparse_vectors[i].ids_ = new uint32_t[sparse_vectors[i].len_];
        sparse_vectors[i].vals_ = new float[sparse_vectors[i].len_];
        std::unordered_set<uint32_t> unique_ids;
        for (int d = 0; d < sparse_vectors[i].len_; d++) {
            auto u_id = distrib_id(rng);
            while (unique_ids.count(u_id) > 0) {
                u_id = distrib_id(rng);
            }
            unique_ids.insert(u_id);
            sparse_vectors[i].ids_[d] = u_id;
            sparse_vectors[i].vals_[d] = distrib_real(rng);
        }

        std::sort(sparse_vectors[i].ids_, sparse_vectors[i].ids_ + sparse_vectors[i].len_);
    }

    return sparse_vectors;
}

int
main(int argc, char** argv) {
    vsag::init();

    /******************* Prepare Base and Query Dataset *****************/
    uint32_t num_base = 10000;
    uint32_t num_query = 100;
    int64_t max_dim = 128;
    int64_t max_id = 30000;
    float min_val = 0;
    float max_val = 10;
    int seed_base = 114;
    int seed_query = 514;
    int64_t k = 10;

    std::vector<int64_t> ids(num_base);
    for (int64_t i = 0; i < num_base; ++i) {
        ids[i] = i;
    }

    auto sv_base = GenerateSparseVectors(num_base, max_dim, max_id, min_val, max_val, seed_base);
    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    auto sv_query =
        GenerateSparseVectors(num_query, max_dim / 2, max_id, min_val, max_val, seed_query);
    auto query = vsag::Dataset::Make();

    std::vector<vsag::DatasetPtr> gt_results(num_query);

    /******************* Create Index *****************/
    /*
     * build_params is the configuration for building a sparse index.
     *
     * - dtype: Must be set to "sparse", indicating the data type of the vectors.
     * - dim: Dimensionality of the sparse vectors (must be >0, but does not affect the result).
     * - metric_type: Distance metric type, currently only "ip" (inner product) is supported.
     * - index_param: Parameters specific to sparse indexing:
     *   - use_reorder: If true, enables full-precision re-ranking of results. This requires storing additional data.
     *     When doc_prune_ratio is 0, use_reorder can be false while still maintaining full-precision results.
     *   - term_id_limit: Maximum term id (e.g., when term_id_limit = 10, then, term [15: 0.1] in sparse vector is not allowed)
     *   - doc_prune_ratio: Ratio of term pruning in documents (0 = no pruning).
     *   - window_size: Window size for table scanning. Related to L3 cache size; 100000 is an empirically optimal value.
     */
    auto build_params = R"(
    {
        "dtype": "sparse",
        "dim": 128,
        "metric_type": "ip",
        "index_param": {
            "use_reorder": true,
            "term_id_limit": 1000000,
            "doc_prune_ratio": 0.0,
            "window_size": 100000
        }
    }
    )";

    auto bf_index = vsag::Factory::CreateIndex("sparse_index", build_params).value();
    bf_index->Build(base);

    for (auto i = 0; i < num_query; i++) {
        query->NumElements(1)->SparseVectors(sv_query.data() + i)->Owner(false);
        gt_results[i] = bf_index->KnnSearch(query, k, "").value();
    }

    auto index = vsag::Factory::CreateIndex("sindi", build_params).value();

    /******************* Build Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Sparse Term Index contains: " << index->GetNumElements()
                  << std::endl;
    } else if (build_result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
        std::cerr << "Failed to build index: internalError" << std::endl;
        exit(-1);
    }

    /******************* Save Index to OStream *****************/
    auto tmp_file = "/tmp/vsag-persistent-streaming-sparse.index";
    std::ofstream out_stream(tmp_file);
    auto serialize_result = index->Serialize(out_stream);
    out_stream.close();
    if (not serialize_result.has_value()) {
        std::cerr << serialize_result.error().message << std::endl;
        abort();
    } else {
        std::cout << "finish serialize" << std::endl;
    }

    /******************* Load Index from IStream *****************/
    index = nullptr;
    if (auto create_index = vsag::Factory::CreateIndex("sindi", build_params);
        not create_index.has_value()) {
        std::cout << "create index failed: " << create_index.error().message << std::endl;
        abort();
    } else {
        index = *create_index;
    }
    std::ifstream in_stream(tmp_file);
    if (auto deserialize = index->Deserialize(in_stream); not deserialize.has_value()) {
        std::cerr << "load index failed: " << deserialize.error().message << std::endl;
        abort();
    } else {
        std::cout << "finish deserialize" << std::endl;
    }

    /******************* KnnSearch *****************/
    /*
     * search_params is the configuration for sparse index search.
     *
     * - sindi: Parameters specific to sparse indexing search:
     *   - query_prune_ratio: Ratio of term pruning for the query (0 = no pruning).
     *   - n_candidate: Number of candidates for re-ranking. Must be greater than topK.
     *     This parameter is ignored if use_reorder is false in the build parameters.
     */
    auto search_params = R"(
    {
        "sindi": {
            "query_prune_ratio": 0,
            "n_candidate": 0
        }
    }
    )";

    std::cout << "start query" << std::endl;

    float recall = 0.0f;
    int64_t total_search_time_ns = 0;

    for (int i = 0; i < num_query; ++i) {
        query->NumElements(1)->SparseVectors(sv_query.data() + i)->Owner(false);

        auto start_time = std::chrono::high_resolution_clock::now();
        auto result = index->KnnSearch(query, k, search_params).value();
        auto end_time = std::chrono::high_resolution_clock::now();

        auto search_time_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        total_search_time_ns += search_time_ns;

        recall += compute_recall(gt_results[i], result, k);
    }

    recall /= num_query;

    double total_search_time_s = total_search_time_ns / 1e9;
    double qps = num_query / total_search_time_s;

    std::cout << "Recall: " << recall << std::endl;
    std::cout << "QPS: " << qps << std::endl;
    gt_results
        .clear();  // Ensure that the results obtained from bf_index are cleared before bf_index is destroyed.
    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
    for (auto& item : sv_query) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
    return 0;
}
