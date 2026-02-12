
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

#include <iostream>
#include <unordered_set>

int
main(int argc, char** argv) {
    vsag::init();

    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 1000;
    int64_t max_dim = 128;
    int64_t max_id = 30000;
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real(0, 10);
    std::uniform_int_distribution<int> distrib_dim(64, 128);
    std::uniform_int_distribution<int> distrib_id(0, max_id);

    std::vector<int64_t> ids(num_vectors);
    std::vector<vsag::SparseVector> sparse_vectors(num_vectors);

    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
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

    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->SparseVectors(sparse_vectors.data())
        ->Ids(ids.data())
        ->Owner(false);

    /******************* Create SINDI Index *****************/
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
    auto sindi_build_parameters = R"({
        "dtype": "sparse",
        "dim": 128,
        "metric_type": "ip",
        "index_param": {
            "use_reorder": true,
            "term_id_limit": 1000000,
            "doc_prune_ratio": 0.0,
            "window_size": 60000
        }
    })";

    auto index = vsag::Factory::CreateIndex("sindi", sindi_build_parameters).value();

    /******************* Build SINDI Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index SINDI contains: " << index->GetNumElements()
                  << std::endl;
    } else if (build_result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
        std::cerr << "Failed to build index: internalError" << std::endl;
        exit(-1);
    }

    /******************* Prepare Query Dataset *****************/
    vsag::SparseVector query_vector;
    query_vector.len_ = 64;
    query_vector.ids_ = new uint32_t[query_vector.len_];
    query_vector.vals_ = new float[query_vector.len_];
    std::unordered_set<uint32_t> unique_query_ids;
    for (int d = 0; d < query_vector.len_; d++) {
        auto u_id = distrib_id(rng);
        while (unique_query_ids.count(u_id) > 0) {
            u_id = distrib_id(rng);
        }
        unique_query_ids.insert(u_id);
        query_vector.ids_[d] = u_id;
        query_vector.vals_[d] = distrib_real(rng);
    }
    std::sort(query_vector.ids_, query_vector.ids_ + query_vector.len_);

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->SparseVectors(&query_vector)->Owner(false);

    /******************* KnnSearch For SINDI Index *****************/
    /*
     * search_params is the configuration for sparse index search.
     *
     * - sindi: Parameters specific to sparse indexing search:
     *   - query_prune_ratio: Ratio of term pruning for the query (0 = no pruning).
     *   - n_candidate: Number of candidates for re-ranking. Must be greater than topK.
     *     This parameter is ignored if use_reorder is false in the build parameters.
     */
    auto sindi_search_parameters = R"({
        "sindi": {
            "query_prune_ratio": 0,
            "n_candidate": 0
        }
    })";

    int64_t topk = 10;
    auto result = index->KnnSearch(query, topk, sindi_search_parameters).value();

    /******************* Print Search Result *****************/
    std::cout << "results: " << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }

    /******************* Cleanup *****************/
    for (auto& item : sparse_vectors) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
    delete[] query_vector.vals_;
    delete[] query_vector.ids_;

    return 0;
}
