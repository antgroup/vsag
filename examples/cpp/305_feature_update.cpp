
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

int
main(int argc, char** argv) {
    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 10000;
    int64_t dim = 64;
    std::vector<int64_t> ids(num_vectors);
    std::vector<float> vectors(num_vectors * dim);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    /******************* Create HNSW Index *****************/
    auto hnsw_build_paramesters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 64,
        "hnsw": {
            "max_degree": 16,
            "ef_construction": 100
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("hnsw", hnsw_build_paramesters).value();

    /******************* Build HNSW Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index Hnsw contains: " << index->GetNumElements() << std::endl;
    } else {
        std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
        exit(-1);
    }

    /******************* Update Vector in Index *****************/
    int64_t update_id = 9527;
    std::vector<float> new_vector(dim);
    for (int64_t i = 0; i < dim; ++i) {
        new_vector[i] = distrib_real(rng);
    }
    auto update_dataset = vsag::Dataset::Make();
    update_dataset->NumElements(1)
        ->Dim(dim)
        ->Ids(&update_id)
        ->Float32Vectors(new_vector.data())
        ->Owner(false);

    // try to update determines by the distance between the new vector and the old vector
    if (auto update_status = index->UpdateVector(update_id, update_dataset, false);
        not update_status.has_value()) { /* update returns an error */
        std::cerr << "update vector failed: " << update_status.error().message << std::endl;
        abort();
    } else if (*update_status) { /* updated, new vector is near to the old vector */
        std::cout << "updated, new vector is near to the old vector" << std::endl;
    } else { /* not update, new vector is far away from the old vector */
        std::cout << "not update, new vector is far away from the old vector" << std::endl;

        // not good to update in-place, choose to delete and insert
        if (auto remove = index->Remove(update_id);
            not remove.has_value()) { /* remove returns an error */
            std::cerr << "delete vector failed: " << remove.error().message << std::endl;
            abort();
        } else if (not *remove) { /* id not exists, should NOT happend in this example */
            std::cerr << "example error" << std::endl;
            abort();
        } else { /* delete vector success */
            std::cout << "delete old vector" << std::endl;
            if (auto add = index->Add(update_dataset);
                not add.has_value()) { /* add returns an error */
                std::cout << "insert vector failed: " << add.error().message << std::endl;
                abort();
            } else if (
                not add->empty()) { /* not insert, id is already exist in index, shoud NOT happen in this example */
                std::cerr << "example error" << std::endl;
                abort();
            } else {
                std::cout << "insert new vector" << std::endl;
            }
        }
    }

    /******************* Search and Print Results *****************/
    auto topk = 10;
    auto query_vector = new float[dim];
    for (uint64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(false);
    auto search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        }
    }
    )";
    if (auto knn_search = index->KnnSearch(query, topk, search_parameters);
        not knn_search.has_value()) {
        std::cerr << "search knn failed: " << knn_search.error().message << std::endl;
        abort();
    } else {
        auto result = *knn_search;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            std::cout << result->GetIds()[i] << " " << result->GetDistances()[i] << std::endl;
        }
    }
}
