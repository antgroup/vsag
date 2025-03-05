
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

const std::string tmp_dir = "/tmp/";

int
main(int argc, char** argv) {
    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 1000;
    int64_t dim = 128;
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }
    auto base = vsag::Dataset::Make();
    // Transfer the ownership of the data (ids, vectors) to the base.
    base->NumElements(num_vectors)->Dim(dim)->Ids(ids)->Float32Vectors(vectors);

    /******************* Create DiskANN Index *****************/
    // diskann_build_paramesters is the configuration for building a DiskANN index.
    // The "dtype" specifies the data type, "metric_type" indicates the distance metric,
    // and "dim" represents the dimensionality of the feature vectors.
    // The "diskann" section contains parameters specific to DiskANN:
    // - "max_degree": Maximum degree of the graph
    // - "ef_construction": Construction phase efficiency factor
    // - "pq_sample_rate": PQ sampling rate
    // - "pq_dims": PQ dimensionality
    // - "use_pq_search": Indicates whether to cache the graph in memory and use PQ vectors for retrieval (optional)
    // - "use_async_io": Specifies whether to use asynchronous I/O (optional)
    // - "use_bsa": Determines whether to use the BSA method for lossless filtering during the reordering phase (optional)
    // Other parameters are mandatory.
    auto diskann_build_paramesters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "diskann": {
            "max_degree": 16,
            "ef_construction": 200,
            "pq_sample_rate": 0.5,
            "pq_dims": 9,
            "use_pq_search": true,
            "use_async_io": true,
            "use_bsa": true
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("diskann", diskann_build_paramesters).value();

    /******************* Build DiskANN Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index DiskANN contains: " << index->GetNumElements()
                  << std::endl;
    } else {
        std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
        exit(-1);
    }

    /******************* KnnSearch For DiskANN Index *****************/
    auto query_vector = new float[dim];
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }

    // diskann_search_parameters is the configuration for searching in a DiskANN index.
    // The "diskann" section contains parameters specific to the search operation:
    // - "ef_search": The search efficiency factor, which influences accuracy and speed.
    // - "beam_search": The number of beams to use during the search process, balancing exploration and exploitation.
    // - "io_limit": The maximum number of I/O operations allowed during the search.
    // - "use_reorder": Indicates whether to perform reordering of results for better accuracy (optional).

    auto diskann_search_parameters = R"(
    {
        "diskann": {
            "ef_search": 100,
            "beam_search": 4,
            "io_limit": 50,
            "use_reorder": true
        }
    }
    )";
    int64_t topk = 10;
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(true);

    /******************* Print Search Result *****************/
    auto knn_result = index->KnnSearch(query, topk, diskann_search_parameters);
    if (knn_result.has_value()) {
        auto result = knn_result.value();
        std::cout << "results: " << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
        }
    } else {
        std::cerr << "Search Error: " << knn_result.error().message << std::endl;
    }

    /******************* Serialize and Clean Index *****************/
    std::unordered_map<std::string, size_t> meta_info;
    {
        if (auto bs = index->Serialize(); bs.has_value()) {
            index = nullptr;
            auto keys = bs->GetKeys();
            for (auto key : keys) {
                vsag::Binary b = bs->Get(key);
                std::ofstream file(tmp_dir + "diskann.index." + key, std::ios::binary);
                file.write((const char*)b.data.get(), b.size);
                file.close();
                meta_info[key] = b.size;
            }
        } else if (bs.error().type == vsag::ErrorType::NO_ENOUGH_MEMORY) {
            std::cerr << "no enough memory to serialize index" << std::endl;
        }
    }

    // Through the Reader, disk-based DiskANN retrieval can be achieved, and the Reader can be customized.
    // Here we provide an example of a Reader based on the local file system.
    /******************* Deserialize with Reader *****************/
    {
        vsag::ReaderSet rs;
        for (const auto& [key, size] : meta_info) {
            auto reader =
                vsag::Factory::CreateLocalFileReader(tmp_dir + "diskann.index." + key, 0, size);
            rs.Set(key, reader);
        }

        if (auto result = vsag::Factory::CreateIndex("diskann", diskann_build_paramesters);
            result.has_value()) {
            index = result.value();
        } else {
            std::cout << "Build DiskANN Error" << std::endl;
            exit(-1);
        }
        index->Deserialize(rs);
    }

    /******************* Print Search Result *****************/
    knn_result = index->KnnSearch(query, topk, diskann_search_parameters);
    if (knn_result.has_value()) {
        auto result = knn_result.value();
        std::cout << "results: " << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
        }
    } else {
        std::cerr << "Search Error: " << knn_result.error().message << std::endl;
    }

    return 0;
}
