
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

std::vector<std::pair<std::vector<float>, std::string>>
loadVectorFilterPairs(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        return {};
    }

    std::vector<std::pair<std::vector<float>, std::string>> result;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::vector<float> vec(32);
        std::istringstream iss(line);
        char comma;
        bool success = true;

        for (int i = 0; i < 32 && success; ++i) {
            if (!(iss >> vec[i])) {
                success = false;
                break;
            }
            if (i != 31 && !(iss >> comma)) {
                success = false;
                break;
            }
        }

        if (!success) {
            continue;
        }

        std::string filterLine;
        if (!std::getline(file, filterLine)) {
            break;
        }

        result.emplace_back(vec, filterLine);
    }

    file.close();
    return result;
}

int
main(int argc, char** argv) {
    vsag::init();

    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 10000;
    int64_t dim = 32;
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

    /******************* Create IVF Index *****************/
    std::string ivf_build_params = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 32,
        "index_param": {
            "buckets_count": 100,
            "base_quantization_type": "fp32",
            "partition_strategy_type": "ivf",
            "ivf_train_type": "kmeans",
            "use_reorder":false,
            "use_attribute_filter":true
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("ivf", ivf_build_params).value();

    // /******************* Build IVF Index *****************/
    // if (auto build_result = index->Build(base); build_result.has_value()) {
    //     std::cout << "After Build(), Index IVF contains: " << index->GetNumElements() << std::endl;
    // } else if (build_result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
    //     std::cerr << "Failed to build index: internalError" << std::endl;
    //     exit(-1);
    // }
    std::ifstream infile("/home/tianlan.lht/suguan.dx_ivf.index.bin.10", std::ios::binary);

    index->Deserialize(infile);

    /******************* Prepare Query Dataset *****************/
    // std::vector<float> query_vector(dim);
    // for (int64_t i = 0; i < dim; ++i) {
    //     query_vector[i] = distrib_real(rng);
    // }

    /******************* KnnSearch For IVF Index *****************/
    auto ivf_search_parameters = R"(
    {
        "ivf": {
            "scan_buckets_count": 30
        }
    })";
    int64_t topk = 500;

    auto querys = loadVectorFilterPairs("/home/tianlan.lht/suguan.dx_req10");

    int query_count = querys.size();
    while (true) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < query_count; ++i) {
            //        std::cout << "query: " << i << std::endl;
            auto query = vsag::Dataset::Make();
            query->NumElements(1)->Dim(dim)->Float32Vectors(querys[i].first.data())->Owner(false);
            vsag::SearchRequest req;
            req.query_ = query;
            req.topk_ = topk;
            req.enable_attribute_filter_ = true;
            req.attribute_filter_str_ = querys[i].second;
            req.params_str_ = ivf_search_parameters;
            auto result = index->SearchWithRequest(req).value();

            // std::cout << "results: " << std::endl;
            // for (int64_t i = 0; i < result->GetDim(); ++i) {
            //     std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
            // }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "query count: " << query_count << ", time: " << duration.count() << "ms"
                  << std::endl;
    }

    /******************* Print Search Result *****************/

    return 0;
}
