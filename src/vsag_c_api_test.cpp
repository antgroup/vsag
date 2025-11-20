
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

#include <vsag/vsag_c_api.h>

#include <catch2/catch_test_macros.hpp>
#include <random>

#include "fixtures.h"

TEST_CASE("vsag_c_api basic test", "[vsag_c_api][ut]") {
    const char* index_name = "hgraph";
    const char* index_param = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 64,
            "ef_construction": 200,
            "alpha":1.2
        }
    }
    )";
    auto index = vsag_index_factory(index_name, index_param);
    REQUIRE(index != nullptr);

    int64_t num_vectors = 500;
    int64_t dim = 128;
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
    Error_t ret = vsag_index_build(index, datas.data(), ids.data(), dim, num_vectors);
    REQUIRE(ret.code == VSAG_SUCCESS);
    auto func = [&](vsag_index_t index1) {
        const char* hgraph_search_parameters = R"(
            {
                "hgraph": {
                    "ef_search": 200
                }
            }
            )";
        int64_t topk = 10;
        for (int i = 0; i < num_vectors; ++i) {
            std::vector<int64_t> results(topk);
            std::vector<float> scores(topk);
            ret = vsag_index_knn_search(index1,
                                        datas.data() + i * dim,
                                        dim,
                                        topk,
                                        hgraph_search_parameters,
                                        scores.data(),
                                        results.data());
            REQUIRE(ret.code == VSAG_SUCCESS);
            bool in_results = false;
            for (int64_t j = 0; j < topk; ++j) {
                if (results[j] == i) {
                    in_results = true;
                    break;
                }
            }
            REQUIRE(in_results);
        }
    };
    func(index);

    auto dir = fixtures::TempDir("vsag_c_api");
    auto path = dir.GenerateRandomFile();
    ret = vsag_serialize_file(index, path.data());
    REQUIRE(ret.code == VSAG_SUCCESS);
    vsag_index_t index2 = vsag_index_factory(index_name, index_param);
    ret = vsag_deserialize_file(index2, path.data());
    REQUIRE(ret.code == VSAG_SUCCESS);
    REQUIRE(index2 != nullptr);

    func(index2);

    vsag_index_destroy(index);
    vsag_index_destroy(index2);
}
