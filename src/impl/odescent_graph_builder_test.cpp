
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

#include "odescent_graph_builder.h"

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <set>

#include "data_cell/flatten_interface.h"
#include "io/memory_io_parameter.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "safe_allocator.h"

size_t
calculate_overlap(const std::vector<uint32_t>& vec1, vsag::Vector<uint32_t>& vec2, int K) {
    int size1 = std::min(K, static_cast<int>(vec1.size()));
    int size2 = std::min(K, static_cast<int>(vec2.size()));

    std::vector<uint32_t> top_k_vec1(vec1.begin(), vec1.begin() + size1);
    std::vector<uint32_t> top_k_vec2(vec2.begin(), vec2.begin() + size2);

    std::sort(top_k_vec1.rbegin(), top_k_vec1.rend());
    std::sort(top_k_vec2.rbegin(), top_k_vec2.rend());

    std::set<uint32_t> set1(top_k_vec1.begin(), top_k_vec1.end());
    std::set<uint32_t> set2(top_k_vec2.begin(), top_k_vec2.end());

    std::set<uint32_t> intersection;
    std::set_intersection(set1.begin(),
                          set1.end(),
                          set2.begin(),
                          set2.end(),
                          std::inserter(intersection, intersection.begin()));
    return intersection.size();
}

TEST_CASE("build nndescent", "[ut][nndescent]") {
    int64_t num_vectors = 2000;
    size_t dim = 128;
    int64_t max_degree = 32;

    auto vectors = new float[dim * num_vectors];

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }

    std::vector<std::vector<std::pair<float, uint32_t>>> ground_truths(num_vectors);
    vsag::IndexCommonParam param;
    param.dim_ = dim;
    param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    param.thread_pool_ = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    vsag::FlattenDataCellParamPtr flatten_param =
        std::make_shared<vsag::FlattenDataCellParameter>();
    flatten_param->quantizer_parameter_ = std::make_shared<vsag::FP32QuantizerParameter>();
    flatten_param->io_parameter_ = std::make_shared<vsag::MemoryIOParameter>();
    vsag::FlattenInterfacePtr flatten_interface_ptr =
        vsag::FlattenInterface::MakeInstance(flatten_param, param);
    flatten_interface_ptr->Train(vectors, num_vectors);
    flatten_interface_ptr->BatchInsertVector(vectors, num_vectors);

    vsag::DatasetPtr dataset = vsag::Dataset::Make();
    dataset->NumElements(num_vectors)->Float32Vectors(vectors)->Dim(dim)->Owner(true);
    vsag::ODescent graph(max_degree,
                         1,
                         30,
                         0.3,
                         flatten_interface_ptr,
                         param.allocator_.get(),
                         param.thread_pool_.get(),
                         false);
    graph.Build();

    auto extract_graph = graph.GetGraph();

    float hit_edge_count = 0;
    for (int i = 0; i < num_vectors; ++i) {
        for (int j = 0; j < num_vectors; ++j) {
            if (i != j) {
                ground_truths[i].emplace_back(flatten_interface_ptr->ComputePairVectors(i, j), j);
            }
        }
        std::sort(ground_truths[i].begin(), ground_truths[i].end());
        std::vector<uint32_t> truths_edges(max_degree);
        for (int j = 0; j < max_degree; ++j) {
            truths_edges[j] = ground_truths[i][j].second;
        }
        hit_edge_count += calculate_overlap(truths_edges, extract_graph[i], max_degree);
    }
    REQUIRE(hit_edge_count / (num_vectors * max_degree) > 0.95);
}
