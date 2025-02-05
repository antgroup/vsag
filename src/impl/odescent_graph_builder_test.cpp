
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
#include <catch2/generators/catch_generators.hpp>
#include <filesystem>
#include <set>

#include "data_cell/flatten_interface.h"
#include "data_cell/graph_interface.h"
#include "fixtures.h"
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

TEST_CASE("ODescent Build Test", "[ut][ODescent]") {
    auto num_vectors = GENERATE(2, 4, 11, 2000);
    size_t dim = 128;
    int64_t max_degree = 32;
    auto partial_data = GENERATE(true, false);

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_vectors, dim);
    // prepare common param
    vsag::IndexCommonParam param;
    param.dim_ = dim;
    param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    param.thread_pool_ = vsag::SafeThreadPool::FactoryDefaultThreadPool();

    // prepare data param
    vsag::FlattenDataCellParamPtr flatten_param =
        std::make_shared<vsag::FlattenDataCellParameter>();
    flatten_param->quantizer_parameter_ = std::make_shared<vsag::FP32QuantizerParameter>();
    flatten_param->io_parameter_ = std::make_shared<vsag::MemoryIOParameter>();
    vsag::FlattenInterfacePtr flatten_interface_ptr =
        vsag::FlattenInterface::MakeInstance(flatten_param, param);
    flatten_interface_ptr->Train(vectors.data(), num_vectors);
    flatten_interface_ptr->BatchInsertVector(vectors.data(), num_vectors);

    // prepare graph param
    vsag::GraphDataCellParamPtr graph_param_ptr = std::make_shared<vsag::GraphDataCellParameter>();
    graph_param_ptr->io_parameter_ = std::make_shared<vsag::MemoryIOParameter>();
    graph_param_ptr->max_degree_ = partial_data ? 2 * max_degree : max_degree;
    // build graph
    vsag::ODescent graph(max_degree,
                         1,
                         30,
                         0.3,
                         flatten_interface_ptr,
                         param.allocator_.get(),
                         param.thread_pool_.get(),
                         false);
    std::shared_ptr<uint32_t[]> valid_ids = nullptr;
    if (partial_data) {
        num_vectors /= 2;
        valid_ids.reset(new uint32_t[num_vectors]);
        for (int i = 0; i < num_vectors; ++i) {
            valid_ids[i] = 2 * i;
        }
    }
    if (num_vectors <= 0) {
        REQUIRE_THROWS(graph.Build(valid_ids.get(), num_vectors));
        return;
    }
    graph.Build(valid_ids.get(), num_vectors);

    // check result
    vsag::GraphInterfacePtr graph_interface = nullptr;
    graph_interface = vsag::GraphInterface::MakeInstance(graph_param_ptr, param, partial_data);
    graph.SaveGraph(graph_interface);

    auto id_map = [&](uint32_t id) -> uint32_t { return partial_data ? valid_ids[id] : id; };

    if (num_vectors == 1) {
        REQUIRE(graph_interface->TotalCount() == 1);
        REQUIRE(graph_interface->GetNeighborSize(id_map(0)) == 0);
        return;
    }

    float hit_edge_count = 0;
    int64_t indeed_max_degree = std::min(max_degree, (int64_t)num_vectors - 1);
    for (int i = 0; i < num_vectors; ++i) {
        std::vector<std::pair<float, uint32_t>> ground_truths;
        uint32_t i_id = id_map(i);
        for (int j = 0; j < num_vectors; ++j) {
            uint32_t j_id = id_map(j);
            if (i_id != j_id) {
                ground_truths.emplace_back(flatten_interface_ptr->ComputePairVectors(i_id, j_id),
                                           j_id);
            }
        }
        std::sort(ground_truths.begin(), ground_truths.end());
        std::vector<uint32_t> truths_edges(indeed_max_degree);
        for (int j = 0; j < indeed_max_degree; ++j) {
            truths_edges[j] = ground_truths[j].second;
        }
        vsag::Vector<uint32_t> edges(param.allocator_.get());
        graph_interface->GetNeighbors(i_id, edges);
        hit_edge_count += calculate_overlap(truths_edges, edges, indeed_max_degree);
    }
    REQUIRE(hit_edge_count / (num_vectors * indeed_max_degree) > 0.95);
}
