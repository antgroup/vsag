
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
#include "impl/allocator/safe_allocator.h"
#include "io/memory_io_parameter.h"
#include "quantization/fp32_quantizer_parameter.h"

size_t
calculate_overlap(const vsag::Vector<uint32_t>& vec1, const vsag::Vector<uint32_t>& vec2, int K) {
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
    auto use_thread_pool = GENERATE(true);

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_vectors, dim);
    // prepare common param
    vsag::IndexCommonParam param;
    param.dim_ = dim;
    param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto thread_pool = vsag::Engine::CreateThreadPool(4);
    if (use_thread_pool) {
        param.thread_pool_ = std::make_shared<vsag::SafeThreadPool>(thread_pool->get(), false);
    } else {
        param.thread_pool_ = nullptr;
    }

    // prepare data param
    constexpr const char* graph_param_temp =
        R"(
        {{
            "io_params": {{
                "type": "block_memory_io"
            }},
            "max_degree": {}
        }}
        )";
    auto param_str = fmt::format(graph_param_temp, max_degree);
    auto graph_param_json = vsag::JsonType::parse(param_str);

    vsag::FlattenDataCellParamPtr flatten_param =
        std::make_shared<vsag::FlattenDataCellParameter>();
    flatten_param->quantizer_parameter = std::make_shared<vsag::FP32QuantizerParameter>();
    flatten_param->io_parameter = std::make_shared<vsag::MemoryIOParameter>();
    vsag::FlattenInterfacePtr flatten_interface_ptr =
        vsag::FlattenInterface::MakeInstance(flatten_param, param);
    flatten_interface_ptr->Train(vectors.data(), num_vectors);
    flatten_interface_ptr->BatchInsertVector(vectors.data(), num_vectors);

    // prepare graph param
    auto graph_type = partial_data ? vsag::GraphStorageTypes::GRAPH_STORAGE_TYPE_SPARSE
                                   : vsag::GraphStorageTypes::GRAPH_STORAGE_TYPE_FLAT;
    vsag::GraphInterfaceParamPtr graph_param_ptr =
        vsag::GraphInterfaceParameter::GetGraphParameterByJson(graph_type, graph_param_json);
    // build graph
    auto odescent_param = std::make_shared<vsag::ODescentParameter>();
    odescent_param->max_degree = max_degree;
    vsag::ODescent graph(odescent_param,
                         flatten_interface_ptr,
                         param.allocator_.get(),
                         param.thread_pool_.get(),
                         false);
    vsag::Vector<vsag::InnerIdType> valid_ids(param.allocator_.get());
    if (partial_data) {
        num_vectors /= 2;
        valid_ids.resize(num_vectors);
        for (int i = 0; i < num_vectors; ++i) {
            valid_ids[i] = 2 * i;
        }
    }
    if (num_vectors <= 0) {
        REQUIRE_THROWS(graph.Build(valid_ids));
        return;
    }

    auto id_map = [&](uint32_t id) -> uint32_t { return partial_data ? valid_ids[id] : id; };

    // check result
    vsag::GraphInterfacePtr graph_interface =
        vsag::GraphInterface::MakeInstance(graph_param_ptr, param);
    vsag::GraphInterfacePtr half_graph_interface =
        vsag::GraphInterface::MakeInstance(graph_param_ptr, param);
    vsag::GraphInterfacePtr merged_graph_interface =
        vsag::GraphInterface::MakeInstance(graph_param_ptr, param);
    graph.Build(valid_ids);
    graph.SaveGraph(graph_interface);

    for (int i = 0; i < num_vectors; ++i) {
        auto id = id_map(i);
        vsag::Vector<vsag::InnerIdType> edges(param.allocator_.get());
        graph_interface->GetNeighbors(id, edges);
        auto origin_size = edges.size();
        edges.resize(origin_size / 2);
        half_graph_interface->InsertNeighborsById(id, edges);
    }
    graph.Build(valid_ids, half_graph_interface);
    graph.SaveGraph(merged_graph_interface);

    if (num_vectors == 1) {
        REQUIRE(graph_interface->TotalCount() == 1);
        REQUIRE(graph_interface->GetNeighborSize(id_map(0)) == 0);
        return;
    }

    float hit_edge_count = 0;
    float hit_edge_count_merge = 0;
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
        vsag::Vector<uint32_t> truths_edges(indeed_max_degree, param.allocator_.get());
        for (int j = 0; j < indeed_max_degree; ++j) {
            truths_edges[j] = ground_truths[j].second;
        }
        vsag::Vector<uint32_t> edges(param.allocator_.get());
        vsag::Vector<uint32_t> edges_merged(param.allocator_.get());
        graph_interface->GetNeighbors(i_id, edges);
        merged_graph_interface->GetNeighbors(i_id, edges_merged);
        REQUIRE(edges.size() == indeed_max_degree);
        REQUIRE(edges_merged.size() == indeed_max_degree);
        hit_edge_count += calculate_overlap(truths_edges, edges, indeed_max_degree);
        hit_edge_count_merge += calculate_overlap(truths_edges, edges, indeed_max_degree);
    }
    REQUIRE(hit_edge_count / (num_vectors * indeed_max_degree) > 0.95);
    REQUIRE(hit_edge_count_merge >= hit_edge_count);
}
