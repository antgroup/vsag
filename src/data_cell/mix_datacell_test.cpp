
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

#include "mix_datacell.h"

#include "catch2/catch_template_test_macros.hpp"
#include "default_allocator.h"
#include "fixtures.h"
#include "io/memory_io.h"
#include "quantization/fp32_quantizer.h"
#include "quantization/sq8_quantizer.h"

using namespace vsag;

template <typename QuantT, typename IOT, typename GraphT>
void
TestMixDataCellBasicUsage(std::unique_ptr<MixDataCell<QuantT, IOT, GraphT>>& data_cell,
                          uint64_t dim,
                          MetricType metric,
                          float error = 1e-5) {
    uint64_t base_size = 1000;
    uint64_t query_size = 10;
    auto vectors = fixtures::generate_vectors(base_size, dim);
    auto queries = fixtures::generate_vectors(query_size, dim);

    auto func = [&]() {
        std::vector<uint64_t> idx(base_size);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), std::mt19937(std::random_device()()));
        std::vector<float> dists(base_size);
        for (auto i = 0; i < query_size; ++i) {
            auto computer = data_cell->FactoryComputer(queries.data() + i * dim);
            data_cell->Query(dists.data(), computer, idx.data(), base_size);
            float gt;
            for (auto j = 0; j < base_size; ++j) {
                if (metric == vsag::MetricType::METRIC_TYPE_IP ||
                    metric == vsag::MetricType::METRIC_TYPE_COSINE) {
                    gt =
                        InnerProduct(vectors.data() + idx[j] * dim, queries.data() + i * dim, &dim);
                } else if (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
                    gt = L2Sqr(vectors.data() + idx[j] * dim, queries.data() + i * dim, &dim);
                }
                REQUIRE(std::fabs(gt - dists[j]) < error);
            }
        }
    };
    data_cell->TrainQuantizer(vectors.data(), base_size);
    data_cell->BatchInsertVector(vectors.data(), base_size);
    func();
    REQUIRE(data_cell->TotalCount() == base_size);
}

uint32_t
generate_to_be_visited(uint32_t neighbor_size, std::vector<uint32_t>& to_be_visit) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, neighbor_size + 1);
    uint32_t count_no_visit = dist(gen);
    if (count_no_visit >= neighbor_size) {
        if (neighbor_size != 0) {
            count_no_visit = neighbor_size - 1;
        } else {
            count_no_visit = 0;
        }
    }

    for (uint32_t i = 0; i < count_no_visit; ++i) {
        auto idx = dist(gen);
        if (idx >= neighbor_size) {
            if (neighbor_size != 0) {
                idx = neighbor_size - 1;
            } else {
                idx = 0;
            }
        }
        to_be_visit[i] = idx;
    }

    return count_no_visit;
}

template <typename QuantT, typename IOT, typename GraphT>
void
TestMixDataCellBasicRedundant(std::unique_ptr<MixDataCell<QuantT, IOT, GraphT>>& data_cell,
                              std::shared_ptr<GraphT> graph,
                              uint64_t dim,
                              uint32_t M,
                              MetricType metric,
                              float error = 1e-5) {
    uint64_t base_size = 1000;
    uint64_t query_size = 10;
    double redundant_rate = 0.3;
    auto vectors = fixtures::generate_vectors(base_size, dim);
    auto queries = fixtures::generate_vectors(query_size, dim);
    std::vector<uint32_t> to_be_visited(M, 0);

    data_cell->TrainQuantizer(vectors.data(), base_size);
    data_cell->BatchInsertVector(vectors.data(), base_size);
    REQUIRE(data_cell->TotalCount() == base_size);

    data_cell->MakeRedundant(redundant_rate);
    REQUIRE(data_cell->GetRedundantTotalCount() ==
            (uint64_t)(redundant_rate * data_cell->TotalCount()));

    for (int i = 0; i < base_size; i++) {
        std::vector<uint64_t> neighbor_ids;
        uint64_t neighbor_id;
        graph->GetNeighbors(i, neighbor_ids);
        std::vector<float> dists(neighbor_ids.size());

        for (int j = 0; j < query_size; j++) {
            auto query = queries.data() + j * dim;
            auto count_no_visit = generate_to_be_visited(neighbor_ids.size(), to_be_visited);
            if (not data_cell->IsRedundant(i)) {
                for (int k = 0; k < count_no_visit; k++) {
                    to_be_visited[k] = neighbor_ids[to_be_visited[k]];
                }
            }

            std::unique_ptr<Computer<QuantT>> computer =
                std::move(data_cell->FactoryComputer(query));
            data_cell->QueryLine(dists.data(), computer, i, to_be_visited, count_no_visit);

            for (int k = 0; k < count_no_visit; k++) {
                if (data_cell->IsRedundant(i)) {
                    neighbor_id = neighbor_ids[to_be_visited[k]];
                } else {
                    neighbor_id = to_be_visited[k];
                }
                float gt = 0;
                if (metric == vsag::MetricType::METRIC_TYPE_IP ||
                    metric == vsag::MetricType::METRIC_TYPE_COSINE) {
                    gt = InnerProduct(vectors.data() + neighbor_id * dim, query, &dim);
                } else if (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
                    gt = L2Sqr(vectors.data() + neighbor_id * dim, query, &dim);
                }
                REQUIRE(std::fabs(gt - dists[k]) < error);
            }
        }
    }
}

TEST_CASE("fp32 basic usage in mix data cell", "[ut][flatten_storage]") {
    int dim = 960;
    int M = 32;
    auto allocator = new DefaultAllocator();
    auto graph_io = std::make_shared<MemoryIO>(allocator);
    auto graph_data_cell = std::make_shared<GraphDataCell<MemoryIO, false>>(M);
    {
        auto data_cell =
            std::make_unique<MixDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>,
                                         MemoryIO,
                                         GraphDataCell<MemoryIO, false>>>(graph_data_cell);
        data_cell->SetQuantizer(
            std::make_unique<FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>>(dim));
        data_cell->SetIO(std::make_unique<MemoryIO>(allocator),
                         std::make_unique<MemoryIO>(allocator));
        TestMixDataCellBasicUsage(data_cell, dim, vsag::MetricType::METRIC_TYPE_L2SQR);
    }
    {
        auto data_cell =
            std::make_unique<MixDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_IP>,
                                         MemoryIO,
                                         GraphDataCell<MemoryIO, false>>>(graph_data_cell);
        data_cell->SetQuantizer(
            std::make_unique<FP32Quantizer<vsag::MetricType::METRIC_TYPE_IP>>(dim));
        data_cell->SetIO(std::make_unique<MemoryIO>(allocator),
                         std::make_unique<MemoryIO>(allocator));
        TestMixDataCellBasicUsage(data_cell, dim, vsag::MetricType::METRIC_TYPE_IP);
    }
    {
        auto data_cell =
            std::make_unique<MixDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>,
                                         MemoryIO,
                                         GraphDataCell<MemoryIO, false>>>(graph_data_cell);
        data_cell->SetQuantizer(
            std::make_unique<FP32Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>>(dim));
        data_cell->SetIO(std::make_unique<MemoryIO>(allocator),
                         std::make_unique<MemoryIO>(allocator));
        TestMixDataCellBasicUsage(data_cell, dim, vsag::MetricType::METRIC_TYPE_COSINE);
    }
}

TEST_CASE("fp32 redundant usage in mix data cell", "[ut][flatten_storage]") {
    int dim = 960;
    int data_size = 1000;
    int M = 32;

    std::random_device rd;
    std::mt19937 gen(rd());
    auto allocator = new DefaultAllocator();
    auto graph_io = std::make_shared<MemoryIO>(allocator);
    auto graph_data_cell = std::make_shared<GraphDataCell<MemoryIO, false>>(M);
    graph_data_cell->SetIO(graph_io);
    for (int i = 0; i < data_size; i++) {
        std::vector<uint64_t> neighbor_ids(gen() % (M + 1));
        for (int j = 0; j < neighbor_ids.size(); j++) {
            neighbor_ids[j] = gen() % data_size;
        }

        graph_data_cell->InsertNode(neighbor_ids);
    }

    {
        auto data_cell =
            std::make_unique<MixDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>,
                                         MemoryIO,
                                         GraphDataCell<MemoryIO, false>>>(graph_data_cell);
        data_cell->SetQuantizer(
            std::make_unique<FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>>(dim));
        data_cell->SetIO(std::make_unique<MemoryIO>(allocator),
                         std::make_unique<MemoryIO>(allocator));
        TestMixDataCellBasicRedundant(
            data_cell, graph_data_cell, dim, M, vsag::MetricType::METRIC_TYPE_L2SQR);
    }
    {
        auto data_cell =
            std::make_unique<MixDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_IP>,
                                         MemoryIO,
                                         GraphDataCell<MemoryIO, false>>>(graph_data_cell);
        data_cell->SetQuantizer(
            std::make_unique<FP32Quantizer<vsag::MetricType::METRIC_TYPE_IP>>(dim));
        data_cell->SetIO(std::make_unique<MemoryIO>(allocator),
                         std::make_unique<MemoryIO>(allocator));
        TestMixDataCellBasicRedundant(
            data_cell, graph_data_cell, dim, M, vsag::MetricType::METRIC_TYPE_IP);
    }
    {
        auto data_cell =
            std::make_unique<MixDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>,
                                         MemoryIO,
                                         GraphDataCell<MemoryIO, false>>>(graph_data_cell);
        data_cell->SetQuantizer(
            std::make_unique<FP32Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>>(dim));
        data_cell->SetIO(std::make_unique<MemoryIO>(allocator),
                         std::make_unique<MemoryIO>(allocator));
        TestMixDataCellBasicRedundant(
            data_cell, graph_data_cell, dim, M, vsag::MetricType::METRIC_TYPE_COSINE);
    }
}

TEST_CASE("sq8 basic usage in mix data cell", "[ut][flatten_storage]") {
    int dim = 960;
    int M = 32;
    float error = 0.01;
    {
        auto allocator = new DefaultAllocator();
        auto graph_io = std::make_shared<MemoryIO>(allocator);
        auto graph_data_cell = std::make_shared<GraphDataCell<MemoryIO, false>>(M);
        graph_data_cell->SetIO(graph_io);

        auto data_cell =
            std::make_unique<MixDataCell<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>,
                                         MemoryIO,
                                         GraphDataCell<MemoryIO, false>>>(graph_data_cell);
        data_cell->SetQuantizer(
            std::make_unique<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>>(dim));
        data_cell->SetIO(std::make_unique<MemoryIO>(allocator),
                         std::make_unique<MemoryIO>(allocator));
        TestMixDataCellBasicUsage(data_cell, dim, vsag::MetricType::METRIC_TYPE_L2SQR, error);
    }
    {
        auto allocator = new DefaultAllocator();
        auto graph_io = std::make_shared<MemoryIO>(allocator);
        auto graph_data_cell = std::make_shared<GraphDataCell<MemoryIO, false>>(M);
        graph_data_cell->SetIO(graph_io);

        auto data_cell =
            std::make_unique<MixDataCell<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_IP>,
                                         MemoryIO,
                                         GraphDataCell<MemoryIO, false>>>(graph_data_cell);
        data_cell->SetQuantizer(
            std::make_unique<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_IP>>(dim));
        data_cell->SetIO(std::make_unique<MemoryIO>(allocator),
                         std::make_unique<MemoryIO>(allocator));
        TestMixDataCellBasicUsage(data_cell, dim, vsag::MetricType::METRIC_TYPE_IP, error);
    }
    {
        auto allocator = new DefaultAllocator();

        auto graph_io = std::make_shared<MemoryIO>(allocator);
        auto graph_data_cell = std::make_shared<GraphDataCell<MemoryIO, false>>(M);
        graph_data_cell->SetIO(graph_io);

        auto data_cell =
            std::make_unique<MixDataCell<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>,
                                         MemoryIO,
                                         GraphDataCell<MemoryIO, false>>>(graph_data_cell);
        data_cell->SetQuantizer(
            std::make_unique<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>>(dim));
        data_cell->SetIO(std::make_unique<MemoryIO>(allocator),
                         std::make_unique<MemoryIO>(allocator));
        TestMixDataCellBasicUsage(data_cell, dim, vsag::MetricType::METRIC_TYPE_COSINE, error);
    }
}

TEST_CASE("sq8 redundant usage in mix data cell", "[ut][flatten_storage]") {
    int dim = 960;
    int data_size = 1000;
    int M = 32;
    float error = 0.01;

    std::random_device rd;
    std::mt19937 gen(rd());
    auto allocator = new DefaultAllocator();
    auto graph_io = std::make_shared<MemoryIO>(allocator);
    auto graph_data_cell = std::make_shared<GraphDataCell<MemoryIO, false>>(M);
    graph_data_cell->SetIO(graph_io);
    for (int i = 0; i < data_size; i++) {
        std::vector<uint64_t> neighbor_ids(gen() % (M + 1));
        for (int j = 0; j < neighbor_ids.size(); j++) {
            neighbor_ids[j] = gen() % data_size;
        }

        graph_data_cell->InsertNode(neighbor_ids);
    }

    {
        auto data_cell =
            std::make_unique<MixDataCell<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>,
                                         MemoryIO,
                                         GraphDataCell<MemoryIO, false>>>(graph_data_cell);
        data_cell->SetQuantizer(
            std::make_unique<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>>(dim));
        data_cell->SetIO(std::make_unique<MemoryIO>(allocator),
                         std::make_unique<MemoryIO>(allocator));
        TestMixDataCellBasicRedundant(
            data_cell, graph_data_cell, dim, M, vsag::MetricType::METRIC_TYPE_L2SQR, error);
    }
    {
        auto data_cell =
            std::make_unique<MixDataCell<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_IP>,
                                         MemoryIO,
                                         GraphDataCell<MemoryIO, false>>>(graph_data_cell);
        data_cell->SetQuantizer(
            std::make_unique<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_IP>>(dim));
        data_cell->SetIO(std::make_unique<MemoryIO>(allocator),
                         std::make_unique<MemoryIO>(allocator));
        TestMixDataCellBasicRedundant(
            data_cell, graph_data_cell, dim, M, vsag::MetricType::METRIC_TYPE_IP, error);
    }
    {
        auto data_cell =
            std::make_unique<MixDataCell<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>,
                                         MemoryIO,
                                         GraphDataCell<MemoryIO, false>>>(graph_data_cell);
        data_cell->SetQuantizer(
            std::make_unique<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>>(dim));
        data_cell->SetIO(std::make_unique<MemoryIO>(allocator),
                         std::make_unique<MemoryIO>(allocator));
        TestMixDataCellBasicRedundant(
            data_cell, graph_data_cell, dim, M, vsag::MetricType::METRIC_TYPE_COSINE, error);
    }
}