
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

#include "flatten_datacell.h"

#include <algorithm>
#include <random>

#include "catch2/catch_template_test_macros.hpp"
#include "default_allocator.h"
#include "fixtures.h"
#include "io/memory_io.h"
#include "quantization/fp32_quantizer.h"
#include "quantization/sq8_quantizer.h"

using namespace vsag;

void
TestFlattenDataCell(const std::shared_ptr<FlattenInterface>& data_cell,
                    uint64_t dim,
                    MetricType metric,
                    float error = 1e-5) {
    int64_t base_size = 1000;
    int64_t query_size = 10;
    auto vectors = fixtures::generate_vectors(base_size, dim);
    auto querys = fixtures::generate_vectors(query_size, dim);

    data_cell->Train(vectors.data(), base_size);
    data_cell->BatchInsertVector(vectors.data(), base_size);

    auto func = [&](FlattenInterface* vs) {
        std::vector<uint64_t> idx(base_size);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), std::mt19937(std::random_device()()));
        std::vector<float> dists(base_size);
        for (auto i = 0; i < query_size; ++i) {
            auto computer = vs->FactoryComputer(querys.data() + i * dim);
            vs->Query(dists.data(), computer, idx.data(), base_size);
            float gt;
            for (auto j = 0; j < base_size; ++j) {
                if (metric == vsag::MetricType::METRIC_TYPE_IP ||
                    metric == vsag::MetricType::METRIC_TYPE_COSINE) {
                    gt = InnerProduct(vectors.data() + idx[j] * dim, querys.data() + i * dim, &dim);
                } else if (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
                    gt = L2Sqr(vectors.data() + idx[j] * dim, querys.data() + i * dim, &dim);
                }
                REQUIRE(std::abs(gt - dists[j]) < error);
            }
        }
    };

    func(data_cell.get());
    REQUIRE(data_cell->TotalCount() == base_size);
}

TEST_CASE("fp32[ut][flatten_data_cell]") {
    int dim = 32;
    auto alloctor = new DefaultAllocator();
    {
        auto data_cell = std::make_shared<FlattenDataCell<FP32Quantizer<>, MemoryIO>>();
        data_cell->SetQuantizer(std::make_shared<FP32Quantizer<>>(dim));
        data_cell->SetIO(std::make_shared<MemoryIO>(alloctor));
        TestFlattenDataCell(data_cell, dim, vsag::MetricType::METRIC_TYPE_L2SQR);
    }
    {
        auto data_cell = std::make_shared<
            FlattenDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_IP>, MemoryIO>>();
        data_cell->SetQuantizer(
            std::make_shared<FP32Quantizer<vsag::MetricType::METRIC_TYPE_IP>>(dim));
        data_cell->SetIO(std::make_shared<MemoryIO>(alloctor));
        TestFlattenDataCell(data_cell, dim, vsag::MetricType::METRIC_TYPE_IP);
    }
    {
        auto data_cell = std::make_shared<
            FlattenDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>, MemoryIO>>();
        data_cell->SetQuantizer(
            std::make_shared<FP32Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>>(dim));
        data_cell->SetIO(std::make_shared<MemoryIO>(alloctor));
        TestFlattenDataCell(data_cell, dim, vsag::MetricType::METRIC_TYPE_COSINE);
    }
    delete alloctor;
}

TEST_CASE("sq8[ut][flatten_data_cell]") {
    int dim = 32;
    auto alloctor = new DefaultAllocator();
    {
        auto data_cell = std::make_shared<FlattenDataCell<SQ8Quantizer<>, MemoryIO>>();
        data_cell->SetQuantizer(std::make_shared<SQ8Quantizer<>>(dim));
        data_cell->SetIO(std::make_shared<MemoryIO>(alloctor));
        TestFlattenDataCell(data_cell, dim, vsag::MetricType::METRIC_TYPE_L2SQR, 0.01);
    }
    {
        auto data_cell = std::make_shared<
            FlattenDataCell<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_IP>, MemoryIO>>();
        data_cell->SetQuantizer(
            std::make_shared<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_IP>>(dim));
        data_cell->SetIO(std::make_shared<MemoryIO>(alloctor));
        TestFlattenDataCell(data_cell, dim, vsag::MetricType::METRIC_TYPE_IP, 0.01);
    }
    {
        auto data_cell = std::make_shared<
            FlattenDataCell<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>, MemoryIO>>();
        data_cell->SetQuantizer(
            std::make_shared<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>>(dim));
        data_cell->SetIO(std::make_shared<MemoryIO>(alloctor));
        TestFlattenDataCell(data_cell, dim, vsag::MetricType::METRIC_TYPE_COSINE, 0.01);
    }
    delete alloctor;
}
