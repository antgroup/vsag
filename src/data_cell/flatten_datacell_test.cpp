
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

#include "catch2/catch_template_test_macros.hpp"
#include "default_allocator.h"
#include "fixtures.h"
#include "flatten_interface_test.h"
#include "io/io_headers.h"
#include "quantization/quantizer_headers.h"

using namespace vsag;

template <typename QuantTmpl, typename IOTmpl, MetricType metric>
void
TestFlattenDataCell(int dim,
                    std::shared_ptr<Allocator> allocator,
                    const JsonType& quantizer_json,
                    const JsonType& io_json,
                    float error = 1e-5) {
    auto counts = {100, 1000};
    IndexCommonParam common;
    common.dim_ = dim;
    common.allocator_ = allocator.get();
    common.metric_ = metric;
    for (auto count : counts) {
        auto flatten =
            std::make_shared<FlattenDataCell<QuantTmpl, IOTmpl>>(quantizer_json, io_json, common);
        FlattenInterfaceTest test(flatten, metric);
        test.BasicTest(dim, count, error);
        auto other =
            std::make_shared<FlattenDataCell<QuantTmpl, IOTmpl>>(quantizer_json, io_json, common);
        test.TestSerializeAndDeserialize(dim, other, error);
    }
}

template <typename IOTmpl>
void
TestFlattenDataCellFP32(int dim,
                        std::shared_ptr<Allocator> allocator,
                        const JsonType& quantizer_json,
                        const JsonType& io_json,
                        float error = 1e-5) {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    TestFlattenDataCell<FP32Quantizer<metrics[0]>, IOTmpl, metrics[0]>(
        dim, allocator, quantizer_json, io_json, error);
    TestFlattenDataCell<FP32Quantizer<metrics[1]>, IOTmpl, metrics[1]>(
        dim, allocator, quantizer_json, io_json, error);
    TestFlattenDataCell<FP32Quantizer<metrics[2]>, IOTmpl, metrics[2]>(
        dim, allocator, quantizer_json, io_json, error);
}

TEST_CASE("fp32", "[ut][flatten_data_cell]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    auto fp32_param = JsonType::parse("{}");
    auto io_param = JsonType::parse("{}");
    auto dims = {8, 64, 512};
    float error = 1e-5;
    for (auto dim : dims) {
        TestFlattenDataCellFP32<MemoryIO>(dim, allocator, fp32_param, io_param, error);
        TestFlattenDataCellFP32<MemoryBlockIO>(dim, allocator, fp32_param, io_param, error);
    }
}

template <typename IOTmpl>
void
TestFlattenDataCellSQ8(int dim,
                       std::shared_ptr<Allocator> allocator,
                       const JsonType& quantizer_json,
                       const JsonType& io_json,
                       float error = 1e-5) {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    TestFlattenDataCell<SQ8Quantizer<metrics[0]>, IOTmpl, metrics[0]>(
        dim, allocator, quantizer_json, io_json, error);
    TestFlattenDataCell<SQ8Quantizer<metrics[1]>, IOTmpl, metrics[1]>(
        dim, allocator, quantizer_json, io_json, error);
    TestFlattenDataCell<SQ8Quantizer<metrics[2]>, IOTmpl, metrics[2]>(
        dim, allocator, quantizer_json, io_json, error);
}

TEST_CASE("sq8", "[ut][flatten_data_cell]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    auto sq8_param = JsonType::parse("{}");
    auto io_param = JsonType::parse("{}");
    auto dims = {32, 64, 512};
    auto error = 2e-2f;
    for (auto dim : dims) {
        TestFlattenDataCellSQ8<MemoryIO>(dim, allocator, sq8_param, io_param, error);
        TestFlattenDataCellSQ8<MemoryBlockIO>(dim, allocator, sq8_param, io_param, error);
    }
}
