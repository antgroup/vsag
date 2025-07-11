
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

#include "fp32_quantizer.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "fixtures.h"
#include "impl/allocator/default_allocator.h"
#include "impl/allocator/safe_allocator.h"
#include "quantizer_test.h"

using namespace vsag;

const auto dims = {64, 128};
const auto counts = {10, 101};

template <MetricType metric>
void
TestQuantizerEncodeDecodeMetricFP32(uint64_t dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    FP32Quantizer<metric> quantizer(dim, allocator.get());
    TestQuantizerEncodeDecode(quantizer, dim, count, error);
    TestQuantizerEncodeDecodeSame(quantizer, dim, count, 65536, error);
}

TEST_CASE("FP32 Encode and Decode", "[ut][FP32Quantizer]") {
    constexpr MetricType metrics[2] = {MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP};
    float error = 2e-5f;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestQuantizerEncodeDecodeMetricFP32<metrics[0]>(dim, count, error);
            TestQuantizerEncodeDecodeMetricFP32<metrics[1]>(dim, count, error);
        }
    }
}

template <MetricType metric>
void
TestComputeMetricFP32(uint64_t dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    FP32Quantizer<metric> quantizer(dim, allocator.get());
    TestComputeCodes<FP32Quantizer<metric>, metric>(quantizer, dim, count, error);
    TestComputeCodesSame<FP32Quantizer<metric>, metric>(quantizer, dim, count, 65536);
    TestComputer<FP32Quantizer<metric>, metric>(quantizer, dim, count, error);
}

TEST_CASE("FP32 Compute", "[ut][FP32Quantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    float error = 2e-5f;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestComputeMetricFP32<metrics[0]>(dim, count, error);
            TestComputeMetricFP32<metrics[1]>(dim, count, error);
            TestComputeMetricFP32<metrics[2]>(dim, count, error);
        }
    }
}

template <MetricType metric>
void
TestSerializeAndDeserializeMetricFP32(uint64_t dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    FP32Quantizer<metric> quantizer1(dim, allocator.get());
    FP32Quantizer<metric> quantizer2(0, allocator.get());
    TestSerializeAndDeserialize<FP32Quantizer<metric>, metric>(
        quantizer1, quantizer2, dim, count, error);
}

TEST_CASE("FP32 Serialize and Deserialize", "[ut][FP32Quantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    float error = 2e-5f;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestSerializeAndDeserializeMetricFP32<metrics[0]>(dim, count, error);
            TestSerializeAndDeserializeMetricFP32<metrics[1]>(dim, count, error);
            TestSerializeAndDeserializeMetricFP32<metrics[2]>(dim, count, error);
        }
    }
}
