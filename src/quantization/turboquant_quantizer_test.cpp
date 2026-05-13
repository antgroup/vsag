
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

#include "turboquant_quantizer.h"

#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"
#include "quantizer_test.h"

using namespace vsag;

const auto turboquant_dims = {32, 64};
const auto turboquant_counts = {10, 101};

template <MetricType metric>
void
TestQuantizerEncodeDecodeMetricTurboQuant(uint64_t dim, int count) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    TurboQuantizer<metric> quantizer(dim, 6, true, true, dim, allocator.get());
    TestQuantizerEncodeDecode(quantizer, dim, count, 2.5F, true);
}

TEST_CASE("TurboQuant Encode and Decode", "[ut][TurboQuantQuantizer]") {
    constexpr MetricType metrics[2] = {MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP};
    for (auto dim : turboquant_dims) {
        for (auto count : turboquant_counts) {
            TestQuantizerEncodeDecodeMetricTurboQuant<metrics[0]>(dim, count);
            TestQuantizerEncodeDecodeMetricTurboQuant<metrics[1]>(dim, count);
        }
    }
}

template <MetricType metric>
void
TestComputeMetricTurboQuant(uint64_t dim, int count) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    TurboQuantizer<metric> quantizer(dim, 6, true, true, dim, allocator.get());
    TestComputer<TurboQuantizer<metric>, metric>(
        quantizer, dim, count, 20.0F, 2.0F, true, 1.0F, 1.0F);
}

TEST_CASE("TurboQuant Compute", "[ut][TurboQuantQuantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    for (auto dim : turboquant_dims) {
        for (auto count : turboquant_counts) {
            TestComputeMetricTurboQuant<metrics[0]>(dim, count);
            TestComputeMetricTurboQuant<metrics[1]>(dim, count);
            TestComputeMetricTurboQuant<metrics[2]>(dim, count);
        }
    }
}

template <MetricType metric>
void
TestSerializeAndDeserializeMetricTurboQuant(uint64_t dim, int count) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    TurboQuantizer<metric> quantizer1(dim, 6, false, true, dim / 2, allocator.get());
    TurboQuantizer<metric> quantizer2(dim, 6, false, true, dim / 2, allocator.get());
    TestSerializeAndDeserialize<TurboQuantizer<metric>, metric>(
        quantizer1, quantizer2, dim, count, 5.0F, 2.0F, 1.0F, 1.0F);
}

TEST_CASE("TurboQuant Serialize and Deserialize", "[ut][TurboQuantQuantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    for (auto dim : turboquant_dims) {
        for (auto count : turboquant_counts) {
            TestSerializeAndDeserializeMetricTurboQuant<metrics[0]>(dim, count);
            TestSerializeAndDeserializeMetricTurboQuant<metrics[1]>(dim, count);
            TestSerializeAndDeserializeMetricTurboQuant<metrics[2]>(dim, count);
        }
    }
}
