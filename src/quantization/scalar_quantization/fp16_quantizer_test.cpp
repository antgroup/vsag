
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

#include "fp16_quantizer.h"

#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"
#include "quantization/quantizer_test.h"

using namespace vsag;

const auto dims = fixtures::get_common_used_dims(3, 225);
const auto counts = {10, 101};

template <MetricType metric>
void
TestQuantizerEncodeDecodeMetricFP16(uint64_t dim, int count, float error = 1e-3) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    FP16Quantizer<metric> quantizer(dim, allocator.get());
    TestQuantizerEncodeDecode(quantizer, dim, count, error);
    TestQuantizerEncodeDecodeSame(quantizer, dim, count, 65536, error);
}

TEST_CASE("FP16 Encode and Decode", "[ut][FP16Quantizer]") {
    constexpr MetricType metrics[2] = {MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP};
    float error = 2e-3F;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestQuantizerEncodeDecodeMetricFP16<metrics[0]>(dim, count, error);
            TestQuantizerEncodeDecodeMetricFP16<metrics[1]>(dim, count, error);
        }
    }
}

template <MetricType metric>
void
TestComputeMetricFP16(uint64_t dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    FP16Quantizer<metric> quantizer(dim, allocator.get());
    TestComputeCodes<FP16Quantizer<metric>, metric>(quantizer, dim, count, error);
    TestComputer<FP16Quantizer<metric>, metric>(quantizer, dim, count, error);
}

TEST_CASE("FP16 Compute", "[ut][FP16Quantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    float error = 2e-3F;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestComputeMetricFP16<metrics[0]>(dim, count, error);
            TestComputeMetricFP16<metrics[1]>(dim, count, error);
            TestComputeMetricFP16<metrics[2]>(dim, count, error);
        }
    }
}

template <MetricType metric>
void
TestSerializeAndDeserializeMetricFP16(uint64_t dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    FP16Quantizer<metric> quantizer1(dim, allocator.get());
    FP16Quantizer<metric> quantizer2(0, allocator.get());
    TestSerializeAndDeserialize<FP16Quantizer<metric>, metric>(
        quantizer1, quantizer2, dim, count, error);
}

TEST_CASE("FP16 Serialize and Deserialize", "[ut][FP16Quantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    float error = 2e-3F;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestSerializeAndDeserializeMetricFP16<metrics[0]>(dim, count, error);
            TestSerializeAndDeserializeMetricFP16<metrics[1]>(dim, count, error);
            TestSerializeAndDeserializeMetricFP16<metrics[2]>(dim, count, error);
        }
    }
}
