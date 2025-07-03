
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

#include "sq8_uniform_quantizer.h"

#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"
#include "quantization/quantizer_test.h"

using namespace vsag;

const auto dims = fixtures::get_common_used_dims();
const auto counts = {10, 101};

template <MetricType metric>
void
TestQuantizerEncodeDecodeMetricSQ8Uniform(uint64_t dim,
                                          int count,
                                          float error = 1e-5,
                                          float error_same = 1e-2) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    SQ8UniformQuantizer<metric> quantizer(dim, allocator.get());
    TestQuantizerEncodeDecode(quantizer, dim, count, error);
    TestQuantizerEncodeDecodeSame(quantizer, dim, count, 255, error_same);
}

TEST_CASE("SQ8 Uniform Encode and Decode", "[ut][SQ8UniformQuantizer]") {
    constexpr MetricType metrics[2] = {MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP};
    float error = 2 * 1.0f / 255.0f;
    for (auto dim : dims) {
        for (auto count : counts) {
            auto error_same = (float)(dim * 255 * 0.01);
            TestQuantizerEncodeDecodeMetricSQ8Uniform<metrics[0]>(dim, count, error, error_same);
            TestQuantizerEncodeDecodeMetricSQ8Uniform<metrics[1]>(dim, count, error, error_same);
        }
    }
}

template <MetricType metric>
void
TestComputeMetricSQ8Uniform(uint64_t dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    SQ8UniformQuantizer<metric> quantizer(dim, allocator.get());
    TestComputeCodesSame<SQ8UniformQuantizer<metric>, metric>(quantizer, dim, count, error);
}

TEST_CASE("SQ8 Uniform Compute", "[ut][SQ8UniformQuantizer]") {
    constexpr MetricType metrics[2] = {MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP};
    float error = 4 * 1.0f / 255.0f;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestComputeMetricSQ8Uniform<metrics[0]>(dim, count, error);
            TestComputeMetricSQ8Uniform<metrics[1]>(dim, count, error);
        }
    }
}

template <MetricType metric>
void
TestSerializeAndDeserializeMetricSQ8Uniform(uint64_t dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    SQ8UniformQuantizer<metric> quantizer1(dim, allocator.get());
    SQ8UniformQuantizer<metric> quantizer2(0, allocator.get());
    TestSerializeAndDeserialize<SQ8UniformQuantizer<metric>, metric, true>(
        quantizer1, quantizer2, dim, count, error);
}

TEST_CASE("SQ8 Uniform Serialize and Deserialize", "[ut][SQ8UniformQuantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    for (auto dim : dims) {
        float error = 4 * 1.0f / 255.0f;
        for (auto count : counts) {
            TestSerializeAndDeserializeMetricSQ8Uniform<metrics[0]>(dim, count, error);
            //            TestSerializeAndDeserializeMetricSQ8Uniform<metrics[1]>(dim, count, error);
            TestSerializeAndDeserializeMetricSQ8Uniform<metrics[2]>(dim, count, error);
        }
    }
}
