
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

#include "sq8_quantizer.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "default_allocator.h"
#include "fixtures.h"
#include "quantizer_test.h"

using namespace vsag;

const auto counts = {10, 101};

template <MetricType metric>
void
TestQuantizerEncodeDecodeMetricSQ8(uint64_t dim,
                                   int count,
                                   float error = 1e-5,
                                   float error_same = 1e-2) {
    auto allocator = std::make_shared<DefaultAllocator>();
    SQ8Quantizer<metric> quantizer(dim, allocator.get());
    TestQuantizerEncodeDecode(quantizer, dim, count, error);
    TestQuantizerEncodeDecodeSame(quantizer, dim, count, 255, error_same);
}

TEST_CASE("encode&decode [SQ8Quantizer]") {
    auto dims = fixtures::get_common_used_dims();
    constexpr MetricType metrics[2] = {MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP};
    float error = 1e-2f;
    for (auto dim : dims) {
        for (auto count : counts) {
            auto error_same = (float)(dim * 255 * 0.01);
            TestQuantizerEncodeDecodeMetricSQ8<metrics[0]>(dim, count, error, error_same);
            TestQuantizerEncodeDecodeMetricSQ8<metrics[1]>(dim, count, error, error_same);
        }
    }
}

template <MetricType metric>
void
TestComputeMetricSQ8(uint64_t dim, int count, float error = 1e-5) {
    auto allocator = std::make_shared<DefaultAllocator>();
    SQ8Quantizer<metric> quantizer(dim, allocator.get());
    TestComputeCodes<SQ8Quantizer<metric>, metric>(quantizer, dim, count, error);
    TestComputer<SQ8Quantizer<metric>, metric>(quantizer, dim, count, error);
}

TEST_CASE("compute [ut][sq8_quantizer]") {
    auto dims = fixtures::get_common_used_dims();
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    float error = 0.05;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestComputeMetricSQ8<metrics[0]>(dim, count, error);
            TestComputeMetricSQ8<metrics[1]>(dim, count, error);
            TestComputeMetricSQ8<metrics[2]>(dim, count, error);
        }
    }
}

template <MetricType metric>
void
TestSerializeAndDeserializeMetricSQ8(uint64_t dim, int count, float error = 1e-5) {
    auto allocator = std::make_shared<DefaultAllocator>();
    SQ8Quantizer<metric> quantizer1(dim, allocator.get());
    SQ8Quantizer<metric> quantizer2(0, allocator.get());
    TestSerializeAndDeserialize<SQ8Quantizer<metric>, metric>(
        quantizer1, quantizer2, dim, count, error);
}

TEST_CASE("serialize&deserialize [ut][sq8_quantizer]") {
    auto dims = fixtures::get_common_used_dims();
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    float error = 0.05f;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestSerializeAndDeserializeMetricSQ8<metrics[0]>(dim, count, error);
            TestSerializeAndDeserializeMetricSQ8<metrics[1]>(dim, count, error);
            TestSerializeAndDeserializeMetricSQ8<metrics[2]>(dim, count, error);
        }
    }
}
