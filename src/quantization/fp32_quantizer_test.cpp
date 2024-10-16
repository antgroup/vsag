
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
#include "quantizer_test.h"

using namespace vsag;

const auto dims = {64, 128};
const auto counts = {10, 101};

template <MetricType Metric>
void
TestQuantizerEncodeDecodeMetricFP32(uint64_t dim, int count, float error = 1e-5) {
    FP32Quantizer<Metric> quantizer(dim);
    TestQuantizerEncodeDecode(quantizer, dim, count, error);
    TestQuantizerEncodeDecodeSame(quantizer, dim, count, 65536, error);
}

TEST_CASE("encode&decode [ut][fp32_quantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    float error = 1e-5f;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestQuantizerEncodeDecodeMetricFP32<metrics[0]>(dim, count, error);
            TestQuantizerEncodeDecodeMetricFP32<metrics[1]>(dim, count, error);
            TestQuantizerEncodeDecodeMetricFP32<metrics[2]>(dim, count, error);
        }
    }
}

template <MetricType Metric>
void
TestComputeMetricFP32(uint64_t dim, int count, float error = 1e-5) {
    FP32Quantizer<Metric> quantizer(dim);
    TestComputeCodes<FP32Quantizer<Metric>, Metric>(quantizer, dim, count, error);
    TestComputeCodesSame<FP32Quantizer<Metric>, Metric>(quantizer, dim, count, 65536);
    TestComputer<FP32Quantizer<Metric>, Metric>(quantizer, dim, count, error);
}

TEST_CASE("compute [ut][fp32_quantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    float error = 1e-5f;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestComputeMetricFP32<metrics[0]>(dim, count, error);
            TestComputeMetricFP32<metrics[1]>(dim, count, error);
            TestComputeMetricFP32<metrics[2]>(dim, count, error);
        }
    }
}

template <MetricType Metric>
void
TestSerializeAndDeserializeMetricFP32(uint64_t dim, int count, float error = 1e-5) {
    FP32Quantizer<Metric> quantizer1(dim);
    FP32Quantizer<Metric> quantizer2(0);
    TestSerializeAndDeserialize<FP32Quantizer<Metric>, Metric>(
        quantizer1, quantizer2, dim, count, error);
}

TEST_CASE("serialize&deserialize [ut][fp32_quantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    float error = 1e-5f;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestSerializeAndDeserializeMetricFP32<metrics[0]>(dim, count, error);
            TestSerializeAndDeserializeMetricFP32<metrics[1]>(dim, count, error);
            TestSerializeAndDeserializeMetricFP32<metrics[2]>(dim, count, error);
        }
    }
}
