
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

#include "sq4_quantizer.h"

#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "fixtures.h"
#include "quantizer_test.h"
using namespace vsag;

const auto dims = {64, 128};
const auto counts = {10, 101};

template <MetricType Metric>
void
TestQuantizerEncodeDecodeMetricSQ4(uint64_t dim,
                                   int count,
                                   float error = 1e-5,
                                   float error_same = 1e-2) {
    SQ4Quantizer<Metric> quantizer(dim);
    TestQuantizerEncodeDecode(quantizer, dim, count, error);
    TestQuantizerEncodeDecodeSame(quantizer, dim, count, 15, error_same);
}

TEST_CASE("Encode and Decode", "[ut][SQ4Quantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    float error = 2 * 1.0f / 15.0f;
    for (auto dim : dims) {
        for (auto count : counts) {
            auto error_same = (float)(dim * 255 * 0.01);
            TestQuantizerEncodeDecodeMetricSQ4<metrics[0]>(dim, count, error, error_same);
            TestQuantizerEncodeDecodeMetricSQ4<metrics[1]>(dim, count, error, error_same);
            TestQuantizerEncodeDecodeMetricSQ4<metrics[2]>(dim, count, error, error_same);
        }
    }
}

template <MetricType Metric>
void
TestComputeMetricSQ4(uint64_t dim, int count, float error = 1e-5) {
    SQ4Quantizer<Metric> quantizer(dim);
    TestComputeCodes<SQ4Quantizer<Metric>, Metric>(quantizer, dim, count, error);
    TestComputeCodesSame<SQ4Quantizer<Metric>, Metric>(quantizer, dim, count, error);
    TestComputer<SQ4Quantizer<Metric>, Metric>(quantizer, dim, count, error);
}

TEST_CASE("compute [ut][sq4_quantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    float error = 0.2;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestComputeMetricSQ4<metrics[0]>(dim, count, error);
            TestComputeMetricSQ4<metrics[1]>(dim, count, error);
            TestComputeMetricSQ4<metrics[2]>(dim, count, error);
        }
    }
}
