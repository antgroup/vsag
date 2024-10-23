
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

#include "sq4_uniform_quantizer.h"

#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "../../tests/fixtures/fixtures.h"
#include "quantizer_test.h"

using namespace vsag;

const auto dims = fixtures::get_common_used_dims();
const auto counts = {10, 101};

template <MetricType Metric>
void
TestQuantizerEncodeDecodeMetricSQ4Uniform(uint64_t dim,
                                          int count,
                                          float error = 1e-5,
                                          float error_same = 1e-2) {
    SQ4UniformQuantizer<Metric> quantizer(dim);
    TestQuantizerEncodeDecode(quantizer, dim, count, error);
    TestQuantizerEncodeDecodeSame(quantizer, dim, count, 15, error_same);
}

TEST_CASE("SQ4 Uniform Encode and Decode", "[ut][SQ4UniformQuantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    float error = 2 * 1.0f / 15.0f;
    for (auto dim : dims) {
        for (auto count : counts) {
            auto error_same = (float)(dim * 255 * 0.01);
            TestQuantizerEncodeDecodeMetricSQ4Uniform<metrics[0]>(dim, count, error, error_same);
            TestQuantizerEncodeDecodeMetricSQ4Uniform<metrics[1]>(dim, count, error, error_same);
            TestQuantizerEncodeDecodeMetricSQ4Uniform<metrics[2]>(dim, count, error, error_same);
        }
    }
}

template <MetricType Metric>
void
TestComputeMetricSQ4Uniform(uint64_t dim, int count, float error = 1e-5) {
    SQ4UniformQuantizer<Metric> quantizer(dim);
    TestComputeCodesSame<SQ4UniformQuantizer<Metric>, Metric>(quantizer, dim, count, error);
}

TEST_CASE("compute [ut][SQ4UniformQuantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    float error = 4 * 1.0f / 15.0f;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestComputeMetricSQ4Uniform<metrics[0]>(dim, count, error);
            TestComputeMetricSQ4Uniform<metrics[1]>(dim, count, error);
            TestComputeMetricSQ4Uniform<metrics[2]>(dim, count, error);
        }
    }
}
