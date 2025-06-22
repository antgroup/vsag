
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

#include "sq8_row_quantizer.h"

#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "fixtures.h"
#include "quantization/quantizer_test.h"
#include "safe_allocator.h"

using namespace vsag;

const auto dims = fixtures::get_common_used_dims();
const auto counts = {10, 101};

template <MetricType metric>
void
TestQuantizerEncodeDecodeMetricSQ8Row(uint64_t dim,
                                      int count,
                                      float error = 1e-5,
                                      float error_same = 1e-2) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    SQ8RowQuantizer<metric> quantizer(dim, allocator.get());
    TestQuantizerEncodeDecode(quantizer, dim, count, error);
    TestQuantizerEncodeDecodeSame(quantizer, dim, count, 255, error_same);
}

TEST_CASE("SQ8 Row Encode and Decode", "[ut][SQ8RowQuantizer]") {
    constexpr MetricType metrics[2] = {MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP};
    float error = 2 * 1.0f / 255.0f;
    for (auto dim : dims) {
        for (auto count : counts) {
            auto error_same = (float)(dim * 255 * 0.01);
            TestQuantizerEncodeDecodeMetricSQ8Row<metrics[0]>(dim, count, error, error_same);
            TestQuantizerEncodeDecodeMetricSQ8Row<metrics[1]>(dim, count, error, error_same);
        }
    }
}

template <MetricType metric>
void
TestComputeMetricSQ8Row(uint64_t dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    SQ8RowQuantizer<metric> quantizer(dim, allocator.get());
    TestComputeCodesSame<SQ8RowQuantizer<metric>, metric>(quantizer, dim, count, error);
}

TEST_CASE("SQ8 Row Compute", "[ut][SQ8RowQuantizer]") {
    constexpr MetricType metrics[2] = {MetricType::METRIC_TYPE_L2SQR};
    float error = 4 * 1.0f / 255.0f;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestComputeMetricSQ8Row<metrics[0]>(dim, count, error);
            TestComputeMetricSQ8Row<metrics[1]>(dim, count, error);
        }
    }
}

template <MetricType metric>
void
TestSerializeAndDeserializeMetricSQ8Row(uint64_t dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    SQ8RowQuantizer<metric> quantizer1(dim, allocator.get());
    SQ8RowQuantizer<metric> quantizer2(0, allocator.get());
    TestSerializeAndDeserialize<SQ8RowQuantizer<metric>, metric, true>(
        quantizer1, quantizer2, dim, count, error);
}

TEST_CASE("SQ8 Row Serialize and Deserialize", "[ut][SQ8RowQuantizer]") {
    constexpr MetricType metrics[3] = {MetricType::METRIC_TYPE_L2SQR};
    for (auto dim : dims) {
        float error = 4 * 1.0f / 255.0f;
        for (auto count : counts) {
            TestSerializeAndDeserializeMetricSQ8Row<metrics[0]>(dim, count, error);
            //            TestSerializeAndDeserializeMetricSQ8Row<metrics[1]>(dim, count, error);
            TestSerializeAndDeserializeMetricSQ8Row<metrics[2]>(dim, count, error);
        }
    }
}
