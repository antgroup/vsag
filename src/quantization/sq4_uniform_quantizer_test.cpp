
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

TEST_CASE("SQ4 Uniform Encode and Decode", "[ut][SQ4UniformQuantizer]") {
    auto dims = fixtures::get_common_used_dims();
    uint32_t size = 1000;
    float error = 2 * 1.0f / 15.0f;
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};

    for (auto dim : dims) {
        {
            SQ4UniformQuantizer<metrics[0]> quantizer{dim};
            TestQuantizerEncodeDecode<>(quantizer, dim, size, error);
            TestQuantizerEncodeDecodeSame<>(quantizer, dim, size, 15);
        }
        {
            SQ4UniformQuantizer<metrics[1]> quantizer{dim};
            TestQuantizerEncodeDecode<>(quantizer, dim, size, error);
            TestQuantizerEncodeDecodeSame<>(quantizer, dim, size, 15);
        }
        {
            SQ4UniformQuantizer<metrics[2]> quantizer{dim};
            TestQuantizerEncodeDecode<>(quantizer, dim, size, error);
            TestQuantizerEncodeDecodeSame<>(quantizer, dim, size, 15);
        }
    }
}

TEST_CASE("SQ4 Uniform Compute Code with Code", "[ut][SQ4UniformQuantizer]") {
    auto dims = fixtures::get_common_used_dims();
    uint32_t size = 1000;
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};

    for (auto dim : dims) {
        {
            SQ4UniformQuantizer<metrics[0]> quantizer{dim};
            TestComputeCodesSame<>(quantizer, dim, size, metrics[0]);
        }

        {
            SQ4UniformQuantizer<metrics[1]> quantizer{dim};
            TestComputeCodesSame<>(quantizer, dim, size, metrics[1]);
        }

        {
            SQ4UniformQuantizer<metrics[2]> quantizer{dim};
            TestComputeCodesSame<>(quantizer, dim, size, metrics[2]);
        }
    }
}

TEST_CASE("SQ4 Uniform Compute Computer with Code", "[ut][SQ4UniformQuantizer]") {
    auto dims = fixtures::get_common_used_dims();
    uint32_t size = 1000;
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    for (auto dim : dims) {
        {
            SQ4UniformQuantizer<metrics[0]> quantizer{dim};
            TestComputerSame<>(quantizer, dim, size, metrics[0]);
        }

        {
            SQ4UniformQuantizer<metrics[1]> quantizer{dim};
            TestComputerSame<>(quantizer, dim, size, metrics[1]);
        }

        {
            SQ4UniformQuantizer<metrics[2]> quantizer{dim};
            TestComputerSame<>(quantizer, dim, size, metrics[2]);
        }
    }
}
