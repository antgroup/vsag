
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

#include "rabitq_quantizer.h"

#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "default_allocator.h"
#include "fixtures.h"
#include "quantization/quantizer_test.h"
#include "safe_allocator.h"

using namespace vsag;

const auto dims = fixtures::get_common_used_dims();
const auto counts = {10, 100};

TEST_CASE("RaBitQ Encode and Decode", "[ut][RaBitQuantizer]") {
    for (auto dim : dims) {
        for (auto count : counts) {
            auto allocator = SafeAllocator::FactoryDefaultAllocator();
            RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(dim, allocator.get());

            TestEncodeDecodeRaBitQ<RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR>>(
                quantizer, dim, count);
        }
    }
}

TEST_CASE("RaBitQ Compute", "[ut][RaBitQuantizer]") {
    for (auto dim : dims) {
        for (auto count : counts) {
            float numeric_error = 4.0 / std::sqrt(dim) * dim;
            auto allocator = SafeAllocator::FactoryDefaultAllocator();
            RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(dim, allocator.get());

            TestComputer<RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                         MetricType::METRIC_TYPE_L2SQR>(quantizer, dim, count, numeric_error);
        }
    }
}

TEST_CASE("RaBitQ Serialize and Deserialize", "[ut][RaBitQuantizer]") {
    for (auto dim : dims) {
        float numeric_error = 4.0 / std::sqrt(dim) * dim;
        for (auto count : counts) {
            auto allocator = SafeAllocator::FactoryDefaultAllocator();
            RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer1(dim, allocator.get());
            RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer2(dim, allocator.get());

            TestSerializeAndDeserialize<RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                                        MetricType::METRIC_TYPE_L2SQR>(
                quantizer1, quantizer2, dim, count, numeric_error, true);
        }
    }
}
