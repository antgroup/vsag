
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
#include <catch2/generators/catch_generators.hpp>
#include <vector>

#include "../scalar_quantization/sq4_uniform_quantizer.h"
#include "default_allocator.h"
#include "fixtures.h"
#include "quantization/quantizer_test.h"
#include "safe_allocator.h"

using namespace vsag;

const auto dims = fixtures::get_common_used_dims();
const auto counts = {10, 100};

TEST_CASE("RaBitQ Basic Test", "[ut][RaBitQuantizer]") {
    auto num_bits_per_dim = GENERATE(4, 32);
    for (auto dim : dims) {
        uint64_t pca_dim = dim;
        if (dim >= 1500) {
            pca_dim = dim / 2;
        }
        for (auto count : counts) {
            auto allocator = SafeAllocator::FactoryDefaultAllocator();
            auto vecs = fixtures::generate_vectors(count, dim);
            RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
                dim, pca_dim, num_bits_per_dim, allocator.get());

            // name
            REQUIRE(quantizer.NameImpl() == QUANTIZATION_TYPE_VALUE_RABITQ);

            // train
            REQUIRE(quantizer.TrainImpl(vecs.data(), 0) == false);
            REQUIRE(quantizer.TrainImpl(vecs.data(), count) == true);
            REQUIRE(quantizer.TrainImpl(vecs.data(), count) == true);
        }
    }
}

TEST_CASE("RaBitQ Encode and Decode", "[ut][RaBitQuantizer]") {
    auto num_bits_per_dim = GENERATE(4, 32);
    for (auto dim : dims) {
        for (auto count : counts) {
            auto allocator = SafeAllocator::FactoryDefaultAllocator();
            RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
                dim, dim, num_bits_per_dim, allocator.get());

            TestEncodeDecodeRaBitQ<RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR>>(
                quantizer, dim, count);
        }
    }
}

TEST_CASE("RaBitQ Compute", "[ut][RaBitQuantizer]") {
    auto num_bits_per_dim = GENERATE(4, 32);
    for (auto dim : dims) {
        float numeric_error = 0.01 / std::sqrt(dim) * dim;
        float related_error = 0.05f;
        float unbounded_numeric_error_rate = 0.05f;
        float unbounded_related_error_rate = 0.1f;
        if (num_bits_per_dim == 4) {
            unbounded_related_error_rate = 0.12f;
        }
        if (dim < 900) {
            continue;
        }
        for (auto count : counts) {
            auto allocator = SafeAllocator::FactoryDefaultAllocator();
            RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
                dim, dim, num_bits_per_dim, allocator.get());

            TestComputer<RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                         MetricType::METRIC_TYPE_L2SQR>(quantizer,
                                                        dim,
                                                        count,
                                                        numeric_error,
                                                        related_error,
                                                        true,
                                                        unbounded_numeric_error_rate,
                                                        unbounded_related_error_rate);
            REQUIRE_THROWS(TestComputeCodes<RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                                            MetricType::METRIC_TYPE_L2SQR>(
                quantizer, dim, count, numeric_error, false));
        }
    }
}

TEST_CASE("RaBitQ Serialize and Deserialize", "[ut][RaBitQuantizer]") {
    auto num_bits_per_dim = GENERATE(4, 32);
    for (auto dim : dims) {
        float numeric_error = 0.01 / std::sqrt(dim) * dim;
        float related_error = 0.05f;
        float unbounded_numeric_error_rate = 0.05f;
        float unbounded_related_error_rate = 0.1f;
        if (num_bits_per_dim == 4) {
            unbounded_related_error_rate = 0.12f;
        }
        if (dim < 900) {
            continue;
        }
        for (auto count : counts) {
            auto allocator = SafeAllocator::FactoryDefaultAllocator();
            RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer1(
                dim, dim, num_bits_per_dim, allocator.get());
            RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer2(
                dim, dim, num_bits_per_dim, allocator.get());

            TestSerializeAndDeserialize<RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                                        MetricType::METRIC_TYPE_L2SQR>(quantizer1,
                                                                       quantizer2,
                                                                       dim,
                                                                       count,
                                                                       numeric_error,
                                                                       related_error,
                                                                       unbounded_numeric_error_rate,
                                                                       unbounded_related_error_rate,
                                                                       true);
        }
    }
}

TEST_CASE("RaBitQ Query SQ4 Transform", "[ut][RaBitQuantizer]") {
    int dim = 5;
    uint64_t num_bits_per_dim_query = 4;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
        dim, dim, num_bits_per_dim_query, allocator.get());

    std::vector<float> original_data = {1, 2, 4, 8, 15, 0};
    // input  [0010 0001, 1000 0100, 0000 1111]
    std::vector<uint8_t> input = {0x21, 0x84, 0x0f};
    std::vector<uint8_t> sq_data(4 + 4 + 4, 0);

    // test sq
    SQ4UniformQuantizer<MetricType::METRIC_TYPE_IP> sq4_quantizer(6, allocator.get(), 0.0f);
    sq4_quantizer.Train(original_data.data(), 1);
    sq4_quantizer.EncodeOneImpl(original_data.data(), sq_data.data());
    auto is_consistent = std::memcmp(sq_data.data(), input.data(), input.size());
    REQUIRE(is_consistent == 0);
    REQUIRE(std::abs(*(float*)(&sq_data[4]) - 30) < 1e-5);
    REQUIRE(std::abs(*(float*)(&sq_data[8]) - 30) < 1e-5);

    // test reorder
    // output  [0001 0001, 0001 0010, 0001 0100, 0001 1000]
    std::vector<uint8_t> expected_output;
    expected_output.reserve(64 * 4);
    for (auto i = 0; i < 64 * 4; i++) {
        if (i == 0) {
            expected_output.push_back(0x11);
        } else if (i == 64) {
            expected_output.push_back(0x12);
        } else if (i == 128) {
            expected_output.push_back(0x14);
        } else if (i == 192) {
            expected_output.push_back(0x18);
        } else {
            expected_output.push_back(0);
        }
    }
    std::vector<uint8_t> output(64 * 4, 0);
    std::vector<uint8_t> recovered_input(3, 0);

    // reorder the input
    quantizer.ReOrderSQ4(input.data(), output.data());
    is_consistent = std::memcmp(expected_output.data(), output.data(), output.size());
    REQUIRE(is_consistent == 0);

    // recover the original order
    quantizer.RecoverOrderSQ4(output.data(), recovered_input.data());
    is_consistent = std::memcmp(recovered_input.data(), input.data(), input.size());
    REQUIRE(is_consistent == 0);
}
