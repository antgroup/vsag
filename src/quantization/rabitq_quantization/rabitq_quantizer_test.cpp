
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
#include "fixtures.h"
#include "impl/allocator/default_allocator.h"
#include "impl/allocator/safe_allocator.h"
#include "quantization/quantizer_test.h"

using namespace vsag;

const auto dims = fixtures::get_common_used_dims();
const auto counts = {10, 100};

TEST_CASE("RaBitQ Basic Test ROM", "[ut][RaBitQuantizer]") {
    bool use_fht = GENERATE(true, false);
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
                dim, pca_dim, num_bits_per_dim, use_fht, allocator.get());

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
    bool use_fht = GENERATE(true, false);
    auto num_bits_per_dim = GENERATE(4, 32);
    for (auto dim : dims) {
        for (auto count : counts) {
            auto allocator = SafeAllocator::FactoryDefaultAllocator();
            RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
                dim, dim, num_bits_per_dim, use_fht, allocator.get());

            TestEncodeDecodeRaBitQ<RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR>>(
                quantizer, dim, count);
        }
    }
}

TEST_CASE("RaBitQ Compute", "[ut][RaBitQuantizer]") {
    bool use_fht = GENERATE(true, false);
    auto num_bits_per_dim = GENERATE(4, 32);
    for (auto dim : dims) {
        float numeric_error = 0.01 / std::sqrt(dim) * dim;
        float related_error = 0.05f;
        float unbounded_numeric_error_rate = 0.05f;
        float unbounded_related_error_rate = 0.1f;
        if (num_bits_per_dim == 4) {
            unbounded_related_error_rate = 0.12f;
        }
        if (use_fht) {
            unbounded_related_error_rate += 0.05f;
        }
        if (dim < 900) {
            continue;
        }
        for (auto count : counts) {
            auto allocator = SafeAllocator::FactoryDefaultAllocator();
            RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
                dim, dim, num_bits_per_dim, use_fht, allocator.get());

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
    bool use_fht = GENERATE(true, false);
    auto num_bits_per_dim = GENERATE(4, 32);
    for (auto dim : dims) {
        float numeric_error = 0.01 / std::sqrt(dim) * dim;
        float related_error = 0.05F;
        float unbounded_numeric_error_rate = 0.05F;
        float unbounded_related_error_rate = 0.1F;
        if (num_bits_per_dim == 4) {
            unbounded_related_error_rate = 0.15F;
        }
        if (dim < 900) {
            continue;
        }
        for (auto count : counts) {
            auto allocator = SafeAllocator::FactoryDefaultAllocator();
            RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer1(
                dim, dim, num_bits_per_dim, use_fht, allocator.get());
            RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer2(
                dim, dim, num_bits_per_dim, use_fht, allocator.get());

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
    bool use_fht = GENERATE(true, false);
    int dim = 6;
    uint64_t num_bits_per_dim_query = 4;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
        dim, dim, num_bits_per_dim_query, use_fht, allocator.get());

    std::vector<float> original_data = {1, 2, 4, 8, 15, 0};
    // input  [0010 0001, 1000 0100, 0000 1111]
    std::vector<uint8_t> input = {0x21, 0x84, 0x0f};
    std::vector<uint8_t> sq_data(4 + 4 + 4, 0);

    // test sq
    SQ4UniformQuantizer<MetricType::METRIC_TYPE_IP> sq4_quantizer(dim, allocator.get(), 0.0f);
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

TEST_CASE("RaBitQ Query SQ4 Transform dim=15", "[ut][RaBitQuantizer]") {
    bool use_fht = GENERATE(true, false);
    int dim = 15;
    int aligned_dim = ((dim + 511) / 512) * 512;
    uint64_t num_bits_per_dim_query = 4;
    int sq_code_size = aligned_dim / 8 * num_bits_per_dim_query;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
        dim, dim, num_bits_per_dim_query, use_fht, allocator.get());

    std::vector<float> original_data = {1, 2, 4, 8, 0, 3, 11, 15, 9, 13, 10, 6, 7, 12, 14};
    // input  [0010 0001, 1000 0100, 0011 0000, 1111 1011, 1101 1001, 0110 1010, 1100 0111, 0000 1110]
    std::vector<uint8_t> input = {0x21, 0x84, 0x30, 0xfb, 0xd9, 0x6a, 0xc7, 0x0e};
    std::vector<uint8_t> sq_data(dim + 4 + 4, 0);

    // test sq
    SQ4UniformQuantizer<MetricType::METRIC_TYPE_IP> sq4_quantizer(dim, allocator.get(), 0.0f);
    sq4_quantizer.Train(original_data.data(), 1);
    sq4_quantizer.EncodeOneImpl(original_data.data(), sq_data.data());

    auto is_consistent = std::memcmp(sq_data.data(), input.data(), input.size());
    REQUIRE(is_consistent == 0);

    // test reorder
    // output:
    //     1110 0001 0001 0011 000000000...
    //     1110 0010 0101 1100 000000000...
    //     1000 0100 0111 1010 000000000...
    //     1100 1000 0110 0111 000000000...
    std::vector<uint8_t> expected_output(sq_code_size, 0);
    expected_output[0] = 0xe1;
    expected_output[1] = 0x13;

    expected_output[aligned_dim / 8] = 0xe2;
    expected_output[aligned_dim / 8 * 1 + 1] = 0x5c;

    expected_output[aligned_dim / 8 * 2] = 0x84;
    expected_output[aligned_dim / 8 * 2 + 1] = 0x7a;

    expected_output[aligned_dim / 8 * 3] = 0xc8;
    expected_output[aligned_dim / 8 * 3 + 1] = 0x67;

    std::vector<uint8_t> output(sq_code_size, 0);
    std::vector<uint8_t> recovered_input(dim, 0);

    // reorder the input
    quantizer.ReOrderSQ4(input.data(), output.data());
    is_consistent = std::memcmp(expected_output.data(), output.data(), output.size());
    REQUIRE(is_consistent == 0);

    // recover the original order
    quantizer.RecoverOrderSQ4(output.data(), recovered_input.data());
    is_consistent = std::memcmp(recovered_input.data(), input.data(), input.size());
    REQUIRE(is_consistent == 0);
}

TEST_CASE("RaBitQ Query SQ Encode Decode", "[ut][RaBitQuantizer]") {
    bool use_fht = GENERATE(true, false);
    int dim = 6;
    uint64_t num_bits_per_dim_query = 4;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
        dim, dim, num_bits_per_dim_query, use_fht, allocator.get());

    std::vector<float> original_data = {1, 2, 4, 8, 15, 0};
    std::vector<uint8_t> sq_data(dim, 0);
    std::vector<uint8_t> expected_data = {1, 2, 4, 8, 15, 0};

    float upper_bound = std::numeric_limits<float>::max();
    float lower_bound = std::numeric_limits<float>::max();
    float delta = 0.0F;
    sum_type query_sum = 0;
    // test sq encode
    quantizer.EncodeSQ(
        original_data.data(), sq_data.data(), upper_bound, lower_bound, delta, query_sum);
    auto is_consistent = std::memcmp(sq_data.data(), expected_data.data(), expected_data.size());
    REQUIRE(is_consistent == 0);

    // test sq decode
    std::vector<float> decode_data(dim, 0);
    quantizer.DecodeSQ(sq_data.data(), decode_data.data(), upper_bound, lower_bound);
    for (int i = 0; i < dim; ++i) {
        REQUIRE(is_approx_zero(original_data[i] - decode_data[i]));
    }

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
    std::vector<uint8_t> recovered_input(dim, 0);
    // reorder the input
    quantizer.ReOrderSQ(sq_data.data(), output.data());
    is_consistent = std::memcmp(expected_output.data(), output.data(), output.size());
    REQUIRE(is_consistent == 0);

    // recover the original order
    quantizer.RecoverOrderSQ(output.data(), recovered_input.data());
    is_consistent = std::memcmp(recovered_input.data(), sq_data.data(), sq_data.size());
    REQUIRE(is_consistent == 0);
}

TEST_CASE("RaBitQ Query SQ Transform dim=15", "[ut][RaBitQuantizer]") {
    bool use_fht = GENERATE(true, false);
    int dim = 15;
    int aligned_dim = ((dim + 511) / 512) * 512;
    uint64_t num_bits_per_dim_query = 4;
    int sq_code_size = aligned_dim / 8 * num_bits_per_dim_query;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
        dim, dim, num_bits_per_dim_query, use_fht, allocator.get());

    std::vector<float> original_data = {1, 2, 4, 8, 0, 3, 11, 15, 9, 13, 10, 6, 7, 12, 14};
    std::vector<uint8_t> expected_data = {1, 2, 4, 8, 0, 3, 11, 15, 9, 13, 10, 6, 7, 12, 14};
    std::vector<uint8_t> sq_data(dim, 0);

    // test sq
    float upper_bound = std::numeric_limits<float>::max();
    float lower_bound = std::numeric_limits<float>::max();
    float delta = 0.0F;
    sum_type query_sum = 0;
    // test sq encode
    quantizer.EncodeSQ(
        original_data.data(), sq_data.data(), upper_bound, lower_bound, delta, query_sum);
    auto is_consistent = std::memcmp(sq_data.data(), expected_data.data(), expected_data.size());
    REQUIRE(is_consistent == 0);

    // test sq decode
    std::vector<float> decode_data(dim, 0);
    quantizer.DecodeSQ(sq_data.data(), decode_data.data(), upper_bound, lower_bound);
    for (int i = 0; i < dim; ++i) {
        REQUIRE(is_approx_zero(original_data[i] - decode_data[i]));
    }

    // test reorder
    // output:
    //     1110 0001 0001 0011 000000000...
    //     1110 0010 0101 1100 000000000...
    //     1000 0100 0111 1010 000000000...
    //     1100 1000 0110 0111 000000000...
    std::vector<uint8_t> expected_output(sq_code_size, 0);
    expected_output[0] = 0xe1;
    expected_output[1] = 0x13;

    expected_output[aligned_dim / 8] = 0xe2;
    expected_output[aligned_dim / 8 * 1 + 1] = 0x5c;

    expected_output[aligned_dim / 8 * 2] = 0x84;
    expected_output[aligned_dim / 8 * 2 + 1] = 0x7a;

    expected_output[aligned_dim / 8 * 3] = 0xc8;
    expected_output[aligned_dim / 8 * 3 + 1] = 0x67;

    std::vector<uint8_t> output(sq_code_size, 0);
    std::vector<uint8_t> recovered_input(dim, 0);

    // reorder the input
    quantizer.ReOrderSQ(sq_data.data(), output.data());
    is_consistent = std::memcmp(expected_output.data(), output.data(), output.size());
    REQUIRE(is_consistent == 0);

    // recover the original order
    quantizer.RecoverOrderSQ(output.data(), recovered_input.data());
    is_consistent = std::memcmp(recovered_input.data(), sq_data.data(), sq_data.size());
    REQUIRE(is_consistent == 0);
}

TEST_CASE("RaBitQ Query SQ Transform With All Same Element", "[ut][RaBitQuantizer]") {
    bool use_fht = GENERATE(true, false);
    int dim = 15;
    int aligned_dim = ((dim + 511) / 512) * 512;
    uint64_t num_bits_per_dim_query = 4;
    int sq_code_size = aligned_dim / 8 * num_bits_per_dim_query;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
        dim, dim, num_bits_per_dim_query, use_fht, allocator.get());

    std::vector<float> original_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<uint8_t> sq_data(dim, 0);

    // test sq
    float upper_bound = std::numeric_limits<float>::max();
    float lower_bound = std::numeric_limits<float>::max();
    float delta = 0.0F;
    sum_type query_sum = 0;
    quantizer.EncodeSQ(
        original_data.data(), sq_data.data(), upper_bound, lower_bound, delta, query_sum);
    std::vector<float> decode_data(dim, 0);
    quantizer.DecodeSQ(sq_data.data(), decode_data.data(), upper_bound, lower_bound);
    for (int i = 0; i < dim; ++i) {
        REQUIRE(is_approx_zero(original_data[i] - decode_data[i]));
    }
}