
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

#include "fht_kac_rotate_transformer.h"

#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"
#include "storage/serialization_template_test.h"

using namespace vsag;

void
TestSame(FhtKacRotator& rom1, FhtKacRotator& rom2, uint64_t dim) {
    size_t flip_len = (dim + 7) / FhtKacRotator::BYTE_LEN * FhtKacRotator::ROUND;
    std::vector<uint8_t> mat1(flip_len);
    rom1.CopyFlip(mat1.data());
    std::vector<uint8_t> mat2(flip_len);
    rom2.CopyFlip(mat2.data());
    uint64_t count_same = 0;
    for (uint64_t i = 0; i < flip_len; i++) {
        if (mat1[i] == mat2[i]) {
            count_same++;
        }
    }

    REQUIRE(count_same == flip_len);
}

void
TestTransform(FhtKacRotator& rom, uint32_t dim) {
    std::vector<float> vec = fixtures::generate_vectors(1, dim);
    std::vector<float> original_vec = vec;
    std::vector<float> inverse_vec = vec;

    rom.Transform(original_vec.data(), vec.data());
    //test
    rom.InverseTransform(vec.data(), inverse_vec.data());
    // verify that the length remains constant (orthogonal matrix preserving length)
    double original_length = 0.0, transformed_length = 0.0, inverse_length = 0.0;
    for (uint32_t i = 0; i < dim; ++i) {
        original_length += original_vec[i] * original_vec[i];
        transformed_length += vec[i] * vec[i];
        inverse_length += inverse_vec[i] * inverse_vec[i];

        REQUIRE(std::fabs(original_vec[i] - inverse_vec[i]) < 1e-4);
    }
    REQUIRE(std::fabs(original_length - transformed_length) < 1e-4);
    REQUIRE(std::fabs(original_length - inverse_length) < 1e-4);
}

TEST_CASE("Basic Hadamard Test", "[ut][FhtKacRotator]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const auto dims = fixtures::get_common_used_dims();
    for (auto dim : dims) {
        INFO(fmt::format("dim = {}", dim));
        FhtKacRotator rom(allocator.get(), dim);
        rom.Train();
        TestTransform(rom, dim);
    }
}

TEST_CASE("Hadamard Matrix Serialize / Deserialize Test", "[ut][FhtKacRotator]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const auto dims = fixtures::get_common_used_dims();

    for (auto dim : dims) {
        FhtKacRotator rom1(allocator.get(), dim);
        FhtKacRotator rom2(allocator.get(), dim);
        rom1.Train();
        rom2.Train();

        test_serializion(rom1, rom2);

        TestSame(rom1, rom2, dim);
    }
}

TEST_CASE("Hadamard transform matches dense reference", "[ut][FhtKacRotator]") {
    constexpr uint64_t dim = 32;
    constexpr uint64_t flip_offset = dim / FhtKacRotator::BYTE_LEN;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    FhtKacRotator rom(allocator.get(), dim);
    std::vector<uint8_t> flips(flip_offset * FhtKacRotator::ROUND);

    SECTION("trained masks") {
        rom.Train();
        rom.CopyFlip(flips.data());
    }
    SECTION("two matching bytes are valid") {
        // Independent uniform masks can share bytes: for 16 bytes, P(2+ matches)
        // is about 0.18%. Such collisions do not violate the transform contract.
        std::vector<uint8_t> other_flips(flips.size());
        uint64_t matching_bytes = 0;
        for (uint64_t i = 0; i < flips.size(); ++i) {
            other_flips[i] = static_cast<uint8_t>(17 * i);
            flips[i] = i < 2 ? other_flips[i] : static_cast<uint8_t>(other_flips[i] ^ 0xa5);
            matching_bytes += flips[i] == other_flips[i];
        }
        REQUIRE(matching_bytes == 2);
        std::stringstream stream;
        IOStreamWriter writer(stream);
        StreamWriter::WriteVector(writer, flips);
        IOStreamReader reader(stream);
        rom.Deserialize(reader);
    }

    // Check every basis vector against H[r,c] = (-1)^popcount(r & c) / sqrt(dim).
    // This oracle uses dense multiplication, independently of the SIMD butterflies.
    // It detects missing sign flips, incorrect round order, scaling, and a no-op
    // transform, which norm preservation and a round trip alone cannot detect.
    for (uint64_t basis = 0; basis < dim; ++basis) {
        CAPTURE(basis);
        std::vector<float> input(dim, 0.0F), actual(dim), inverse(dim);
        input[basis] = 1.0F;
        std::vector<double> expected(input.begin(), input.end());
        for (int round = 0; round < FhtKacRotator::ROUND; ++round) {
            std::vector<double> next(dim, 0.0);
            for (uint64_t row = 0; row < dim; ++row) {
                for (uint64_t col = 0; col < dim; ++col) {
                    int sign = 1;
                    for (uint64_t bits = row & col; bits != 0; bits &= bits - 1) {
                        sign = -sign;
                    }
                    if ((flips[round * flip_offset + col / 8] & (1U << (col % 8))) != 0) {
                        sign = -sign;
                    }
                    next[row] += sign * expected[col] / std::sqrt(static_cast<double>(dim));
                }
            }
            expected = std::move(next);
        }
        rom.Transform(input.data(), actual.data());
        rom.InverseTransform(actual.data(), inverse.data());
        for (uint64_t row = 0; row < dim; ++row) {
            CAPTURE(row);
            REQUIRE(std::fabs(actual[row] - expected[row]) < 1e-5);
            REQUIRE(std::fabs(inverse[row] - input[row]) < 1e-5);
        }
    }
}
