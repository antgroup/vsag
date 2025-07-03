
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

using namespace vsag;

void
TestRandomness(FhtKacRotator& rom1, FhtKacRotator& rom2, int dim) {
    size_t flip_len = (dim + 7) / FhtKacRotator::BYTE_LEN * FhtKacRotator::ROUND;
    std::vector<uint8_t> mat1(flip_len);
    rom1.CopyFlip(mat1.data());

    std::vector<uint8_t> mat2(flip_len);
    rom2.CopyFlip(mat2.data());

    uint64_t count_same = 0, count_non_zero = 0;
    for (uint64_t i = 0; i < flip_len; i++) {
        if (mat1[i] == mat2[i]) {
            count_same++;
        }
        count_non_zero++;
    }
    uint64_t threshold = std::max(1UL, static_cast<uint64_t>(0.1 * count_non_zero));
    REQUIRE(count_same <= threshold);
}

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
        FhtKacRotator rom_alter(allocator.get(), dim);
        rom.Train();
        rom_alter.Train();
        TestTransform(rom, dim);
        TestRandomness(rom, rom_alter, dim);
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

        fixtures::TempDir dir("hadamard");
        auto filename = dir.GenerateRandomFile();
        std::ofstream outfile(filename.c_str(), std::ios::binary);
        IOStreamWriter writer(outfile);
        rom1.Serialize(writer);
        outfile.close();
        std::ifstream infile(filename.c_str(), std::ios::binary);
        IOStreamReader reader(infile);
        rom2.Deserialize(reader);
        infile.close();

        TestSame(rom1, rom2, dim);
    }
}
