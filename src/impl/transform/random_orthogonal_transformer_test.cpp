
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

#include "random_orthogonal_transformer.h"

#include <cblas.h>

#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"

using namespace vsag;

void
TestSame(RandomOrthogonalMatrix& rom1, RandomOrthogonalMatrix& rom2, uint64_t dim) {
    std::vector<float> mat1(dim * dim);
    rom1.CopyOrthogonalMatrix(mat1.data());

    std::vector<float> mat2(dim * dim);
    rom2.CopyOrthogonalMatrix(mat2.data());

    uint64_t count_same = 0;
    for (uint64_t i = 0; i < dim * dim; i++) {
        if (std::abs(mat1[i] - mat2[i]) < 1e-3) {
            count_same++;
        }
    }

    REQUIRE(count_same == dim * dim);
}

void
TestRandomness(RandomOrthogonalMatrix& rom1, RandomOrthogonalMatrix& rom2, uint64_t dim) {
    std::vector<float> mat1(dim * dim);
    rom1.CopyOrthogonalMatrix(mat1.data());

    std::vector<float> mat2(dim * dim);
    rom2.CopyOrthogonalMatrix(mat2.data());

    uint64_t count_same = 0, count_non_zero = 0;
    for (uint64_t i = 0; i < dim * dim; i++) {
        if (not(std::abs(mat1[i]) < 1e-3 and std::abs(mat2[i]) < 1e-3)) {
            if (std::abs(mat1[i] - mat2[i]) < 1e-3) {
                count_same++;
            }
            count_non_zero++;
        }
    }

    REQUIRE(count_same <= (uint64_t)(0.1 * count_non_zero));
}

void
TestOrthogonality(RandomOrthogonalMatrix& rom, uint64_t dim) {
    std::vector<float> Q(dim * dim);
    rom.CopyOrthogonalMatrix(Q.data());

    // compute Q ^ T * Q
    std::vector<float> result(dim * dim, 0.0);
    cblas_sgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                dim,
                dim,
                dim,
                1.0f,
                Q.data(),
                dim,
                Q.data(),
                dim,
                0.0f,
                result.data(),
                dim);

    // constructing unit matrices
    std::vector<float> identity(dim * dim, 0.0);
    for (int i = 0; i < dim; ++i) {
        identity[i * dim + i] = 1.0;
    }

    // verify that Q ^ T * Q is close to the unit matrix
    REQUIRE(result.size() == identity.size());
    for (size_t i = 0; i < result.size(); ++i) {
        REQUIRE(std::fabs(result[i] - identity[i]) < 1e-4);
    }
}

void
TestTransform(RandomOrthogonalMatrix& rom, uint32_t dim) {
    std::vector<float> vec = fixtures::generate_vectors(1, dim);
    std::vector<float> original_vec = vec;
    std::vector<float> inverse_vec = vec;

    rom.Transform(original_vec.data(), vec.data());
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

void
TestDeterminant(RandomOrthogonalMatrix& rom) {
    double det = rom.ComputeDeterminant();
    REQUIRE(std::fabs(det - 1) < 1e-4);
}

TEST_CASE("Random Orthogonal Matrix Basic Test", "[ut][RandomOrthogonalMatrix]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const auto dims = fixtures::get_common_used_dims();

    for (auto dim : dims) {
        RandomOrthogonalMatrix rom(allocator.get(), dim);
        RandomOrthogonalMatrix rom_alter(allocator.get(), dim);

        REQUIRE(rom.GenerateRandomOrthogonalMatrix() == true);
        REQUIRE(rom_alter.GenerateRandomOrthogonalMatrix() == true);

        TestOrthogonality(rom, dim);
        TestTransform(rom, dim);
        TestDeterminant(rom);
        TestRandomness(rom, rom_alter, dim);
    }
}

TEST_CASE("Random Orthogonal Matrix Serialize / Deserialize Test", "[ut][RandomOrthogonalMatrix]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const auto dims = fixtures::get_common_used_dims();

    for (auto dim : dims) {
        RandomOrthogonalMatrix rom1(allocator.get(), dim);
        RandomOrthogonalMatrix rom2(allocator.get(), dim);

        REQUIRE(rom1.GenerateRandomOrthogonalMatrix() == true);
        REQUIRE(rom2.GenerateRandomOrthogonalMatrix() == true);

        fixtures::TempDir dir("rom");
        auto filename = dir.GenerateRandomFile();
        std::ofstream outfile(filename.c_str(), std::ios::binary);
        IOStreamWriter writer(outfile);
        rom1.Serialize(writer);
        outfile.close();

        std::ifstream infile(filename.c_str(), std::ios::binary);
        IOStreamReader reader(infile);
        rom2.Deserialize(reader);
        infile.close();

        TestOrthogonality(rom1, dim);
        TestTransform(rom1, dim);
        TestDeterminant(rom1);

        TestSame(rom1, rom2, dim);
    }
}
