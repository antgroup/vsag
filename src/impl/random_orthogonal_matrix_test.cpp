
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

#include "random_orthogonal_matrix.h"

#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "safe_allocator.h"

using namespace vsag;

void
TestOrthogonality(RandomOrthogonalMatrix& rom, uint32_t dim) {
    std::vector<float> Q(dim * dim);
    rom.GetOrthogonalMatrix(Q.data());

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

    // verify that Q^T * Q is close to the unit matrix
    REQUIRE(result.size() == identity.size());
    for (size_t i = 0; i < result.size(); ++i) {
        REQUIRE(std::fabs(result[i] - identity[i]) < 1e-4);
    }
}

void
TestTransform(RandomOrthogonalMatrix& rom, uint32_t dim) {
    std::vector<float> vec = fixtures::generate_vectors(1, dim);
    std::vector<float> original_vec = vec;

    rom.Transform(vec.data());

    // verify that the length remains constant (orthogonal matrix preserving length)
    double original_length = 0.0, transformed_length = 0.0;
    for (uint32_t i = 0; i < dim; ++i) {
        original_length += original_vec[i] * original_vec[i];
        transformed_length += vec[i] * vec[i];
    }
    REQUIRE(std::fabs(original_length - transformed_length) < 1e-4);
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
        RandomOrthogonalMatrix rom(dim, allocator.get());
        REQUIRE(rom.GenerateRandomOrthogonalMatrix() == true);
        TestOrthogonality(rom, dim);
        TestTransform(rom, dim);
        TestDeterminant(rom);
    }
}
