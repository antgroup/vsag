
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

#pragma once

#include <cblas.h>
#include <lapacke.h>

#include <random>

#include "../logger.h"
#include "typing.h"
#include "vsag/allocator.h"

namespace vsag {

class RandomOrthogonalMatrix {
public:
    RandomOrthogonalMatrix(uint64_t dim, Allocator* allocator, uint64_t retries = 3)
        : dim_(dim), allocator_(allocator) {
        orthogonal_matrix_ = (float*)allocator_->Allocate(sizeof(float) * dim_ * dim_);
        for (uint64_t i = 0; i < retries; i++) {
            bool result_gen = GenerateRandomOrthogonalMatrix();
            if (result_gen) {
                break;
            } else {
                logger::error(
                    fmt::format("Retrying generating random orthogonal matrix: {} times", i + 1));
            }
        }
    }

    ~RandomOrthogonalMatrix() {
        allocator_->Deallocate(orthogonal_matrix_);
    }

    void
    CopyOrthogonalMatrix(float* out_matrix) const;

    void
    Transform(const float* vec, float* out_vec) const;

    bool
    GenerateRandomOrthogonalMatrix();

    double
    ComputeDeterminant() const;

private:
    Allocator* const allocator_{nullptr};

    const uint64_t dim_{0};

    float* orthogonal_matrix_{nullptr};  // OpenBLAS and LAPACK use double vector
};

}  // namespace vsag
