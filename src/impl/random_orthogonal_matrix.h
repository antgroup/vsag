
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
#include "stream_reader.h"
#include "stream_writer.h"
#include "typing.h"
#include "vsag/allocator.h"

namespace vsag {

static const uint64_t MAX_RETRIES = 3;

class RandomOrthogonalMatrix {
public:
    RandomOrthogonalMatrix(uint64_t dim, Allocator* allocator, uint64_t retries = MAX_RETRIES)
        : dim_(dim),
          allocator_(allocator),
          orthogonal_matrix_(allocator),
          generate_retries_(retries) {
        orthogonal_matrix_.resize(dim * dim);
    }

    void
    CopyOrthogonalMatrix(float* out_matrix) const;

    void
    Transform(const float* original_vec, float* transformed_vec) const;

    void
    InverseTransform(const float* transformed_vec, float* original_vec) const;

    bool
    GenerateRandomOrthogonalMatrix();

    void
    GenerateRandomOrthogonalMatrixWithRetry();

    double
    ComputeDeterminant() const;

    void
    Serialize(StreamWriter& writer);

    void
    Deserialize(StreamReader& reader);

private:
    Allocator* const allocator_{nullptr};

    const uint64_t dim_{0};

    vsag::Vector<float> orthogonal_matrix_;
    const uint64_t generate_retries_{0};
};

}  // namespace vsag
