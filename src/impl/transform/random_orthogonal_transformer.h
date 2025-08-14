
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

#include "vector_transformer.h"

namespace vsag {

struct ROMMeta : public TransformerMeta {};

class RandomOrthogonalMatrix : public VectorTransformer {
public:
    explicit RandomOrthogonalMatrix(Allocator* allocator,
                                    int64_t dim,
                                    uint64_t retries = MAX_RETRIES);

    virtual ~RandomOrthogonalMatrix() override = default;

    TransformerMetaPtr
    Transform(const float* original_vec, float* transformed_vec) const override;

    void
    InverseTransform(const float* transformed_vec, float* original_vec) const override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(StreamReader& reader) override;

    void
    Train(const float* data, uint64_t count) override;

public:
    void
    CopyOrthogonalMatrix(float* out_matrix) const;

    bool
    GenerateRandomOrthogonalMatrix();

    void
    GenerateRandomOrthogonalMatrixWithRetry();

    double
    ComputeDeterminant() const;

public:
    static constexpr uint64_t MAX_RETRIES = 3;

private:
    Vector<float> orthogonal_matrix_;
    const uint64_t generate_retries_{0};
};

}  // namespace vsag
