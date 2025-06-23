
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

#include "vector_transformer.h"

namespace vsag {

// aka PCA
class PCATransformer : public VectorTransformer {
public:
    // interface
    explicit PCATransformer(Allocator* allocator, int64_t input_dim, int64_t output_dim);

    void
    Train(const float* data, uint64_t count) override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(StreamReader& reader) override;

    void
    Transform(const float* input_vec, float* output_vec) const override;

    void
    InverseTransform(const float* input_vec, float* output_vec) const override;

public:
    // make public for test
    void
    CopyPCAMatrixForTest(float* out_pca_matrix) const;

    void
    CopyMeanForTest(float* out_mean) const;

    void
    SetMeanForTest(const float* input_mean);

    void
    SetPCAMatrixForText(const float* input_pca_matrix);

    void
    ComputeColumnMean(const float* data, uint64_t count);

    void
    ComputeCovarianceMatrix(const float* centralized_data,
                            uint64_t count,
                            float* covariance_matrix) const;

    bool
    PerformEigenDecomposition(const float* covariance_matrix);

    void
    CentralizeData(const float* original_data, float* centralized_data) const;

private:
    Vector<float> pca_matrix_;  // [input_dim_ * output_dim_]
    Vector<float> mean_;        // [input_dim_ * 1]
};

}  // namespace vsag
