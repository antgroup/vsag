
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

// aka PCA
class PrincipalComponentAnalysis {
public:
    // interface
    PrincipalComponentAnalysis(uint64_t original_dim, uint64_t target_dim, Allocator* allocator)
        : original_dim_(original_dim),
          target_dim_(target_dim),
          allocator_(allocator),
          pca_matrix_(allocator),
          mean_(allocator) {
        pca_matrix_.resize(target_dim * original_dim);
        mean_.resize(original_dim);
    }

    bool
    Train(const float* data, uint64_t count);

    void
    Transform(const float* original_vec, float* transformed_vec) const;

    void
    Serialize(StreamWriter& writer);

    void
    Deserialize(StreamReader& reader);

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
    Allocator* const allocator_{nullptr};

    const uint64_t original_dim_;
    const uint64_t target_dim_;

    vsag::Vector<float> pca_matrix_;  // [target_dim_ * original_dim_]
    vsag::Vector<float> mean_;        // [original_dim_ * 1]
};

}  // namespace vsag
