
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

struct PCAMeta : public TransformerMeta {
    float residual_norm;

    void
    EncodeMeta(uint8_t* code) override {
        *((float*)code) = residual_norm;
    }

    void
    DecodeMeta(uint8_t* code, uint32_t align_size) override {
        residual_norm = *((float*)code);
    }

    static uint32_t
    GetMetaSize() {
        return sizeof(float);
    }

    static uint32_t
    GetMetaSize(uint32_t align_size) {
        return std::max(static_cast<uint32_t>(sizeof(float)), align_size);
    }

    static uint32_t
    GetAlignSize() {
        return sizeof(float);
    }
};

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

    TransformerMetaPtr
    Transform(const float* input_vec, float* output_vec) const override;

    void
    InverseTransform(const float* input_vec, float* output_vec) const override;

    float
    RecoveryDistance(float dist, const uint8_t* meta1, const uint8_t* meta2) const override {
        return dist;
    };

    uint32_t
    GetMetaSize() const override {
        return PCAMeta::GetMetaSize();
    }

    uint32_t
    GetMetaSize(uint32_t align_size) const override {
        return PCAMeta::GetMetaSize(align_size);
    }

    uint32_t
    GetAlignSize() const override {
        return PCAMeta::GetAlignSize();
    }

public:
    // make public for test
    void
    CopyPCAMatrixForTest(float* out_pca_matrix) const;

    void
    CopyMeanForTest(float* out_mean) const;

    void
    SetMeanForTest(const float* input_mean);

    void
    SetPCAMatrixForTest(const float* input_pca_matrix);

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
