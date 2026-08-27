
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

#include "pca_transformer.h"

#include <fmt/format.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <random>

#include "impl/blas/blas_function.h"
#include "vsag_exception.h"

namespace vsag {
PCATransformer::PCATransformer(Allocator* allocator, int64_t input_dim, int64_t output_dim)
    : VectorTransformer(allocator, input_dim, output_dim),
      pca_matrix_(allocator),
      mean_(allocator) {
    pca_matrix_.resize(output_dim * input_dim);
    mean_.resize(input_dim);
    this->type_ = VectorTransformerType::PCA;
}

void
PCATransformer::Train(const float* data, uint64_t count) {
    if (data == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "PCA training data pointer is null");
    }
    if (count < 2) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("PCA training requires at least 2 samples, got {}", count));
    }
    vsag::Vector<float> centralized_data(allocator_);
    centralized_data.resize(count * input_dim_, 0.0F);

    vsag::Vector<float> covariance_matrix(allocator_);
    covariance_matrix.resize(input_dim_ * input_dim_, 0.0F);

    // 1. compute mean (stored in mean_)
    ComputeColumnMean(data, count);

    // 2. centralize data
    for (uint64_t i = 0; i < count; ++i) {
        CentralizeData(data + i * input_dim_, centralized_data.data() + i * input_dim_);
    }

    // 3. get covariance matrix
    ComputeCovarianceMatrix(centralized_data.data(), count, covariance_matrix.data());

    // 4. eigen decomposition (stored in pca_matrix_)
    PerformEigenDecomposition(covariance_matrix.data());
}

void
PCATransformer::Transform(const float* input_vec, float* output_vec, uint8_t* meta) const {
    vsag::Vector<float> centralized_vec(allocator_);
    centralized_vec.resize(input_dim_, 0.0F);

    // centralize
    this->CentralizeData(input_vec, centralized_vec.data());

    // output_vec[i] = sum_j(input_vec[j] * pca_matrix_[j, i])
    // e.g., input_dim == 3, output_dim == 2
    //       [1, 0, 0,] * [1,]  = [1,]
    //       [0, 0, 1 ]   [2,]  = [3 ]
    //                    [3 ]
    BlasFunction::Sgemv(BlasFunction::RowMajor,
                        BlasFunction::NoTrans,
                        static_cast<int32_t>(output_dim_),
                        static_cast<int32_t>(input_dim_),
                        1.0F,
                        pca_matrix_.data(),
                        static_cast<int32_t>(input_dim_),
                        centralized_vec.data(),
                        1,
                        0.0F,
                        output_vec,
                        1);

    if (meta != nullptr) {
        // Preserve the existing serialized code layout and value. The metadata is
        // currently not consumed by RecoveryDistance.
        const float residual_norm = 0.0F;
        std::memcpy(meta, &residual_norm, sizeof(residual_norm));
    }
}

void
PCATransformer::InverseTransform(const float* input_vec, float* output_vec) const {
    throw VsagException(ErrorType::INTERNAL_ERROR, "InverseTransform not implement");
}

void
PCATransformer::Serialize(StreamWriter& writer) const {
    StreamWriter::WriteVector(writer, this->pca_matrix_);
    StreamWriter::WriteVector(writer, this->mean_);
}

void
PCATransformer::Deserialize(StreamReader& reader) {
    StreamReader::ReadVector(reader, this->pca_matrix_);
    StreamReader::ReadVector(reader, this->mean_);
}

void
PCATransformer::ComputeColumnMean(const float* data, uint64_t count) {
    std::fill(mean_.begin(), mean_.end(), 0.0F);

    for (uint64_t i = 0; i < count; ++i) {
        for (uint64_t j = 0; j < input_dim_; ++j) {
            mean_[j] += data[i * input_dim_ + j];
        }
    }

    for (uint64_t j = 0; j < input_dim_; ++j) {
        mean_[j] /= static_cast<float>(count);
    }
}

void
PCATransformer::CentralizeData(const float* original_data, float* centralized_data) const {
    for (uint64_t j = 0; j < input_dim_; ++j) {
        centralized_data[j] = original_data[j] - mean_[j];
    }
}

void
PCATransformer::ComputeCovarianceMatrix(const float* centralized_data,
                                        uint64_t count,
                                        float* covariance_matrix) const {
    // C = X^T X / (count - 1). Calling the configured BLAS backend is important for
    // full-dimensional SAQ training; the former scalar triple loop dominated GIST-sized inputs.
    const auto dim = static_cast<int32_t>(input_dim_);
    const auto max_chunk = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    uint64_t processed = 0;
    while (processed < count) {
        const auto chunk = static_cast<int32_t>(std::min(count - processed, max_chunk));
        const float beta = processed == 0 ? 0.0F : 1.0F;
        BlasFunction::Sgemm(BlasFunction::RowMajor,
                            BlasFunction::Trans,
                            BlasFunction::NoTrans,
                            dim,
                            dim,
                            chunk,
                            1.0F / static_cast<float>(count - 1),
                            centralized_data + processed * input_dim_,
                            dim,
                            centralized_data + processed * input_dim_,
                            dim,
                            beta,
                            covariance_matrix,
                            dim);
        processed += static_cast<uint64_t>(chunk);
    }
}

bool
PCATransformer::PerformEigenDecomposition(const float* covariance_matrix) {
    std::vector<float> eigen_values(input_dim_);
    std::vector<float> eigen_vectors(input_dim_ * input_dim_);
    std::copy(
        covariance_matrix, covariance_matrix + input_dim_ * input_dim_, eigen_vectors.begin());

    // 1. decomposition
    int32_t ssyev_result = BlasFunction::Ssyev(BlasFunction::RowMajor,
                                               BlasFunction::JobV,
                                               BlasFunction::Upper,
                                               static_cast<int32_t>(input_dim_),
                                               eigen_vectors.data(),
                                               static_cast<int32_t>(input_dim_),
                                               eigen_values.data());

    if (ssyev_result != 0) {
        logger::error(fmt::format("Error in ssyev: {}", ssyev_result));
        return false;
    }

    // LAPACKE stores eigenvectors column-wise even when the matrix buffer is row-major. The
    // eigenvalues are ascending, while PCA needs the eigenvectors as descending rows so that a
    // row-major matrix-vector product emits principal components in decreasing-variance order.
    for (uint64_t i = 0; i < output_dim_; ++i) {
        for (uint64_t j = 0; j < input_dim_; ++j) {
            pca_matrix_[i * input_dim_ + j] = eigen_vectors[j * input_dim_ + (input_dim_ - 1 - i)];
        }
    }
    return true;
}

void
PCATransformer::CopyPCAMatrix(float* out_pca_matrix) const {
    for (uint64_t i = 0; i < pca_matrix_.size(); i++) {
        out_pca_matrix[i] = pca_matrix_[i];
    }
}

void
PCATransformer::CopyMean(float* out_mean) const {
    for (uint64_t i = 0; i < mean_.size(); i++) {
        out_mean[i] = mean_[i];
    }
}

void
PCATransformer::SetMeanForTest(const float* input_mean) {
    for (uint64_t i = 0; i < mean_.size(); i++) {
        mean_[i] = input_mean[i];
    }
}

void
PCATransformer::SetPCAMatrixForTest(const float* input_pca_matrix) {
    for (uint64_t i = 0; i < pca_matrix_.size(); i++) {
        pca_matrix_[i] = input_pca_matrix[i];
    }
}

}  // namespace vsag
