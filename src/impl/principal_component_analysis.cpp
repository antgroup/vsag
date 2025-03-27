
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

#include "principal_component_analysis.h"

namespace vsag {

bool
PrincipalComponentAnalysis::Train(const float* data, uint64_t count) {
    vsag::Vector<float> centralized_data(allocator_);
    centralized_data.resize(count * original_dim_, 0.0F);

    vsag::Vector<float> covariance_matrix(allocator_);
    covariance_matrix.resize(original_dim_ * original_dim_, 0.0F);

    // 1. compute mean (stored in mean_)
    ComputeColumnMean(data, count);

    // 2. centralize data
    for (uint64_t i = 0; i < count; ++i) {
        CentralizeData(data + i * original_dim_, centralized_data.data() + i * original_dim_);
    }

    // 3. get covariance matrix
    ComputeCovarianceMatrix(centralized_data.data(), count, covariance_matrix.data());

    // 4. eigen decomposition (stored in pca_matrix_)
    bool is_trained = PerformEigenDecomposition(covariance_matrix.data());
    return is_trained;
}

void
PrincipalComponentAnalysis::ComputeColumnMean(const float* data, uint64_t count) {
    std::fill(mean_.begin(), mean_.end(), 0.0F);

    for (uint64_t i = 0; i < count; ++i) {
        for (uint64_t j = 0; j < original_dim_; ++j) {
            mean_[j] += data[i * original_dim_ + j];
        }
    }

    for (uint64_t j = 0; j < original_dim_; ++j) {
        mean_[j] /= static_cast<float>(count);
    }
}

void
PrincipalComponentAnalysis::CentralizeData(const float* original_data,
                                           float* centralized_data) const {
    for (uint64_t j = 0; j < original_dim_; ++j) {
        centralized_data[j] = original_data[j] - mean_[j];
    }
}

void
PrincipalComponentAnalysis::ComputeCovarianceMatrix(const float* centralized_data,
                                                    uint64_t count,
                                                    float* covariance_matrix) const {
    for (uint64_t i = 0; i < count; ++i) {
        for (uint64_t j = 0; j < original_dim_; ++j) {
            for (uint64_t k = 0; k < original_dim_; ++k) {
                covariance_matrix[j * original_dim_ + k] +=
                    centralized_data[i * original_dim_ + j] *
                    centralized_data[i * original_dim_ + k];
            }
        }
    }

    // unbiased estimat
    float scale = 1.0F / static_cast<float>(count - 1);
    for (uint64_t j = 0; j < original_dim_; ++j) {
        for (uint64_t k = 0; k < original_dim_; ++k) {
            covariance_matrix[j * original_dim_ + k] *= scale;
        }
    }
}

bool
PrincipalComponentAnalysis::PerformEigenDecomposition(const float* covariance_matrix) {
    std::vector<float> eigen_values(original_dim_);
    std::vector<float> eigen_vectors(original_dim_ * original_dim_);
    std::copy(covariance_matrix,
              covariance_matrix + original_dim_ * original_dim_,
              eigen_vectors.begin());

    // 1. decomposition
    int ssyev_result = LAPACKE_ssyev(LAPACK_ROW_MAJOR,
                                     'V',
                                     'U',
                                     static_cast<blasint>(original_dim_),
                                     eigen_vectors.data(),
                                     static_cast<blasint>(original_dim_),
                                     eigen_values.data());

    if (ssyev_result != 0) {
        logger::error(fmt::format("Error in sgeqrf: {}", ssyev_result));
        return false;
    }

    // 2. pca_matrix_[i][original_dim_] = eigen_vectors[- 1 - i][original_dim_]
    for (uint64_t i = 0; i < target_dim_; ++i) {
        for (uint64_t j = 0; j < original_dim_; ++j) {
            pca_matrix_[i * original_dim_ + j] =
                eigen_vectors[(original_dim_ - 1 - i) * original_dim_ + j];
        }
    }
    return true;
}

void
PrincipalComponentAnalysis::Transform(const float* original_vec, float* transformed_vec) const {
    vsag::Vector<float> centralized_vec(allocator_);
    centralized_vec.resize(original_dim_, 0.0F);

    // centralize
    this->CentralizeData(original_vec, centralized_vec.data());

    // transformed_vec[i] = sum_j(original_vec[j] * pca_matrix_[j, i])
    // e.g., original_dim == 3, target_dim == 2
    //       [1, 0, 0,] * [1,]  = [1,]
    //       [0, 0, 1 ]   [2,]  = [3 ]
    //                    [3 ]
    cblas_sgemv(CblasRowMajor,
                CblasNoTrans,
                static_cast<blasint>(target_dim_),
                static_cast<blasint>(original_dim_),
                1.0F,
                pca_matrix_.data(),
                static_cast<blasint>(original_dim_),
                centralized_vec.data(),
                1,
                0.0F,
                transformed_vec,
                1);
}

void
PrincipalComponentAnalysis::CopyPCAMatrixForTest(float* out_pca_matrix) const {
    for (uint64_t i = 0; i < pca_matrix_.size(); i++) {
        out_pca_matrix[i] = pca_matrix_[i];
    }
}

void
PrincipalComponentAnalysis::CopyMeanForTest(float* out_mean) const {
    for (uint64_t i = 0; i < mean_.size(); i++) {
        out_mean[i] = mean_[i];
    }
}

void
PrincipalComponentAnalysis::SetMeanForTest(const float* input_mean) {
    for (uint64_t i = 0; i < mean_.size(); i++) {
        mean_[i] = input_mean[i];
    }
}

void
PrincipalComponentAnalysis::SetPCAMatrixForText(const float* input_pca_matrix) {
    for (uint64_t i = 0; i < pca_matrix_.size(); i++) {
        pca_matrix_[i] = input_pca_matrix[i];
    }
}

void
PrincipalComponentAnalysis::Serialize(StreamWriter& writer) {
    StreamWriter::WriteVector(writer, this->pca_matrix_);
    StreamWriter::WriteVector(writer, this->mean_);
}

void
PrincipalComponentAnalysis::Deserialize(StreamReader& reader) {
    StreamReader::ReadVector(reader, this->pca_matrix_);
    StreamReader::ReadVector(reader, this->mean_);
}

}  // namespace vsag
