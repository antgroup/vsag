
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

namespace vsag {

void
RandomOrthogonalMatrix::CopyOrthogonalMatrix(float* out_matrix) const {
    std::copy(orthogonal_matrix_.data(), orthogonal_matrix_.data() + dim_ * dim_, out_matrix);
}

void
RandomOrthogonalMatrix::Transform(const float* original_vec, float* transformed_vec) const {
    // perform matrix-vector multiplication: y = Q * x
    cblas_sgemv(CblasRowMajor,
                CblasNoTrans,
                static_cast<blasint>(dim_),
                static_cast<blasint>(dim_),
                1.0F,
                orthogonal_matrix_.data(),
                static_cast<blasint>(dim_),
                original_vec,
                1,
                0.0F,
                transformed_vec,
                1);
}

void
RandomOrthogonalMatrix::InverseTransform(const float* transformed_vec, float* original_vec) const {
    // perform matrix-vector multiplication: x = Q^T * y
    cblas_sgemv(CblasRowMajor,
                CblasTrans,
                static_cast<blasint>(dim_),
                static_cast<blasint>(dim_),
                1.0F,
                orthogonal_matrix_.data(),
                static_cast<blasint>(dim_),
                transformed_vec,
                1,
                0.0F,
                original_vec,
                1);
}

bool
RandomOrthogonalMatrix::GenerateRandomOrthogonalMatrix() {
    // generate a random matrix with elements following a standard normal distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0, 1.0);

    for (uint64_t i = 0; i < dim_ * dim_; ++i) {
        orthogonal_matrix_[i] = dist(gen);
    }

    // QR decomposition with LAPACK
    std::vector<float> tau(dim_, 0.0);
    auto lda = static_cast<blasint>(dim_);

    int sgeqrf_result = LAPACKE_sgeqrf(LAPACK_ROW_MAJOR,
                                       static_cast<blasint>(dim_),
                                       static_cast<blasint>(dim_),
                                       orthogonal_matrix_.data(),
                                       lda,
                                       tau.data());
    if (sgeqrf_result != 0) {
        logger::error(fmt::format("Error in sgeqrf: {}", sgeqrf_result));
        return false;
    }

    // generate Q matrix
    int sorgqr_result = LAPACKE_sorgqr(LAPACK_ROW_MAJOR,
                                       static_cast<blasint>(dim_),
                                       static_cast<blasint>(dim_),
                                       static_cast<blasint>(dim_),
                                       orthogonal_matrix_.data(),
                                       lda,
                                       tau.data());
    if (sorgqr_result != 0) {
        logger::error(fmt::format("Error in sorgqr: {}", sorgqr_result));
        return false;
    }

    // make sure the determinant of the matrix is +1 (to avoid reflections)
    double det = ComputeDeterminant();  // TODO(ZXY): use another way to compute det
    if (det < 0) {
        // invert the first column
        // TODO(ZXY): use SIMD to accelerate
        for (uint64_t i = 0; i < dim_; ++i) {
            orthogonal_matrix_[i * dim_] = -orthogonal_matrix_[i * dim_];
        }
    }

    return true;
}

void
RandomOrthogonalMatrix::GenerateRandomOrthogonalMatrixWithRetry() {
    for (uint64_t i = 0; i < generate_retries_; i++) {
        bool result_gen = GenerateRandomOrthogonalMatrix();
        if (result_gen) {
            break;
        }
        logger::warn(fmt::format("Retrying generating random orthogonal matrix: {} times", i + 1));
    }
}

double
RandomOrthogonalMatrix::ComputeDeterminant() const {
    // calculate determinants using LU decomposition
    // copy matrix
    std::vector<float> mat(orthogonal_matrix_.data(), orthogonal_matrix_.data() + dim_ * dim_);
    std::vector<int> ipiv(dim_);
    int sgetrf_result = LAPACKE_sgetrf(LAPACK_ROW_MAJOR,
                                       static_cast<blasint>(dim_),
                                       static_cast<blasint>(dim_),
                                       mat.data(),
                                       static_cast<blasint>(dim_),
                                       ipiv.data());
    if (sgetrf_result != 0) {
        logger::error(fmt::format("Error in sgetrf: {}", sgetrf_result));
        return 0;
    }

    double det = 1.0;
    int num_swaps = 0;
    for (uint64_t i = 0; i < dim_; ++i) {
        det *= mat[i * dim_ + i];
        if (ipiv[i] != i + 1) {
            num_swaps++;
        }
    }
    if (num_swaps % 2 != 0) {
        det = -det;
    }
    return det;
}

void
RandomOrthogonalMatrix::Serialize(StreamWriter& writer) {
    StreamWriter::WriteVector(writer, this->orthogonal_matrix_);
}

void
RandomOrthogonalMatrix::Deserialize(StreamReader& reader) {
    StreamReader::ReadVector(reader, this->orthogonal_matrix_);
}

}  // namespace vsag
