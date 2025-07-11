
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

#include "random_orthogonal_transformer.h"

#include <cblas.h>
#include <fmt/format.h>
#include <lapacke.h>

#include <random>

#include "../../logger.h"

namespace vsag {
RandomOrthogonalMatrix::RandomOrthogonalMatrix(Allocator* allocator, int64_t dim, uint64_t retries)
    : VectorTransformer(allocator, dim), orthogonal_matrix_(allocator), generate_retries_(retries) {
    orthogonal_matrix_.resize(dim * dim);
    this->type_ = VectorTransformerType::RANDOM_ORTHOGONAL;
}

void
RandomOrthogonalMatrix::Train(const float* data, uint64_t count) {
    // generate rom
    GenerateRandomOrthogonalMatrixWithRetry();

    // validate rom
    int retries = MAX_RETRIES;
    bool successful_gen = true;
    const float delta = 1e-4;
    double det = ComputeDeterminant();
    if (std::fabs(det - 1) > delta) {
        for (uint64_t i = 0; i < retries; i++) {
            successful_gen = GenerateRandomOrthogonalMatrix();
            if (successful_gen) {
                break;
            }
        }
    }
}

void
RandomOrthogonalMatrix::CopyOrthogonalMatrix(float* out_matrix) const {
    std::copy(orthogonal_matrix_.data(),
              orthogonal_matrix_.data() + this->input_dim_ * this->output_dim_,
              out_matrix);
}

void
RandomOrthogonalMatrix::Transform(const float* original_vec, float* transformed_vec) const {
    // perform matrix-vector multiplication: y = Q * x
    auto dim = static_cast<blasint>(this->input_dim_);
    cblas_sgemv(CblasRowMajor,
                CblasNoTrans,
                static_cast<blasint>(dim),
                static_cast<blasint>(dim),
                1.0F,
                orthogonal_matrix_.data(),
                static_cast<blasint>(dim),
                original_vec,
                1,
                0.0F,
                transformed_vec,
                1);
}

void
RandomOrthogonalMatrix::InverseTransform(const float* transformed_vec, float* original_vec) const {
    // perform matrix-vector multiplication: x = Q^T * y
    auto dim = static_cast<blasint>(this->input_dim_);
    cblas_sgemv(CblasRowMajor,
                CblasTrans,
                static_cast<blasint>(dim),
                static_cast<blasint>(dim),
                1.0F,
                orthogonal_matrix_.data(),
                static_cast<blasint>(dim),
                transformed_vec,
                1,
                0.0F,
                original_vec,
                1);
}

bool
RandomOrthogonalMatrix::GenerateRandomOrthogonalMatrix() {
    auto dim = static_cast<uint64_t>(this->input_dim_);
    // generate a random matrix with elements following a standard normal distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0, 1.0);

    for (uint64_t i = 0; i < dim * dim; ++i) {
        orthogonal_matrix_[i] = dist(gen);
    }

    // QR decomposition with LAPACK
    std::vector<float> tau(dim, 0.0);
    auto lda = static_cast<blasint>(dim);

    int sgeqrf_result = LAPACKE_sgeqrf(LAPACK_ROW_MAJOR,
                                       static_cast<blasint>(dim),
                                       static_cast<blasint>(dim),
                                       orthogonal_matrix_.data(),
                                       lda,
                                       tau.data());
    if (sgeqrf_result != 0) {
        logger::error(fmt::format("Error in sgeqrf: {}", sgeqrf_result));
        return false;
    }

    // generate Q matrix
    int sorgqr_result = LAPACKE_sorgqr(LAPACK_ROW_MAJOR,
                                       static_cast<blasint>(dim),
                                       static_cast<blasint>(dim),
                                       static_cast<blasint>(dim),
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
        for (uint64_t i = 0; i < dim; ++i) {
            orthogonal_matrix_[i * dim] = -orthogonal_matrix_[i * dim];
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
    auto dim = static_cast<uint64_t>(this->input_dim_);
    // calculate determinants using LU decomposition
    // copy matrix
    std::vector<float> mat(orthogonal_matrix_.data(), orthogonal_matrix_.data() + dim * dim);
    std::vector<int> ipiv(dim);
    int sgetrf_result = LAPACKE_sgetrf(LAPACK_ROW_MAJOR,
                                       static_cast<blasint>(dim),
                                       static_cast<blasint>(dim),
                                       mat.data(),
                                       static_cast<blasint>(dim),
                                       ipiv.data());
    if (sgetrf_result != 0) {
        logger::error(fmt::format("Error in sgetrf: {}", sgetrf_result));
        return 0;
    }

    double det = 1.0;
    int num_swaps = 0;
    for (uint64_t i = 0; i < dim; ++i) {
        det *= mat[i * dim + i];
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
RandomOrthogonalMatrix::Serialize(StreamWriter& writer) const {
    StreamWriter::WriteVector(writer, this->orthogonal_matrix_);
}

void
RandomOrthogonalMatrix::Deserialize(StreamReader& reader) {
    StreamReader::ReadVector(reader, this->orthogonal_matrix_);
}

}  // namespace vsag
