
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
    RandomOrthogonalMatrix(uint32_t dim, Allocator* allocator) : dim_(dim), allocator_(allocator) {
        orthogonal_matrix_ = (double*)allocator_->Allocate(sizeof(double) * dim_ * dim_);
        GenerateRandomOrthogonalMatrix();
    }

    ~RandomOrthogonalMatrix() {
        allocator_->Deallocate(orthogonal_matrix_);
    }

    void
    GetOrthogonalMatrix(double* out_matrix) const {
        std::copy(orthogonal_matrix_, orthogonal_matrix_ + dim_ * dim_, out_matrix);
    }

    void
    Transform(float* vec) const {
        // random projection

        std::vector<double> vec_double(dim_);  // OpenBLAS and LAPACK use double vector
        for (uint32_t i = 0; i < dim_; ++i) {
            vec_double[i] = static_cast<double>(vec[i]);
        }

        // perform matrix-vector multiplication: y = Q * x
        std::vector<double> result(dim_, 0.0);
        cblas_dgemv(CblasRowMajor,
                    CblasNoTrans,
                    dim_,
                    dim_,
                    1.0,
                    orthogonal_matrix_,
                    dim_,
                    vec_double.data(),
                    1,
                    0.0,
                    result.data(),
                    1);

        // save result
        for (uint32_t i = 0; i < dim_; ++i) {
            vec[i] = static_cast<float>(result[i]);
        }
    }

    bool
    GenerateRandomOrthogonalMatrix() {
        // generate a random matrix with elements following a standard normal distribution
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, 1.0);

        for (uint32_t i = 0; i < dim_ * dim_; ++i) {
            orthogonal_matrix_[i] = dist(gen);
        }

        // QR decomposition with LAPACK
        std::vector<double> tau(dim_, 0.0);
        int lda = dim_;
        int info;

        info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, dim_, dim_, orthogonal_matrix_, lda, tau.data());
        if (info != 0) {
            logger::error(fmt::format("Error in dgeqrf: {}", info));
            return false;
        }

        // generate Q matrix
        info =
            LAPACKE_dorgqr(LAPACK_ROW_MAJOR, dim_, dim_, dim_, orthogonal_matrix_, lda, tau.data());
        if (info != 0) {
            logger::error(fmt::format("Error in dorgqr: {}", info));
            return false;
        }

        // make sure the determinant of the matrix is +1 (to avoid reflections)
        double det = ComputeDeterminant();
        if (det < 0) {
            // Invert the first column
            for (uint32_t i = 0; i < dim_; ++i) {
                orthogonal_matrix_[i * dim_] = -orthogonal_matrix_[i * dim_];
            }
        }

        return true;
    }

    double
    ComputeDeterminant() const {
        // calculate determinants using LU decomposition
        std::vector<double> mat(dim_ * dim_, 0);  // copy matrix
        for (uint32_t i = 0; i < dim_ * dim_; i++) {
            mat[i] = orthogonal_matrix_[i];
        }
        std::vector<int> ipiv(dim_);
        int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, dim_, dim_, mat.data(), dim_, ipiv.data());
        if (info != 0) {
            logger::error(fmt::format("Error in dgetrf: {}", info));
            return false;
        }

        double det = 1.0;
        int num_swaps = 0;
        for (uint32_t i = 0; i < dim_; ++i) {
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

private:
    Allocator* const allocator_{nullptr};

    const int32_t dim_{0};

    double* orthogonal_matrix_{nullptr};  // OpenBLAS and LAPACK use double vector
};

}  // namespace vsag
