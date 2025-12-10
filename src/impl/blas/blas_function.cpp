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

#include "blas_function.h"

#include <cblas.h>
#include <lapacke.h>

namespace vsag {

void
BlasFunction::Saxpy(int32_t n, float alpha, const float* x, int32_t incx, float* y, int32_t incy) {
    cblas_saxpy(n, alpha, x, incx, y, incy);
}

void
BlasFunction::Sscal(int32_t n, float alpha, float* x, int32_t incx) {
    cblas_sscal(n, alpha, x, incx);
}

void
BlasFunction::Sgemv(int32_t order,
                    int32_t trans,
                    int32_t m,
                    int32_t n,
                    float alpha,
                    const float* a,
                    int32_t lda,
                    const float* x,
                    int32_t incx,
                    float beta,
                    float* y,
                    int32_t incy) {
    cblas_sgemv(static_cast<CBLAS_ORDER>(order),
                static_cast<CBLAS_TRANSPOSE>(trans),
                m,
                n,
                alpha,
                a,
                lda,
                x,
                incx,
                beta,
                y,
                incy);
}

void
BlasFunction::Sgemm(int32_t order,
                    int32_t transa,
                    int32_t transb,
                    int32_t m,
                    int32_t n,
                    int32_t k,
                    float alpha,
                    const float* a,
                    int32_t lda,
                    const float* b,
                    int32_t ldb,
                    float beta,
                    float* c,
                    int32_t ldc) {
    cblas_sgemm(static_cast<CBLAS_ORDER>(order),
                static_cast<CBLAS_TRANSPOSE>(transa),
                static_cast<CBLAS_TRANSPOSE>(transb),
                m,
                n,
                k,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c,
                ldc);
}

int32_t
BlasFunction::Sgeqrf(int32_t order, int32_t m, int32_t n, float* a, int32_t lda, float* tau) {
    return LAPACKE_sgeqrf(static_cast<lapack_int>(order),
                          static_cast<lapack_int>(m),
                          static_cast<lapack_int>(n),
                          a,
                          static_cast<lapack_int>(lda),
                          tau);
}

int32_t
BlasFunction::Sorgqr(
    int32_t order, int32_t m, int32_t n, int32_t k, float* a, int32_t lda, const float* tau) {
    return LAPACKE_sorgqr(static_cast<lapack_int>(order),
                          static_cast<lapack_int>(m),
                          static_cast<lapack_int>(n),
                          static_cast<lapack_int>(k),
                          a,
                          static_cast<lapack_int>(lda),
                          tau);
}

int32_t
BlasFunction::Sgetrf(int32_t order, int32_t m, int32_t n, float* a, int32_t lda, int32_t* ipiv) {
    return LAPACKE_sgetrf(static_cast<lapack_int>(order),
                          static_cast<lapack_int>(m),
                          static_cast<lapack_int>(n),
                          a,
                          static_cast<lapack_int>(lda),
                          reinterpret_cast<lapack_int*>(ipiv));
}

int32_t
BlasFunction::Ssyev(
    int32_t order, char jobz, char uplo, int32_t n, float* a, int32_t lda, float* w) {
    return LAPACKE_ssyev(static_cast<lapack_int>(order),
                         jobz,
                         uplo,
                         static_cast<lapack_int>(n),
                         a,
                         static_cast<lapack_int>(lda),
                         w);
}

}  // namespace vsag
