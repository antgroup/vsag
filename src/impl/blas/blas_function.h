
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

#include <cstdint>

namespace vsag {

class BlasFunction {
public:
    /**
     * @brief Perform the operation y := alpha * x + y.
     * 
     * @param n Number of elements in vector x and y.
     * @param alpha Scaling factor for vector x.
     * @param x Pointer to the input vector x.
     * @param incx Stride of vector x (use 1 for contiguous storage).
     * @param y Pointer to the input/output vector y.
     * @param incy Stride of vector y (use 1 for contiguous storage).
     */
    static void
    Saxpy(int32_t n, float alpha, const float* x, int32_t incx, float* y, int32_t incy);

    /**
     * @brief Scale vector x by alpha.
     * 
     * @param n Number of elements in vector x.
     * @param alpha Scaling factor.
     * @param x Pointer to the input/output vector x.
     * @param incx Stride of vector x (use 1 for contiguous storage).
     */
    static void
    Sscal(int32_t n, float alpha, float* x, int32_t incx);

    /**
     * @brief Perform the matrix-vector operation y := alpha * A * x + beta * y.
     * 
     * @param order Specifies the matrix storage layout (RowMajor or ColMajor).
     * @param trans Specifies the operation (NoTrans, Trans, or ConjTrans).
     * @param m Number of rows in matrix A.
     * @param n Number of columns in matrix A.
     * @param alpha Scaling factor for matrix A.
     * @param a Pointer to the input matrix A.
     * @param lda Leading dimension of matrix A (use m for RowMajor, use n for ColMajor).
     * @param x Pointer to the input vector x.
     * @param incx Stride of vector x (use 1 for contiguous storage).
     * @param beta Scaling factor for vector y.
     * @param y Pointer to the input/output vector y.
     * @param incy Stride of vector y (use 1 for contiguous storage).
     */
    static void
    Sgemv(int32_t order,
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
          int32_t incy);

    /**
     * @brief Perform the matrix-matrix operation C := alpha * A * B + beta * C.
     * 
     * @param order Specifies the matrix storage layout (RowMajor or ColMajor).
     * @param transa Specifies the operation for matrix A (NoTrans, Trans, or ConjTrans).
     * @param transb Specifies the operation for matrix B (NoTrans, Trans, or ConjTrans).
     * @param m Number of rows in matrix A.
     * @param n Number of columns in matrix B.
     * @param k Number of columns in matrix A (if transa == NoTrans, else number of rows in A).
     * @param alpha Scaling factor for matrices A and B.
     * @param a Pointer to the input matrix A.
     * @param lda Leading dimension of matrix A (use m for RowMajor, use k for ColMajor).
     * @param b Pointer to the input matrix B.
     * @param ldb Leading dimension of matrix B (use n for RowMajor, use k for ColMajor).
     * @param beta Scaling factor for matrix C.
     * @param c Pointer to the input/output matrix C.
     * @param ldc Leading dimension of matrix C (use m for RowMajor, use n for ColMajor).
     */
    static void
    Sgemm(int32_t order,
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
          int32_t ldc);

    /**
     * @brief Compute the QR factorization of a matrix A using the Gram-Schmidt process.
     * 
     * @param order Specifies the matrix storage layout (RowMajor or ColMajor).
     * @param m Number of rows in matrix A.
     * @param n Number of columns in matrix A.
     * @param a Pointer to the input/output matrix A.
     * @param lda Leading dimension of matrix A (use m for RowMajor, use n for ColMajor).
     * @param tau Pointer to the output array tau of size min(m, n).
     * @return int32_t Error code (0 for success).
     */
    static int32_t
    Sgeqrf(int32_t order, int32_t m, int32_t n, float* a, int32_t lda, float* tau);

    /**
     * @brief Compute the orthogonal matrix Q from the QR factorization computed by Sgeqrf.
     * 
     * @param order Specifies the matrix storage layout (RowMajor or ColMajor).
     * @param m Number of rows in matrix A.
     * @param n Number of columns in matrix A.
     * @param k Number of columns in matrix A (if transa == NoTrans, else number of rows in A).
     * @param a Pointer to the input/output matrix A.
     * @param lda Leading dimension of matrix A (use m for RowMajor, use n for ColMajor).
     * @param tau Pointer to the input array tau of size min(m, n).
     * @return int32_t Error code (0 for success).
     */
    static int32_t
    Sorgqr(int32_t order, int32_t m, int32_t n, int32_t k, float* a, int32_t lda, const float* tau);

    /**
     * @brief Compute the LU factorization of a matrix A using partial pivoting.
     * 
     * @param order Specifies the matrix storage layout (RowMajor or ColMajor).
     * @param m Number of rows in matrix A.
     * @param n Number of columns in matrix A.
     * @param a Pointer to the input/output matrix A.
     * @param lda Leading dimension of matrix A (use m for RowMajor, use n for ColMajor).
     * @param ipiv Pointer to the output array ipiv of size max(m, n).
     * @return int32_t Error code (0 for success).
     */
    static int32_t
    Sgetrf(int32_t order, int32_t m, int32_t n, float* a, int32_t lda, int32_t* ipiv);

    /**
     * @brief Compute the eigenvalues and, optionally, the eigenvectors of a symmetric matrix A.
     * 
     * @param order Specifies the matrix storage layout (RowMajor or ColMajor).
     * @param jobz Specifies whether to compute eigenvalues only (N) or eigenvalues and eigenvectors (V).
     * @param uplo Specifies whether the upper or lower triangle of A is stored (U or L).
     * @param n Number of rows and columns in matrix A.
     * @param a Pointer to the input/output matrix A.
     * @param lda Leading dimension of matrix A (use n for RowMajor, use n for ColMajor).
     * @param w Pointer to the output array w of size n.
     * @return int32_t Error code (0 for success).
     */
    static int32_t
    Ssyev(int32_t order, char jobz, char uplo, int32_t n, float* a, int32_t lda, float* w);

    // Constants for BLAS operations
    static constexpr int32_t RowMajor = 101;   // Row-major storage
    static constexpr int32_t ColMajor = 102;   // Column-major storage
    static constexpr int32_t NoTrans = 111;    // No transpose
    static constexpr int32_t Trans = 112;      // Transpose
    static constexpr int32_t ConjTrans = 113;  // Conjugate transpose

    // LAPACK specific constants
    static constexpr char JobV = 'V';   // Compute eigenvectors
    static constexpr char JobN = 'N';   // Do not compute eigenvectors
    static constexpr char Upper = 'U';  // Upper triangular
    static constexpr char Lower = 'L';  // Lower triangular
};

}  // namespace vsag
