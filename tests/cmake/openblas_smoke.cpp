
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

#include <cblas.h>
#include <lapacke.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

static void
check(bool success) {
    if (!success) {
        std::abort();
    }
}

static bool
near(float a, float b) {
    return std::fabs(a - b) < 1e-4F;
}

int
main() {
    static_assert(sizeof(lapack_int) == sizeof(int32_t), "VSAG needs LP64 LAPACK integers");
    const char* config = openblas_get_config();
    std::puts(config);
    check(std::strstr(config, "OpenBLAS 0.3.34") != nullptr);
    check(std::strstr(config, "USE64BITINT") == nullptr);
    check(openblas_get_parallel() == 0);
    float x[] = {1, 2};
    float y[] = {3, 4};
    cblas_saxpy(2, 2, x, 1, y, 1);
    check(near(y[0], 5) && near(y[1], 8));
    cblas_sscal(2, 2, x, 1);
    check(near(x[0], 2) && near(x[1], 4));
    float a[] = {1, 2, 3, 4};
    float v[] = {1, 1};
    float out[] = {0, 0};
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 2, 2, 1, a, 2, v, 1, 0, out, 1);
    check(near(out[0], 3) && near(out[1], 7));
    float product[4] = {};
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1, a, 2, a, 2, 0, product, 2);
    check(near(product[0], 7) && near(product[3], 22));
    float qr[] = {1, 2, 3, 4};
    float tau[2];
    check(LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, 2, 2, qr, 2, tau) == 0);
    check(LAPACKE_sorgqr(LAPACK_ROW_MAJOR, 2, 2, 2, qr, 2, tau) == 0);
    check(near(qr[0] * qr[0] + qr[2] * qr[2], 1));
    check(near(qr[0] * qr[1] + qr[2] * qr[3], 0));
    float lu[] = {4, 1, 2, 3};
    int32_t pivots[2];
    check(LAPACKE_sgetrf(LAPACK_ROW_MAJOR, 2, 2, lu, 2, pivots) == 0);
    check(near(lu[3], 2.5F));
    float symmetric[] = {2, 1, 1, 2};
    float eigenvalues[2];
    check(LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'V', 'U', 2, symmetric, 2, eigenvalues) == 0);
    check(near(eigenvalues[0], 1) && near(eigenvalues[1], 3));
    for (char uplo : {'U', 'L'}) {
        float positive[] = {4, 1, 1, 3};
        lapack_int pivot[2];
        lapack_int rank;
        check(LAPACKE_spstrf(LAPACK_ROW_MAJOR, uplo, 2, positive, 2, pivot, &rank, -1) == 0);
        check(rank == 2);
    }
    std::puts("All eight BLAS/LAPACKE operations passed");
}
