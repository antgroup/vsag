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

#include <cmath>
#include <vector>

#include "unittest.h"
using namespace vsag;

static constexpr float EPSILON = 1e-4F;

TEST_CASE("Saxpy Basic Test", "[ut][BlasFunction]") {
    int32_t n = 10;
    float alpha = 2.0F;
    std::vector<float> x(n), y(n), expected(n);

    for (int32_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i * 2);
        expected[i] = alpha * x[i] + y[i];
    }

    BlasFunction::Saxpy(n, alpha, x.data(), 1, y.data(), 1);

    for (int32_t i = 0; i < n; ++i) {
        REQUIRE(std::abs(y[i] - expected[i]) < EPSILON);
    }
}

TEST_CASE("Sscal Basic Test", "[ut][BlasFunction]") {
    int32_t n = 10;
    float alpha = 3.0F;
    std::vector<float> x(n), expected(n);

    for (int32_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i);
        expected[i] = alpha * x[i];
    }

    BlasFunction::Sscal(n, alpha, x.data(), 1);

    for (int32_t i = 0; i < n; ++i) {
        REQUIRE(std::abs(x[i] - expected[i]) < EPSILON);
    }
}

TEST_CASE("Sgemv Basic Test", "[ut][BlasFunction]") {
    int32_t m = 3;
    int32_t n = 4;
    float alpha = 1.0F;
    float beta = 0.0F;

    std::vector<float> a(m * n);
    std::vector<float> x(n);
    std::vector<float> y(m, 0.0F);
    std::vector<float> expected(m, 0.0F);

    for (int32_t i = 0; i < m * n; ++i) {
        a[i] = static_cast<float>(i);
    }
    for (int32_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i + 1);
    }

    for (int32_t i = 0; i < m; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            expected[i] += a[i * n + j] * x[j];
        }
    }

    BlasFunction::Sgemv(BlasFunction::RowMajor,
                        BlasFunction::NoTrans,
                        m,
                        n,
                        alpha,
                        a.data(),
                        n,
                        x.data(),
                        1,
                        beta,
                        y.data(),
                        1);

    for (int32_t i = 0; i < m; ++i) {
        REQUIRE(std::abs(y[i] - expected[i]) < EPSILON);
    }
}

TEST_CASE("Sgemm Basic Test", "[ut][BlasFunction]") {
    int32_t m = 2;
    int32_t n = 3;
    int32_t k = 4;
    float alpha = 1.0F;
    float beta = 0.0F;

    std::vector<float> a(m * k);
    std::vector<float> b(k * n);
    std::vector<float> c(m * n, 0.0F);
    std::vector<float> expected(m * n, 0.0F);

    for (int32_t i = 0; i < m * k; ++i) {
        a[i] = static_cast<float>(i + 1);
    }
    for (int32_t i = 0; i < k * n; ++i) {
        b[i] = static_cast<float>(i + 1);
    }

    for (int32_t i = 0; i < m; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            for (int32_t l = 0; l < k; ++l) {
                expected[i * n + j] += a[i * k + l] * b[l * n + j];
            }
        }
    }

    BlasFunction::Sgemm(BlasFunction::RowMajor,
                        BlasFunction::NoTrans,
                        BlasFunction::NoTrans,
                        m,
                        n,
                        k,
                        alpha,
                        a.data(),
                        k,
                        b.data(),
                        n,
                        beta,
                        c.data(),
                        n);

    for (int32_t i = 0; i < m * n; ++i) {
        REQUIRE(std::abs(c[i] - expected[i]) < EPSILON);
    }
}

TEST_CASE("BlasFunction Constants Test", "[ut][BlasFunction]") {
    REQUIRE(BlasFunction::RowMajor == 101);
    REQUIRE(BlasFunction::ColMajor == 102);
    REQUIRE(BlasFunction::NoTrans == 111);
    REQUIRE(BlasFunction::Trans == 112);
    REQUIRE(BlasFunction::ConjTrans == 113);
    REQUIRE(BlasFunction::JobV == 'V');
    REQUIRE(BlasFunction::JobN == 'N');
    REQUIRE(BlasFunction::Upper == 'U');
    REQUIRE(BlasFunction::Lower == 'L');
}
