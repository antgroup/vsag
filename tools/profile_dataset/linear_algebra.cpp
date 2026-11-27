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

#include "linear_algebra.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vsag::dataset_profile {

namespace {

// Householder reduction of the symmetric matrix held in `v` (row-major) to tridiagonal form.
// On return d holds the diagonal, e the sub-diagonal (e[0] = 0) and v the orthogonal
// transformation. Derived from the Algol procedure tred2 (Bowdler, Martin, Reinsch and
// Wilkinson, Handbook for Automatic Computation, Vol. II) via EISPACK and JAMA.
void
tridiagonalize(std::vector<double>& v, uint64_t n, std::vector<double>& d, std::vector<double>& e) {
    auto at = [&v, n](uint64_t row, uint64_t column) -> double& { return v[row * n + column]; };
    for (uint64_t j = 0; j < n; ++j) {
        d[j] = at(n - 1, j);
    }
    for (uint64_t i = n - 1; i > 0; --i) {
        double scale = 0.0;
        double h = 0.0;
        for (uint64_t k = 0; k < i; ++k) {
            scale += std::abs(d[k]);
        }
        if (scale == 0.0) {
            e[i] = d[i - 1];
            for (uint64_t j = 0; j < i; ++j) {
                d[j] = at(i - 1, j);
                at(i, j) = 0.0;
                at(j, i) = 0.0;
            }
        } else {
            for (uint64_t k = 0; k < i; ++k) {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            double f = d[i - 1];
            double g = std::sqrt(h);
            if (f > 0) {
                g = -g;
            }
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            for (uint64_t j = 0; j < i; ++j) {
                e[j] = 0.0;
            }
            for (uint64_t j = 0; j < i; ++j) {
                f = d[j];
                at(j, i) = f;
                g = e[j] + at(j, j) * f;
                for (uint64_t k = j + 1; k + 1 <= i; ++k) {
                    g += at(k, j) * d[k];
                    e[k] += at(k, j) * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for (uint64_t j = 0; j < i; ++j) {
                e[j] /= h;
                f += e[j] * d[j];
            }
            const double hh = f / (h + h);
            for (uint64_t j = 0; j < i; ++j) {
                e[j] -= hh * d[j];
            }
            for (uint64_t j = 0; j < i; ++j) {
                f = d[j];
                g = e[j];
                for (uint64_t k = j; k + 1 <= i; ++k) {
                    at(k, j) -= (f * e[k] + g * d[k]);
                }
                d[j] = at(i - 1, j);
                at(i, j) = 0.0;
            }
        }
        d[i] = h;
    }
    // Accumulate the transformations.
    for (uint64_t i = 0; i + 1 < n; ++i) {
        at(n - 1, i) = at(i, i);
        at(i, i) = 1.0;
        const double h = d[i + 1];
        if (h != 0.0) {
            for (uint64_t k = 0; k <= i; ++k) {
                d[k] = at(k, i + 1) / h;
            }
            for (uint64_t j = 0; j <= i; ++j) {
                double g = 0.0;
                for (uint64_t k = 0; k <= i; ++k) {
                    g += at(k, i + 1) * at(k, j);
                }
                for (uint64_t k = 0; k <= i; ++k) {
                    at(k, j) -= g * d[k];
                }
            }
        }
        for (uint64_t k = 0; k <= i; ++k) {
            at(k, i + 1) = 0.0;
        }
    }
    for (uint64_t j = 0; j < n; ++j) {
        d[j] = at(n - 1, j);
        at(n - 1, j) = 0.0;
    }
    at(n - 1, n - 1) = 1.0;
    e[0] = 0.0;
}

// Implicit QL iterations on the tridiagonal matrix (EISPACK tql2). `vt` holds the
// transformation transposed (row j = column j of V) so the rotation loops stay contiguous.
void
tridiagonal_ql(std::vector<double>& vt,
               uint64_t n,
               std::vector<double>& d,
               std::vector<double>& e) {
    for (uint64_t i = 1; i < n; ++i) {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    double f = 0.0;
    double tst1 = 0.0;
    const double eps = std::ldexp(1.0, -52);
    constexpr uint64_t max_iterations = 300;
    for (uint64_t l = 0; l < n; ++l) {
        tst1 = std::max(tst1, std::abs(d[l]) + std::abs(e[l]));
        uint64_t m = l;
        while (m < n) {
            if (std::abs(e[m]) <= eps * tst1) {
                break;
            }
            ++m;
        }
        if (m == n) {
            m = n - 1;
        }
        if (m > l) {
            uint64_t iteration = 0;
            do {
                if (++iteration > max_iterations) {
                    throw std::runtime_error("symmetric eigen-decomposition did not converge");
                }
                double g = d[l];
                double p = (d[l + 1] - g) / (2.0 * e[l]);
                double r = std::hypot(p, 1.0);
                if (p < 0) {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                const double dl1 = d[l + 1];
                double h = g - d[l];
                for (uint64_t i = l + 2; i < n; ++i) {
                    d[i] -= h;
                }
                f += h;
                p = d[m];
                double c = 1.0;
                double c2 = c;
                double c3 = c;
                const double el1 = e[l + 1];
                double s = 0.0;
                double s2 = 0.0;
                for (uint64_t i = m; i-- > l;) {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = std::hypot(p, e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    double* column_next = vt.data() + (i + 1) * n;
                    double* column = vt.data() + i * n;
                    for (uint64_t k = 0; k < n; ++k) {
                        const double next = column_next[k];
                        column_next[k] = s * column[k] + c * next;
                        column[k] = c * column[k] - s * next;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
            } while (std::abs(e[l]) > eps * tst1);
        }
        d[l] += f;
        e[l] = 0.0;
    }
}

}  // namespace

void
symmetric_eigen(std::vector<double> matrix,
                uint64_t n,
                std::vector<double>& eigenvalues,
                std::vector<double>& eigenvectors) {
    if (matrix.size() != n * n) {
        throw std::invalid_argument("symmetric_eigen: matrix size does not match n");
    }
    eigenvalues.assign(n, 0.0);
    eigenvectors.assign(n * n, 0.0);
    if (n == 0) {
        return;
    }
    if (n == 1) {
        eigenvalues[0] = matrix[0];
        eigenvectors[0] = 1.0;
        return;
    }
    std::vector<double> off_diagonal(n, 0.0);
    tridiagonalize(matrix, n, eigenvalues, off_diagonal);
    std::vector<double> transposed(n * n);
    for (uint64_t row = 0; row < n; ++row) {
        for (uint64_t column = 0; column < n; ++column) {
            transposed[column * n + row] = matrix[row * n + column];
        }
    }
    tridiagonal_ql(transposed, n, eigenvalues, off_diagonal);
    // Sort ascending (selection sort keeps the original EISPACK tie handling).
    std::vector<uint64_t> order(n);
    for (uint64_t i = 0; i < n; ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&eigenvalues](uint64_t a, uint64_t b) {
        return eigenvalues[a] < eigenvalues[b];
    });
    std::vector<double> sorted_values(n);
    for (uint64_t j = 0; j < n; ++j) {
        sorted_values[j] = eigenvalues[order[j]];
        const double* source = transposed.data() + order[j] * n;
        for (uint64_t k = 0; k < n; ++k) {
            eigenvectors[k * n + j] = source[k];
        }
    }
    eigenvalues = std::move(sorted_values);
}

namespace {

double
dot_columns(const double* a, const double* b, uint64_t length) {
    double s0 = 0.0;
    double s1 = 0.0;
    double s2 = 0.0;
    double s3 = 0.0;
    uint64_t i = 0;
    for (; i + 4 <= length; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < length; ++i) {
        s0 += a[i] * b[i];
    }
    return (s0 + s1) + (s2 + s3);
}

std::vector<double>
oriented_axis(const std::vector<double>& eigenvectors, uint64_t n, uint64_t column) {
    std::vector<double> axis(n);
    uint64_t largest = 0;
    for (uint64_t k = 0; k < n; ++k) {
        axis[k] = eigenvectors[k * n + column];
        if (std::abs(axis[k]) > std::abs(axis[largest])) {
            largest = k;
        }
    }
    // Eigenvectors are defined up to sign; fix it so projections are reproducible.
    if (axis[largest] < 0) {
        for (auto& value : axis) {
            value = -value;
        }
    }
    return axis;
}

}  // namespace

PcaResult
compute_pca(const std::vector<float>& data, uint64_t rows, uint64_t dim) {
    PcaResult result;
    result.rows = rows;
    result.dim = dim;
    result.mean.assign(dim, 0.0);
    if (rows < 2 || dim == 0) {
        return result;
    }
    for (uint64_t r = 0; r < rows; ++r) {
        for (uint64_t d = 0; d < dim; ++d) {
            result.mean[d] += data[r * dim + d];
        }
    }
    for (auto& value : result.mean) {
        value /= static_cast<double>(rows);
    }
    std::vector<double> columns(dim * rows);
    for (uint64_t r = 0; r < rows; ++r) {
        for (uint64_t d = 0; d < dim; ++d) {
            columns[d * rows + r] = static_cast<double>(data[r * dim + d]) - result.mean[d];
        }
    }
    std::vector<double> covariance(dim * dim, 0.0);
    const auto denominator = static_cast<double>(rows - 1);
    const auto signed_dim = static_cast<int64_t>(dim);
    // clang-format off
#pragma omp parallel for schedule(dynamic, 1) default(none) \
    shared(columns, covariance, rows, dim, denominator, signed_dim)
    // clang-format on
    for (int64_t i = 0; i < signed_dim; ++i) {
        const auto row = static_cast<uint64_t>(i);
        for (uint64_t j = row; j < dim; ++j) {
            const double value =
                dot_columns(columns.data() + row * rows, columns.data() + j * rows, rows) /
                denominator;
            covariance[row * dim + j] = value;
            covariance[j * dim + row] = value;
        }
    }
    std::vector<double> eigenvalues;
    std::vector<double> eigenvectors;
    symmetric_eigen(std::move(covariance), dim, eigenvalues, eigenvectors);
    result.eigenvalues.resize(dim);
    for (uint64_t j = 0; j < dim; ++j) {
        result.eigenvalues[j] = std::max(0.0, eigenvalues[dim - 1 - j]);
    }
    result.axis1 = oriented_axis(eigenvectors, dim, dim - 1);
    result.axis2 =
        dim >= 2 ? oriented_axis(eigenvectors, dim, dim - 2) : std::vector<double>(dim, 0.0);
    return result;
}

void
project_2d(const PcaResult& pca, const float* row, double& x, double& y) {
    x = 0.0;
    y = 0.0;
    for (uint64_t d = 0; d < pca.dim; ++d) {
        const double centered = static_cast<double>(row[d]) - pca.mean[d];
        x += centered * pca.axis1[d];
        y += centered * pca.axis2[d];
    }
}

}  // namespace vsag::dataset_profile
