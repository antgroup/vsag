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
#include <vector>

namespace vsag::dataset_profile {

/// Eigen-decomposition of a dense symmetric matrix (row-major, n x n) with Householder
/// tridiagonalization and the implicit QL algorithm (EISPACK tred2/tql2). It is serial and
/// double precision, so results do not depend on BLAS threading.
/// On return `eigenvalues` is ascending and column j of `eigenvectors` (row-major n x n)
/// is the unit eigenvector of eigenvalues[j].
void
symmetric_eigen(std::vector<double> matrix,
                uint64_t n,
                std::vector<double>& eigenvalues,
                std::vector<double>& eigenvectors);

struct PcaResult {
    uint64_t rows{0};
    uint64_t dim{0};
    std::vector<double> mean;
    /// Descending covariance eigenvalues (negative round-off clipped to 0).
    std::vector<double> eigenvalues;
    /// Unit principal axes for the two largest eigenvalues (dim floats each).
    std::vector<double> axis1;
    std::vector<double> axis2;
};

/// PCA of `rows` row-major vectors of dimension `dim` (sample covariance, divides by n - 1).
PcaResult
compute_pca(const std::vector<float>& data, uint64_t rows, uint64_t dim);

/// Projects a row onto the two leading principal axes.
void
project_2d(const PcaResult& pca, const float* row, double& x, double& y);

}  // namespace vsag::dataset_profile
