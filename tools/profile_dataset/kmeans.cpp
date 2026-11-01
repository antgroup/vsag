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

#include "kmeans.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "deterministic_random.h"
#include "vector_math.h"

namespace vsag::dataset_profile {

namespace {

uint32_t
nearest_centroid(const float* row,
                 const std::vector<float>& centroids,
                 uint64_t k,
                 uint64_t dim,
                 double& best_distance) {
    uint32_t best = 0;
    best_distance = std::numeric_limits<double>::infinity();
    for (uint64_t c = 0; c < k; ++c) {
        const double distance = l2_sqr(row, centroids.data() + c * dim, dim);
        if (distance < best_distance) {
            best_distance = distance;
            best = static_cast<uint32_t>(c);
        }
    }
    return best;
}

}  // namespace

KMeansResult
run_kmeans(const std::vector<float>& data,
           uint64_t rows,
           uint64_t dim,
           uint64_t k,
           uint64_t max_iterations,
           uint64_t seed) {
    KMeansResult result;
    k = std::min(k, rows);
    result.k = k;
    if (k == 0 || dim == 0) {
        return result;
    }
    DeterministicRng rng(seed);
    result.centroids.assign(k * dim, 0.0F);
    std::vector<uint8_t> chosen(rows, 0);
    std::vector<double> nearest(rows, std::numeric_limits<double>::infinity());
    uint64_t first = rng.Uniform(rows);
    std::copy_n(data.data() + first * dim, dim, result.centroids.data());
    chosen[first] = 1;
    for (uint64_t c = 1; c < k; ++c) {
        const float* previous = result.centroids.data() + (c - 1) * dim;
        double total = 0.0;
        for (uint64_t r = 0; r < rows; ++r) {
            nearest[r] = std::min(nearest[r], l2_sqr(data.data() + r * dim, previous, dim));
            total += nearest[r];
        }
        uint64_t pick = rows;
        if (total > 0.0) {
            const double target = rng.UniformDouble() * total;
            double cumulative = 0.0;
            for (uint64_t r = 0; r < rows; ++r) {
                cumulative += nearest[r];
                if (cumulative > target && nearest[r] > 0.0) {
                    pick = r;
                    break;
                }
            }
        }
        if (pick == rows) {
            // Every remaining row coincides with a centroid; take the first unused row.
            for (uint64_t r = 0; r < rows; ++r) {
                if (chosen[r] == 0) {
                    pick = r;
                    break;
                }
            }
        }
        chosen[pick] = 1;
        std::copy_n(data.data() + pick * dim, dim, result.centroids.data() + c * dim);
    }

    result.assignments.assign(rows, 0);
    std::vector<double> distances(rows, 0.0);
    std::vector<double> sums(k * dim);
    result.sizes.assign(k, 0);
    const auto signed_rows = static_cast<int64_t>(rows);
    for (uint64_t iteration = 0; iteration < max_iterations; ++iteration) {
        uint64_t changed = 0;
        // clang-format off
#pragma omp parallel for schedule(static) reduction(+ : changed) default(none) \
    shared(data, result, distances, k, dim, iteration, signed_rows)
        // clang-format on
        for (int64_t r = 0; r < signed_rows; ++r) {
            const auto row = static_cast<uint64_t>(r);
            double distance = 0.0;
            const uint32_t best =
                nearest_centroid(data.data() + row * dim, result.centroids, k, dim, distance);
            if (iteration == 0 || best != result.assignments[row]) {
                ++changed;
            }
            result.assignments[row] = best;
            distances[row] = distance;
        }
        result.iterations = iteration + 1;
        if (changed == 0) {
            result.converged = true;
            break;
        }
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(result.sizes.begin(), result.sizes.end(), 0);
        for (uint64_t r = 0; r < rows; ++r) {
            const uint32_t cluster = result.assignments[r];
            ++result.sizes[cluster];
            for (uint64_t d = 0; d < dim; ++d) {
                sums[cluster * dim + d] += data[r * dim + d];
            }
        }
        for (uint64_t c = 0; c < k; ++c) {
            if (result.sizes[c] == 0) {
                uint64_t farthest = 0;
                for (uint64_t r = 1; r < rows; ++r) {
                    if (distances[r] > distances[farthest]) {
                        farthest = r;
                    }
                }
                std::copy_n(data.data() + farthest * dim, dim, result.centroids.data() + c * dim);
                distances[farthest] = 0.0;
                continue;
            }
            for (uint64_t d = 0; d < dim; ++d) {
                result.centroids[c * dim + d] =
                    static_cast<float>(sums[c * dim + d] / static_cast<double>(result.sizes[c]));
            }
        }
    }
    std::fill(result.sizes.begin(), result.sizes.end(), 0);
    double inertia = 0.0;
    for (uint64_t r = 0; r < rows; ++r) {
        double distance = 0.0;
        result.assignments[r] =
            nearest_centroid(data.data() + r * dim, result.centroids, k, dim, distance);
        ++result.sizes[result.assignments[r]];
        inertia += distance;
    }
    result.inertia = inertia / static_cast<double>(rows);
    return result;
}

double
silhouette_score(const std::vector<float>& data,
                 uint64_t rows,
                 uint64_t dim,
                 const std::vector<uint32_t>& assignments,
                 uint64_t k) {
    if (rows < 2 || k < 2) {
        return 0.0;
    }
    std::vector<double> scores(rows, 0.0);
    const auto signed_rows = static_cast<int64_t>(rows);
    // clang-format off
#pragma omp parallel for schedule(dynamic, 16) default(none) \
    shared(data, rows, dim, assignments, k, scores, signed_rows)
    // clang-format on
    for (int64_t i = 0; i < signed_rows; ++i) {
        const auto row = static_cast<uint64_t>(i);
        std::vector<double> sum(k, 0.0);
        std::vector<uint64_t> count(k, 0);
        for (uint64_t j = 0; j < rows; ++j) {
            if (j == row) {
                continue;
            }
            sum[assignments[j]] +=
                std::sqrt(l2_sqr(data.data() + row * dim, data.data() + j * dim, dim));
            ++count[assignments[j]];
        }
        const uint32_t own = assignments[row];
        if (count[own] == 0) {
            scores[row] = 0.0;
            continue;
        }
        const double a = sum[own] / static_cast<double>(count[own]);
        double b = std::numeric_limits<double>::infinity();
        for (uint64_t c = 0; c < k; ++c) {
            if (c != own && count[c] > 0) {
                b = std::min(b, sum[c] / static_cast<double>(count[c]));
            }
        }
        if (!std::isfinite(b)) {
            scores[row] = 0.0;
            continue;
        }
        const double denominator = std::max(a, b);
        scores[row] = denominator > 0.0 ? (b - a) / denominator : 0.0;
    }
    double total = 0.0;
    for (const double score : scores) {
        total += score;
    }
    return total / static_cast<double>(rows);
}

}  // namespace vsag::dataset_profile
