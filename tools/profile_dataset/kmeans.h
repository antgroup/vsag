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

struct KMeansResult {
    uint64_t k{0};
    uint64_t iterations{0};
    bool converged{false};
    /// Cluster index per input row.
    std::vector<uint32_t> assignments;
    std::vector<uint64_t> sizes;
    /// Mean squared Euclidean distance to the assigned centroid.
    double inertia{0.0};
    std::vector<float> centroids;
};

/// Lloyd's k-means with k-means++ seeding. Seeding draws come from DeterministicRng(seed);
/// assignments break ties by the lowest centroid index and centroid sums are accumulated in
/// row order, so the result is reproducible for any thread count. Empty clusters are
/// re-seeded with the row farthest from its centroid.
KMeansResult
run_kmeans(const std::vector<float>& data,
           uint64_t rows,
           uint64_t dim,
           uint64_t k,
           uint64_t max_iterations,
           uint64_t seed);

/// Mean silhouette coefficient (Euclidean) of the first `rows` rows; O(rows^2 * dim).
double
silhouette_score(const std::vector<float>& data,
                 uint64_t rows,
                 uint64_t dim,
                 const std::vector<uint32_t>& assignments,
                 uint64_t k);

}  // namespace vsag::dataset_profile
