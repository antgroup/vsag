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
#include <limits>
#include <vector>

#include "profile_types.h"

namespace vsag::dataset_profile {

struct SparseBatch;

constexpr uint64_t NO_ROW = std::numeric_limits<uint64_t>::max();

struct Neighbor {
    double distance{0.0};
    uint64_t index{NO_ROW};
};

/// Strict total order: distance, then row index. The k smallest elements under this order
/// form a unique set, so results do not depend on scan order or thread scheduling.
inline bool
neighbor_less(const Neighbor& a, const Neighbor& b) {
    return a.distance < b.distance || (a.distance == b.distance && a.index < b.index);
}

/// Keeps the k smallest neighbors seen so far.
class TopK {
public:
    explicit TopK(uint64_t k = 0) : k_(k) {
        heap_.reserve(k);
    }

    void
    Push(double distance, uint64_t index);

    void
    Merge(const TopK& other);

    /// Ascending neighbors.
    [[nodiscard]] std::vector<Neighbor>
    Sorted() const;

private:
    uint64_t k_{0};
    std::vector<Neighbor> heap_;
};

/// Dense query rows with precomputed L2 norms. `self_rows[q]` is excluded from the results
/// of query q (NO_ROW when the query is not part of the scanned split).
struct DenseQuerySet {
    uint64_t dim{0};
    std::vector<float> vectors;
    std::vector<double> norms;
    std::vector<uint64_t> self_rows;

    [[nodiscard]] uint64_t
    Size() const {
        return norms.size();
    }
};

/// Exact k-NN of a bounded query set against a streamed split: memory is O(queries * k),
/// never O(N^2). Distances use the VSAG metric convention; for ip an extra Euclidean list
/// feeds the geometric estimators.
class DenseKnnSearch {
public:
    DenseKnnSearch(const DenseQuerySet& queries, Metric metric, uint64_t k);

    /// Scans `count` rows starting at global row `first_row`. Rows with valid[r] == 0 are
    /// ignored (non-finite values, or zero vectors under cosine).
    void
    Scan(const float* rows,
         const double* norms,
         const uint8_t* valid,
         uint64_t first_row,
         uint64_t count);

    /// Ascending metric-distance neighbors per query.
    [[nodiscard]] std::vector<std::vector<Neighbor>>
    MetricNeighbors() const;

    /// Ascending geometric (Euclidean / chord) distances per query.
    [[nodiscard]] std::vector<std::vector<double>>
    GeometricDistances() const;

private:
    const DenseQuerySet& queries_;
    Metric metric_;
    uint64_t k_;
    std::vector<TopK> metric_heaps_;
    std::vector<TopK> euclidean_heaps_;
};

/// Sparse query rows (ids ascending per row) with L2 norms.
struct SparseQuerySet {
    std::vector<uint64_t> offsets{0};
    std::vector<uint32_t> ids;
    std::vector<float> values;
    std::vector<double> norms;
    std::vector<uint64_t> self_rows;

    [[nodiscard]] uint64_t
    Size() const {
        return offsets.size() - 1;
    }
};

/// Exact k-NN for sparse ip/cosine through an inverted index over the query terms. Rows that
/// share no term with a query have inner product 0; missing neighbors are therefore padded
/// with that exact distance (1 for both ip and cosine).
class SparseKnnSearch {
public:
    SparseKnnSearch(const SparseQuerySet& queries, Metric metric, uint64_t k);

    void
    Scan(const SparseBatch& batch, const std::vector<double>& row_norms);

    [[nodiscard]] std::vector<std::vector<Neighbor>>
    MetricNeighbors() const;

    /// Queries with fewer than k rows sharing at least one term.
    [[nodiscard]] uint64_t
    QueriesWithFewOverlaps() const;

private:
    const SparseQuerySet& queries_;
    Metric metric_;
    uint64_t k_;
    std::vector<std::vector<std::pair<uint32_t, float>>> postings_;
    std::vector<uint32_t> term_ids_;
    std::vector<std::vector<TopK>> thread_heaps_;
    std::vector<uint64_t> overlap_counts_;
};

/// Levina-Bickel maximum-likelihood local intrinsic dimensionality from ascending neighbor
/// distances. Zero distances (duplicates) are skipped; returns NaN with fewer than two
/// positive distances or when all positive distances are equal.
double
lid_mle(const std::vector<double>& ascending_distances);

/// TwoNN estimator (Facco et al., 2017) from per-point ratios r2 / r1 > 1.
double
two_nn_dimension(const std::vector<double>& ratios);

/// Sparse dot product of two id-sorted rows.
double
sparse_dot(const uint32_t* ids_a,
           const float* values_a,
           uint64_t length_a,
           const uint32_t* ids_b,
           const float* values_b,
           uint64_t length_b);

}  // namespace vsag::dataset_profile
