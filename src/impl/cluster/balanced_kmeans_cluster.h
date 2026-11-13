
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
#include <optional>

#include "impl/thread_pool/safe_thread_pool.h"
#include "kmeans_cluster.h"
#include "metric_type.h"
#include "typing.h"

namespace vsag {
class Allocator;

/// Summary of the most recent clustering or assignment done by
/// `BalancedKMeansCluster`, so that tests, tools and internal callers can
/// evaluate balance and loss without recomputing them from the returned
/// labels (issue #2870 observability requirement).
struct BalancedKMeansStats {
    /// The number of clusters the last call was asked for.
    uint32_t cluster_count{0};
    /// The number of vectors the last call assigned.
    uint64_t vector_count{0};
    /// Smallest and largest final cluster size, derived from the returned
    /// labels rather than from the capacity targets.
    uint64_t min_cluster_size{0};
    uint64_t max_cluster_size{0};
    /// Total assignment cost of the last assignment step: the sum over all
    /// vectors of the distance to the centroid they were assigned to, using
    /// the same distance as the assignment itself.
    double loss{0.0};
    /// Number of assignment/centroid-update iterations `Run` executed; 0 for
    /// a single `Assign` call, which does not iterate.
    int32_t iterations_run{0};
};

/// Balanced k-means clustering for IVF partitioning (issue #2870).
///
/// Standard k-means (`KMeansCluster`) assigns every vector to its nearest
/// centroid, which can produce heavily skewed or even empty clusters on
/// skewed or duplicated data. This trainer keeps the same squared-L2
/// objective and mean centroid update, but every assignment step is
/// capacity-constrained: with `n` vectors and `k` clusters each cluster
/// finally holds either `floor(n / k)` or `ceil(n / k)` vectors, so the
/// difference between the largest and the smallest cluster is at most one
/// and no cluster is empty when `1 <= k <= n`.
///
/// The balancing strategy is a regret-greedy capacitated assignment, which
/// is a different point in the design space from a price-based auction or
/// an exact transportation simplex:
///  1. Each vector keeps only its `num_candidates` nearest clusters
///     (bounded per-vector state; no `n x k` distance matrix is ever
///     materialized; distances are computed one vector at a time in fixed
///     parallel blocks and the reduced statistics are order-independent).
///  2. The `n mod k` clusters with the highest unconstrained nearest-centroid
///     demand receive `ceil(n / k)` seats, the others `floor(n / k)`.
///  3. Vectors are placed one at a time, most-constrained first, in decreasing
///     order of their *regret* (the cost gap between their best and second
///     best candidate). Each vector takes the nearest cluster on its candidate
///     list that still has a free seat; if every candidate is full, the
///     distances to all clusters with a free seat are recomputed on demand.
///     Because the capacities sum to exactly `n`, a seat is always available
///     and every vector is placed.
///  4. Centroids are re-set to the mean of their assigned vectors and the
///     cycle repeats until the loss stops improving or the iteration budget
///     is exhausted.
///
/// Expected complexity per iteration: `O(n * k * dim)` distance evaluations
/// plus `O(n log n)` ordering - the same asymptotic work as one standard
/// k-means iteration plus a sort - and `O(n * num_candidates + k * dim)`
/// working memory. The assignment is a heuristic for the capacity-constrained
/// step (it does not price seats or search augmenting swaps), so it can pay
/// a somewhat larger loss than a solver-exact assignment on strongly skewed
/// data; the balance guarantee itself holds for every input.
///
/// Results are deterministic for a given input, iteration count, candidate
/// count and seed, independent of the thread pool size: no shared accumulation
/// happens in parallel code and the assignment order is a total order.
///
/// Like `KMeansCluster`, training always minimizes squared L2 regardless of
/// the index metric; cosine index handling (centroid normalization) stays in
/// the IVF layer above. `Assign` additionally supports the IP and cosine
/// routing keys so that a balanced bucket assignment can mirror the ordering
/// IVF search uses. This trainer always runs on the host; the CUDA build
/// backend only accelerates `KMeansCluster`.
class BalancedKMeansCluster {
public:
    /// Per-vector candidates kept during assignment. Larger values reduce the
    /// number of on-demand full rescans (better loss) at a linear memory cost
    /// of `8 * num_candidates` bytes per vector. Values above the cluster
    /// count are clamped, and `k <= num_candidates` makes every assignment
    /// exact per-iteration since no vector then needs a rescan.
    static constexpr uint32_t DEFAULT_NUM_CANDIDATES = 8;

    explicit BalancedKMeansCluster(int32_t dim,
                                   Allocator* allocator,
                                   SafeThreadPoolPtr thread_pool = nullptr);

    ~BalancedKMeansCluster();

    /// Runs balanced k-means and returns the final vector-to-cluster
    /// assignment (`labels[i]` is the cluster of vector `i`), which also
    /// serves as the observability handle on the balanced partitioning. The
    /// trained centroids are left in `k_centroids_` and summary statistics in
    /// `Stats()`.
    ///
    /// Throws `VsagException(INVALID_ARGUMENT)` when `k == 0`, `count == 0`,
    /// `datas == nullptr`, `k > count`, `iter <= 0`, `num_candidates == 0`,
    /// or `datas` contains a non-finite value.
    Vector<int>
    Run(uint32_t k,
        const float* datas,
        uint64_t count,
        int iter = 25,
        double* loss = nullptr,
        float threshold = 1e-6F,
        KMeansInitMethod init_method = KMeansInitMethod::KMEANS_PLUS_PLUS,
        std::optional<uint32_t> random_seed = std::nullopt,
        uint32_t num_candidates = DEFAULT_NUM_CANDIDATES);

    /// Assigns `count` vectors to fixed centroids under the balanced capacity
    /// constraint, without updating the centroids. `metric` selects the
    /// ordering key: squared L2 for `METRIC_TYPE_L2SQR`, `1 - <x, c>` for IP
    /// and cosine, matching the IVF scan routing key. Cosine input rows are
    /// normalized internally before their assignment costs are compared.
    /// Unlike `Run`, this
    /// accepts `k > count`: then `count` clusters receive one vector each and
    /// the remaining clusters stay empty, which is the best achievable
    /// balance when there are fewer vectors than clusters.
    ///
    /// Throws `VsagException(INVALID_ARGUMENT)` on the same invalid inputs as
    /// `Run`, except that `k > count` is allowed.
    Vector<int>
    Assign(uint32_t k,
           const float* centroids,
           const float* datas,
           uint64_t count,
           MetricType metric = MetricType::METRIC_TYPE_L2SQR,
           uint32_t num_candidates = DEFAULT_NUM_CANDIDATES);

    /// Statistics of the most recent `Run` or `Assign` call.
    const BalancedKMeansStats&
    Stats() const {
        return stats_;
    }

public:
    /// Trained centroids from the most recent `Run`, `k * dim` floats
    /// (cluster-major). Unset after `Assign`, which does not train.
    float* k_centroids_{nullptr};

private:
    /// Validates the shared arguments of `Run` and `Assign`, including the
    /// finiteness of the input data.
    void
    check_arguments(const float* datas, uint64_t count, uint32_t k, bool allow_k_gt_count) const;

    /// Places `count` vectors over `centroids` with exact capacity balance and
    /// returns the total assignment cost; fills `labels`.
    double
    balanced_assign(const float* centroids,
                    const float* datas,
                    uint64_t count,
                    uint32_t k,
                    MetricType metric,
                    uint32_t num_candidates,
                    int32_t* labels);

    /// Recomputes every centroid as the mean of the vectors assigned to it.
    /// Clusters with no member keep their previous centroid.
    void
    update_centroids(const float* datas, const int32_t* labels, uint64_t count, uint32_t k);

    void
    select_initial_centroids_random(const float* datas,
                                    uint64_t count,
                                    uint32_t k,
                                    std::mt19937& gen);

    void
    select_initial_centroids_kmeans_plus_plus(const float* datas,
                                              uint64_t count,
                                              uint32_t k,
                                              std::mt19937& gen);

    /// Fills `stats_` from the final labels of a completed assignment.
    void
    record_stats(uint32_t k, uint64_t count, const int32_t* labels, double loss, int32_t iters);

private:
    Allocator* const allocator_{nullptr};

    SafeThreadPoolPtr thread_pool_{nullptr};

    const int32_t dim_{0};

    BalancedKMeansStats stats_{};
};

}  // namespace vsag
