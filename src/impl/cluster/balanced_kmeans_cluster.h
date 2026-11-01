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

#include <optional>
#include <random>
#include <vector>

#include "impl/thread_pool/safe_thread_pool.h"
#include "kmeans_cluster.h"
#include "typing.h"

#ifdef VSAG_BALANCED_KMEANS_TEST_ACCESS
class BalancedKMeansClusterTestAccess;
#endif

namespace vsag {
class Allocator;

/// How BalancedKMeansCluster::Assign scores a vector against a centroid.
///
/// Every metric becomes a cost to minimise. Two costs that differ only by a
/// per-vector constant select the same assignment, so the constant terms below
/// are there only to make the reported cost read as a distance.
enum class BalancedAssignMetric {
    /// ||x - c||^2. Training always uses it, as KMeansCluster does.
    L2SQR,
    /// 1 - <x, c>: the routing key of an IVF index with metric "ip".
    INNER_PRODUCT,
    /// 1 - <x / ||x||, c>: the routing key of an IVF index with metric "cosine". The
    /// centroids are expected to have unit length.
    COSINE,
};

/// Settings of BalancedKMeansCluster::Run. The defaults suit IVF training.
struct BalancedKMeansOptions {
    /// Lloyd iterations, each a balanced assignment followed by a mean update.
    int32_t iterations{25};

    /// Clusters remembered per vector between two scorings against every centroid.
    /// The working set grows with `count * candidate_count`.
    uint32_t candidate_count{16};

    /// RANDOM seeds with k distinct input vectors. Equal-size clusters place their
    /// centroids with the density of the data, which is how uniform sampling places
    /// them too; KMEANS_PLUS_PLUS spreads them out instead.
    KMeansInitMethod init_method{KMeansInitMethod::RANDOM};

    /// Seed of every random choice; unset draws one from std::random_device.
    std::optional<uint32_t> random_seed{std::nullopt};

    /// Stop once an iteration lowers the loss by less than this fraction of it.
    /// 0 runs every iteration unless the assignment stops changing first.
    double tolerance{0.0};

    /// Tolerance of the final auction phase as a fraction of the typical cost gap
    /// between a vector's closest clusters. A completed assignment is optimal for its
    /// centroids within `count` times that tolerance.
    double epsilon_ratio{1e-4};

    /// Scales the work one assignment may spend (bids and re-scored vectors, counted,
    /// so the outcome does not depend on threads or timing). Exceeding it in the first
    /// iteration switches to the two-level decomposition; in a later iteration the
    /// result is the better of the iterations so far and the two-level one. Infinity
    /// never switches; 0 always takes the two-level path.
    double work_budget{1.0};
};

/// What the last Run or Assign did, for tests, benchmarks and diagnostics.
struct BalancedKMeansStats {
    /// Lloyd iterations whose result was kept; 0 for Assign.
    int32_t iterations{0};
    /// Mean cost per vector after each kept iteration (the loss), or the Assign cost.
    std::vector<double> loss_history;
    /// Bids placed by the auctions.
    uint64_t bids{0};
    /// Vectors scored again against every centroid because every cluster they
    /// remembered had become dearer than the ones they did not.
    uint64_t rescored_vectors{0};
    /// True when the result comes from the two-level decomposition.
    bool two_level{false};
    /// True when an assignment after the first exceeded the work budget. The result
    /// is then the better of the flat iterations so far and the two-level one.
    bool stopped_on_budget{false};
    /// Vectors that Assign placed at their cheapest open cluster after its auction
    /// exceeded the work budget. The sizes stay exact either way.
    uint64_t fallback_assignments{0};
    /// Bytes of the per-vector working set (candidates, bounds, auction state),
    /// excluding the input vectors and the centroids. It never holds a count x k
    /// matrix: distance tiles are bounded per task.
    uint64_t working_set_bytes{0};
};

/// Balanced k-means: Lloyd iterations whose assignment step respects exact capacities.
///
/// With `n` vectors and `k` clusters, `n % k` clusters receive `ceil(n / k)` vectors
/// and the others `floor(n / k)`, so no cluster is empty and no two differ by more
/// than one vector.
///
/// The assignment step is a transportation problem. It is solved by an auction with
/// one price per cluster: a vector bids for the cluster with the lowest
/// `cost + price`, a full cluster gives its seat to the higher bid and evicts its
/// lowest bidder, and prices only rise. With epsilon-scaling, the result is optimal
/// for the current centroids within `n * epsilon`; since the update step takes the
/// mean of every cluster, the loss does not increase between iterations by more
/// than that tolerance.
///
/// Memory stays O(n * m + k * dim): each vector keeps its `m` cheapest clusters and
/// a lower bound for all the others, and distances are computed tile by tile. A
/// vector whose remembered clusters all become dearer than that bound is scored
/// again against every centroid, so the bound never has to be trusted beyond what
/// it proves.
///
/// Data in which large groups of vectors are almost equally close to many centroids
/// (dense, high-dimensional blobs that need many clusters each) can make the auction
/// expensive. The work of every assignment is therefore bounded: past the budget the
/// first iteration hands over to a two-level decomposition, about sqrt(k) super-clusters
/// split into leaves with the same exact capacities, and a later iteration keeps the
/// previous result.
class BalancedKMeansCluster {
public:
    explicit BalancedKMeansCluster(int32_t dim,
                                   Allocator* allocator,
                                   SafeThreadPoolPtr thread_pool = nullptr);

    ~BalancedKMeansCluster();

    BalancedKMeansCluster(const BalancedKMeansCluster&) = delete;
    BalancedKMeansCluster&
    operator=(const BalancedKMeansCluster&) = delete;
    BalancedKMeansCluster(BalancedKMeansCluster&&) = delete;
    BalancedKMeansCluster&
    operator=(BalancedKMeansCluster&&) = delete;

    /// Clusters `count` row-major vectors (`dim` floats each) into `k` balanced
    /// clusters under squared L2 and returns the cluster of every vector. The
    /// centroids are left in `k_centroids_` (k x dim, row-major), the mean squared
    /// distance of every vector to its centroid in `*err`, and the run's
    /// statistics in `stats_`.
    ///
    /// Throws INVALID_ARGUMENT for k == 0, count == 0, k > count, null or
    /// non-finite data, squared norms above FLT_MAX / 4 (distances would overflow),
    /// a non-positive dimension, and invalid options.
    Vector<int32_t>
    Run(uint32_t k,
        const float* datas,
        uint64_t count,
        const BalancedKMeansOptions& options = {},
        double* err = nullptr);

    /// Assigns `count` vectors to `k` fixed centroids with the same capacities and
    /// auction as Run, without moving the centroids. `k` may exceed `count`: every
    /// vector then gets a cluster of its own and the others stay empty, which is
    /// still floor(count / k) or ceil(count / k). `*cost` receives the mean cost of
    /// the assignment under `metric`. Past `work_budget` (see BalancedKMeansOptions)
    /// the vectors still waiting go to their cheapest cluster with a free seat,
    /// which keeps the sizes exact.
    ///
    /// Throws INVALID_ARGUMENT for k == 0, count == 0, null or non-finite input,
    /// squared norms above FLT_MAX / 4 (distances would overflow), a non-positive
    /// dimension, candidate_count == 0, a negative or NaN work_budget, and an
    /// unknown metric.
    Vector<int32_t>
    Assign(const float* centroids,
           uint32_t k,
           const float* datas,
           uint64_t count,
           BalancedAssignMetric metric = BalancedAssignMetric::L2SQR,
           uint32_t candidate_count = 16,
           double* cost = nullptr,
           double work_budget = 1.0);

public:
    float* k_centroids_{nullptr};

    BalancedKMeansStats stats_;

private:
#ifdef VSAG_BALANCED_KMEANS_TEST_ACCESS
    friend class ::BalancedKMeansClusterTestAccess;
#endif

    struct InlineExecution {};

    /// A cluster for one part of a two-level run: it works on the calling thread.
    BalancedKMeansCluster(int32_t dim, Allocator* allocator, InlineExecution /*unused*/);

    void
    allocate_centroids(uint32_t k);

    enum class FlatOutcome {
        COMPLETED,
        /// The first assignment exceeded the work budget; nothing was kept.
        FIRST_OVER_BUDGET,
        /// A later assignment did; the previous iteration's result is in place.
        LATER_OVER_BUDGET,
    };

    /// Flat Lloyd iterations. `capacity`, when given, replaces the floor/ceil sizes.
    FlatOutcome
    run_flat(uint32_t k,
             const float* datas,
             uint64_t count,
             const BalancedKMeansOptions& options,
             const Vector<uint32_t>* capacity,
             Vector<int32_t>& labels);

    /// About sqrt(k) super-clusters whose capacities are whole numbers of leaves,
    /// then every super-cluster split into its leaves, in parallel.
    void
    run_two_level(uint32_t k,
                  const float* datas,
                  uint64_t count,
                  const BalancedKMeansOptions& options,
                  Vector<int32_t>& labels);

    void
    seed_random(const float* datas, uint64_t count, uint32_t k, std::mt19937& gen);

    void
    seed_kmeans_plus_plus(const float* datas, uint64_t count, uint32_t k, std::mt19937& gen);

    void
    update_centroids(const float* datas, uint64_t count, uint32_t k, const Vector<int32_t>& labels);

    double
    mean_squared_distance(const float* datas, uint64_t count, const Vector<int32_t>& labels) const;

private:
    Allocator* const allocator_{nullptr};

    SafeThreadPoolPtr thread_pool_{nullptr};

    const int32_t dim_{0};

    /// Multiple of the work budget that the first assignment, which starts without
    /// useful prices, may spend.
    double first_assignment_budget_factor_{1.0};
};

}  // namespace vsag
