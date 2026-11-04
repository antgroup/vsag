
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

#include "balanced_kmeans_cluster.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <future>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <utility>

#include "impl/logger/logger.h"
#include "simd/fp32_simd.h"
#include "simd/normalize.h"
#include "utils/util_functions.h"
#include "vsag_exception.h"

namespace vsag {

namespace {

/// The largest finite float. Distances are clamped into [-FLT_MAX, FLT_MAX]
/// before they are stored or subtracted: a vector pair whose squared L2 or
/// dot product overflows to infinity would otherwise make two infinite
/// distances subtract to NaN, and a NaN in the ordering comparator of the
/// assignment would leave the sort order undefined. Inputs are checked to be
/// finite, so reaching a clamp means the data magnitudes themselves overflow,
/// which is a pathological but not invalid input.
constexpr float K_MAX_DISTANCE = std::numeric_limits<float>::max();

float
clamp_distance(float distance) {
    if (distance != distance) {
        return K_MAX_DISTANCE;
    }
    return std::min(std::max(distance, -K_MAX_DISTANCE), K_MAX_DISTANCE);
}

/// The distance a balanced assignment orders by. Squared L2 for L2SQR, and
/// the IVF scan routing key `1 - <x, c>` for IP and cosine, matching the
/// constant term the scan layout uses (norms_[b] = 1.0). Cosine centroids are
/// normalized by the IVF training path before assignment, so this key orders
/// the same way as the true cosine distance.
float
metric_distance(MetricType metric, const float* x, const float* c, int32_t dim) {
    if (metric == MetricType::METRIC_TYPE_IP or metric == MetricType::METRIC_TYPE_COSINE) {
        return clamp_distance(1.0F - FP32ComputeIP(x, c, dim));
    }
    return clamp_distance(FP32ComputeL2Sqr(x, c, dim));
}

[[noreturn]] void
throw_invalid_argument(const std::string& message) {
    throw VsagException(ErrorType::INVALID_ARGUMENT, message);
}

}  // namespace

BalancedKMeansCluster::BalancedKMeansCluster(int32_t dim,
                                             Allocator* allocator,
                                             SafeThreadPoolPtr thread_pool)
    : dim_(dim), allocator_(allocator) {
    if (dim <= 0) {
        throw_invalid_argument("dim must be positive");
    }
    if (allocator_ == nullptr) {
        throw_invalid_argument("allocator cannot be null");
    }
    if (thread_pool == nullptr) {
        this->thread_pool_ = SafeThreadPool::FactoryDefaultThreadPool();
    } else {
        this->thread_pool_ = std::move(thread_pool);
    }
}

BalancedKMeansCluster::~BalancedKMeansCluster() {
    if (k_centroids_ != nullptr) {
        allocator_->Deallocate(k_centroids_);
        k_centroids_ = nullptr;
    }
}

void
BalancedKMeansCluster::check_arguments(const float* datas,
                                       uint64_t count,
                                       uint32_t k,
                                       bool allow_k_gt_count) const {
    if (k == 0) {
        throw_invalid_argument("k must be positive");
    }
    if (count == 0) {
        throw_invalid_argument("count must be positive");
    }
    if (datas == nullptr) {
        throw_invalid_argument("datas cannot be null");
    }
    if (not allow_k_gt_count and k > count) {
        throw_invalid_argument("k cannot be larger than count");
    }
    // The assignment math is only well-defined for finite input. Checking
    // every value costs one pass over the data, negligible next to the
    // distance computations of even a single iteration.
    const uint64_t total = count * static_cast<uint64_t>(dim_);
    for (uint64_t i = 0; i < total; ++i) {
        if (datas[i] != datas[i] or datas[i] > std::numeric_limits<float>::max() or
            datas[i] < -std::numeric_limits<float>::max()) {
            throw_invalid_argument("datas must contain only finite values");
        }
    }
}

Vector<int>
BalancedKMeansCluster::Run(uint32_t k,
                           const float* datas,
                           uint64_t count,
                           int iter,
                           double* loss,
                           float threshold,
                           KMeansInitMethod init_method,
                           std::optional<uint32_t> random_seed,
                           uint32_t num_candidates) {
    this->check_arguments(datas, count, k, /*allow_k_gt_count=*/false);
    if (iter <= 0) {
        throw_invalid_argument("iter must be positive");
    }
    if (num_candidates == 0) {
        throw_invalid_argument("num_candidates must be positive");
    }

    if (k_centroids_ != nullptr) {
        allocator_->Deallocate(k_centroids_);
        k_centroids_ = nullptr;
    }
    const uint64_t centroid_values = static_cast<uint64_t>(k) * static_cast<uint64_t>(dim_);
    k_centroids_ = static_cast<float*>(allocator_->Allocate(centroid_values * sizeof(float)));
    std::fill(k_centroids_, k_centroids_ + centroid_values, 0.0F);

    std::random_device rd;
    std::mt19937 gen(random_seed.has_value() ? *random_seed : rd());

    if (init_method == KMeansInitMethod::KMEANS_PLUS_PLUS) {
        select_initial_centroids_kmeans_plus_plus(datas, count, k, gen);
    } else {
        select_initial_centroids_random(datas, count, k, gen);
    }

    logger::trace("BalancedKMeansCluster::Run k: {}, count: {}, iter: {}", k, count, iter);

    Vector<int32_t> labels(count, -1, allocator_);
    double current_loss = 0.0;
    double last_loss = std::numeric_limits<double>::max();
    int32_t iterations_run = 0;

    for (int it = 0; it < iter; ++it) {
        ++iterations_run;
        // The capacity-constrained assignment step of one Lloyd iteration.
        // Training always orders by squared L2, the same objective the plain
        // k-means trainer optimizes regardless of the index metric.
        current_loss = this->balanced_assign(k_centroids_,
                                             datas,
                                             count,
                                             k,
                                             MetricType::METRIC_TYPE_L2SQR,
                                             num_candidates,
                                             labels.data());
        // Centroids become the mean of the vectors assigned to them. With
        // k <= count every cluster holds at least floor(count / k) >= 1
        // vectors by construction, so no centroid loses its support and no
        // empty-cluster re-seeding is needed.
        this->update_centroids(datas, labels.data(), count, k);

        logger::trace("[{}] BalancedKMeansCluster::Run iter: {}/{} finished, cur loss is {}",
                      get_current_time(),
                      it,
                      iter,
                      current_loss);
        if (it > 0 and std::fabs(last_loss - current_loss) / static_cast<double>(count) <
                           static_cast<double>(threshold)) {
            break;
        }
        last_loss = current_loss;
    }

    this->record_stats(k, count, labels.data(), current_loss, iterations_run);
    if (loss != nullptr) {
        *loss = current_loss;
    }
    return labels;
}

Vector<int>
BalancedKMeansCluster::Assign(uint32_t k,
                              const float* centroids,
                              const float* datas,
                              uint64_t count,
                              MetricType metric,
                              uint32_t num_candidates) {
    this->check_arguments(datas, count, k, /*allow_k_gt_count=*/true);
    if (centroids == nullptr) {
        throw_invalid_argument("centroids cannot be null");
    }
    if (num_candidates == 0) {
        throw_invalid_argument("num_candidates must be positive");
    }
    // The same finiteness bound as the data: a non-finite centroid would make
    // every distance to it meaningless.
    const uint64_t centroid_values = static_cast<uint64_t>(k) * static_cast<uint64_t>(dim_);
    for (uint64_t i = 0; i < centroid_values; ++i) {
        if (centroids[i] != centroids[i] or centroids[i] > std::numeric_limits<float>::max() or
            centroids[i] < -std::numeric_limits<float>::max()) {
            throw_invalid_argument("centroids must contain only finite values");
        }
    }

    Vector<float> normalized_datas(allocator_);
    if (metric == MetricType::METRIC_TYPE_COSINE) {
        normalized_datas.resize(count * static_cast<uint64_t>(dim_));
        for (uint64_t i = 0; i < count; ++i) {
            Normalize(datas + i * static_cast<uint64_t>(dim_),
                      normalized_datas.data() + i * static_cast<uint64_t>(dim_),
                      dim_);
        }
        datas = normalized_datas.data();
    }

    Vector<int32_t> labels(count, -1, allocator_);
    const double loss =
        this->balanced_assign(centroids, datas, count, k, metric, num_candidates, labels.data());
    this->record_stats(k, count, labels.data(), loss, /*iters=*/0);
    return labels;
}

double
BalancedKMeansCluster::balanced_assign(const float* centroids,
                                       const float* datas,
                                       uint64_t count,
                                       uint32_t k,
                                       MetricType metric,
                                       uint32_t num_candidates,
                                       int32_t* labels) {
    const uint32_t c = std::min(num_candidates, k);

    // Step 1: per-vector candidate lists. Distances are evaluated one vector
    // against all k centroids in fixed blocks, so peak memory never grows
    // beyond the per-vector lists: no n x k distance matrix is materialized
    // for any n and k. Per-vector work is independent, so the result does not
    // depend on the thread schedule.
    const uint64_t cand_total = count * static_cast<uint64_t>(c);
    Vector<float> cand_dist(cand_total, K_MAX_DISTANCE, allocator_);
    Vector<uint32_t> cand_id(cand_total, 0U, allocator_);
    Vector<uint64_t> demand(k, 0ULL, allocator_);
    std::mutex demand_mutex;

    constexpr uint64_t block_size = 1024;
    auto scan_block = [&](uint64_t begin, uint64_t end) {
        Vector<uint64_t> local_demand(k, 0ULL, allocator_);
        // The small top-c buffers are reused across the block's vectors; c is
        // runtime, so they are heap buffers rather than variable-length arrays.
        // Infinity is only an uninitialized sentinel. A real distance may
        // legitimately clamp to K_MAX_DISTANCE and must still be selectable.
        Vector<float> best_dist(c, std::numeric_limits<float>::infinity(), allocator_);
        Vector<uint32_t> best_id(c, 0U, allocator_);
        for (uint64_t i = begin; i < end; ++i) {
            const float* x = datas + i * static_cast<uint64_t>(dim_);
            std::fill(
                best_dist.begin(), best_dist.end(), std::numeric_limits<float>::infinity());
            std::fill(best_id.begin(), best_id.end(), 0U);
            for (uint32_t j = 0; j < k; ++j) {
                const float dist =
                    metric_distance(metric, x, centroids + j * static_cast<uint64_t>(dim_), dim_);
                if (dist < best_dist[c - 1]) {
                    uint32_t pos = c - 1;
                    while (pos > 0 and best_dist[pos - 1] > dist) {
                        best_dist[pos] = best_dist[pos - 1];
                        best_id[pos] = best_id[pos - 1];
                        --pos;
                    }
                    best_dist[pos] = dist;
                    best_id[pos] = j;
                }
            }
            for (uint32_t j = 0; j < c; ++j) {
                cand_dist[i * static_cast<uint64_t>(c) + j] = best_dist[j];
                cand_id[i * static_cast<uint64_t>(c) + j] = best_id[j];
            }
            ++local_demand[best_id[0]];
        }
        std::lock_guard<std::mutex> lock(demand_mutex);
        for (uint32_t j = 0; j < k; ++j) {
            // Exact integer sums are order-independent, so any merge
            // schedule produces the same demand vector.
            demand[j] += local_demand[j];
        }
    };

    std::vector<std::future<void>> futures;
    for (uint64_t i = 0; i < count; i += block_size) {
        const uint64_t begin = i;
        const uint64_t end = std::min(i + block_size, count);
        if (thread_pool_->IsCurrentThread()) {
            scan_block(begin, end);
        } else {
            futures.emplace_back(thread_pool_->GeneralEnqueue(scan_block, begin, end));
        }
    }
    std::exception_ptr scan_exception;
    for (auto& future : futures) {
        try {
            future.get();
        } catch (...) {
            if (scan_exception == nullptr) {
                scan_exception = std::current_exception();
            }
        }
    }
    // Every submitted block has finished before the captured stack state can
    // unwind, and no placeholder candidate data is consumed after a failure.
    if (scan_exception != nullptr) {
        std::rethrow_exception(scan_exception);
    }

    // Step 2: capacity targets. Exactly n mod k clusters hold ceil(n / k)
    // vectors. The seats go to the clusters that the unconstrained nearest
    // pass drew most, which keeps the constrained assignment closest to the
    // unconstrained one. Ties break by cluster id, so the targets are
    // deterministic.
    const uint64_t base = count / static_cast<uint64_t>(k);
    const uint64_t extra = count % static_cast<uint64_t>(k);
    Vector<uint64_t> remaining(k, base, allocator_);
    {
        Vector<uint32_t> cluster_order(k, 0U, allocator_);
        std::iota(cluster_order.begin(), cluster_order.end(), 0U);
        std::sort(cluster_order.begin(), cluster_order.end(), [&](uint32_t a, uint32_t b) {
            if (demand[a] != demand[b]) {
                return demand[a] > demand[b];
            }
            return a < b;
        });
        for (uint64_t i = 0; i < extra; ++i) {
            ++remaining[cluster_order[i]];
        }
    }

    // Step 3: regret-greedy placement. A vector's regret is the extra cost of
    // losing its best candidate; the vectors that lose the most when pushed
    // are placed first, while their best seats are still free. The order is
    // a total order (regret, then nearest distance, then vector id), so the
    // result does not depend on the thread count.
    Vector<uint64_t> point_order(count, 0ULL, allocator_);
    std::iota(point_order.begin(), point_order.end(), 0ULL);
    auto regret = [&](uint64_t i) {
        if (c >= 2) {
            return cand_dist[i * static_cast<uint64_t>(c) + 1] -
                   cand_dist[i * static_cast<uint64_t>(c)];
        }
        return 0.0F;
    };
    std::sort(point_order.begin(), point_order.end(), [&](uint64_t a, uint64_t b) {
        const float ra = regret(a);
        const float rb = regret(b);
        if (ra != rb) {
            return ra > rb;
        }
        const float da = cand_dist[a * static_cast<uint64_t>(c)];
        const float db = cand_dist[b * static_cast<uint64_t>(c)];
        if (da != db) {
            return da < db;
        }
        return a < b;
    });

    // Total remaining capacity equals the number of unplaced vectors at every
    // point of this loop, so the on-demand rescan always finds a cluster with
    // a free seat.
    for (uint64_t i : point_order) {
        const uint64_t row = i * static_cast<uint64_t>(c);
        uint32_t chosen = k;
        for (uint32_t j = 0; j < c; ++j) {
            const uint32_t id = cand_id[row + j];
            if (remaining[id] > 0) {
                chosen = id;
                break;
            }
        }
        if (chosen == k) {
            // Every remembered candidate is full: recompute distances against
            // the clusters that still have seats instead of keeping the full
            // row. Rare for c > 1, and bounded work even when it repeats.
            const float* x = datas + i * static_cast<uint64_t>(dim_);
            float best = std::numeric_limits<float>::infinity();
            for (uint32_t j = 0; j < k; ++j) {
                if (remaining[j] == 0) {
                    continue;
                }
                const float dist =
                    metric_distance(metric, x, centroids + j * static_cast<uint64_t>(dim_), dim_);
                if (dist < best) {
                    best = dist;
                    chosen = j;
                }
            }
            // Unreachable: some cluster always holds a free seat while vectors
            // remain, and the loop above visits every such cluster.
            if (chosen == k) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "balanced assignment ran out of capacity");
            }
        }
        labels[i] = static_cast<int32_t>(chosen);
        --remaining[chosen];
    }

    // The loss is recomputed from the final (vector, centroid) pairs with the
    // same distance as the assignment, in a fixed order.
    double total_loss = 0.0;
    for (uint64_t i = 0; i < count; ++i) {
        const float* x = datas + i * static_cast<uint64_t>(dim_);
        const float* centroid =
            centroids + static_cast<uint64_t>(labels[i]) * static_cast<uint64_t>(dim_);
        total_loss += static_cast<double>(metric_distance(metric, x, centroid, dim_));
    }
    return total_loss;
}

void
BalancedKMeansCluster::update_centroids(const float* datas,
                                        const int32_t* labels,
                                        uint64_t count,
                                        uint32_t k) {
    const uint64_t centroid_values = static_cast<uint64_t>(k) * static_cast<uint64_t>(dim_);
    Vector<float> sums(centroid_values, 0.0F, allocator_);
    Vector<uint64_t> sizes(k, 0ULL, allocator_);
    // Float addition is order-dependent, so the accumulation is one sequential
    // pass in fixed vector order: the centroids are bit-identical regardless
    // of how the assignment steps were scheduled.
    for (uint64_t i = 0; i < count; ++i) {
        const uint32_t label = static_cast<uint32_t>(labels[i]);
        const float* src = datas + i * static_cast<uint64_t>(dim_);
        float* dest = sums.data() + static_cast<uint64_t>(label) * static_cast<uint64_t>(dim_);
        ++sizes[label];
        for (int32_t d = 0; d < dim_; ++d) {
            dest[d] += src[d];
        }
    }
    for (uint32_t j = 0; j < k; ++j) {
        if (sizes[j] == 0) {
            // Cannot happen for Run with k <= count (every cluster keeps at
            // least floor(count / k) >= 1 vectors by construction) nor for the
            // clusters a count < k Assign fills; a cluster that lost all its
            // vectors keeps its previous centroid.
            continue;
        }
        const float scale = 1.0F / static_cast<float>(sizes[j]);
        float* centroid = k_centroids_ + static_cast<uint64_t>(j) * static_cast<uint64_t>(dim_);
        const float* sum = sums.data() + static_cast<uint64_t>(j) * static_cast<uint64_t>(dim_);
        for (int32_t d = 0; d < dim_; ++d) {
            centroid[d] = sum[d] * scale;
        }
    }
}

void
BalancedKMeansCluster::select_initial_centroids_random(const float* datas,
                                                       uint64_t count,
                                                       uint32_t k,
                                                       std::mt19937& gen) {
    std::uniform_int_distribution<uint64_t> dis(0, count - 1);
    for (uint32_t i = 0; i < k; ++i) {
        const uint64_t index = dis(gen);
        for (int32_t d = 0; d < dim_; ++d) {
            k_centroids_[static_cast<uint64_t>(i) * static_cast<uint64_t>(dim_) +
                         static_cast<uint64_t>(d)] =
                datas[static_cast<uint64_t>(index) * static_cast<uint64_t>(dim_) +
                      static_cast<uint64_t>(d)];
        }
    }
}

void
BalancedKMeansCluster::select_initial_centroids_kmeans_plus_plus(const float* datas,
                                                                 uint64_t count,
                                                                 uint32_t k,
                                                                 std::mt19937& gen) {
    // Standard D^2 seeding, identical in spirit to the plain trainer's, so the
    // two trainers can be compared on equal initialization footing.
    std::uniform_int_distribution<uint64_t> first_dis(0, count - 1);
    const uint64_t first_idx = first_dis(gen);
    for (int32_t d = 0; d < dim_; ++d) {
        k_centroids_[static_cast<uint64_t>(d)] =
            datas[static_cast<uint64_t>(first_idx) * static_cast<uint64_t>(dim_) +
                  static_cast<uint64_t>(d)];
    }

    Vector<float> min_distances(count, K_MAX_DISTANCE, allocator_);
    for (uint32_t ci = 1; ci < k; ++ci) {
        const float* centroid =
            k_centroids_ + static_cast<uint64_t>(ci - 1) * static_cast<uint64_t>(dim_);
        for (uint64_t i = 0; i < count; ++i) {
            const float dist =
                FP32ComputeL2Sqr(datas + i * static_cast<uint64_t>(dim_), centroid, dim_);
            min_distances[i] = std::min(min_distances[i], clamp_distance(dist));
        }
        double total_weight = 0.0;
        for (uint64_t i = 0; i < count; ++i) {
            total_weight += static_cast<double>(min_distances[i]);
        }
        uint64_t selected_idx = 0;
        if (total_weight <= 0.0) {
            // Every vector coincides with an already chosen centroid; any
            // vector is then a valid pick, so keep the generator moving and
            // take a uniform one.
            std::uniform_int_distribution<uint64_t> dis(0, count - 1);
            selected_idx = dis(gen);
        } else {
            std::uniform_real_distribution<double> prob_dis(0.0, total_weight);
            const double target = prob_dis(gen);
            double cumulative = 0.0;
            selected_idx = count - 1;
            for (uint64_t i = 0; i < count; ++i) {
                cumulative += static_cast<double>(min_distances[i]);
                if (cumulative >= target) {
                    selected_idx = i;
                    break;
                }
            }
        }
        for (int32_t d = 0; d < dim_; ++d) {
            k_centroids_[static_cast<uint64_t>(ci) * static_cast<uint64_t>(dim_) +
                         static_cast<uint64_t>(d)] =
                datas[static_cast<uint64_t>(selected_idx) * static_cast<uint64_t>(dim_) +
                      static_cast<uint64_t>(d)];
        }
    }
}

void
BalancedKMeansCluster::record_stats(
    uint32_t k, uint64_t count, const int32_t* labels, double loss, int32_t iters) {
    Vector<uint64_t> sizes(k, 0ULL, allocator_);
    for (uint64_t i = 0; i < count; ++i) {
        ++sizes[static_cast<uint32_t>(labels[i])];
    }
    const auto extremes = std::minmax_element(sizes.begin(), sizes.end());
    stats_.cluster_count = k;
    stats_.vector_count = count;
    stats_.min_cluster_size = *extremes.first;
    stats_.max_cluster_size = *extremes.second;
    stats_.loss = loss;
    stats_.iterations_run = iters;
}

}  // namespace vsag
