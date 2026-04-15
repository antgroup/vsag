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

#include <random>

#include "impl/thread_pool/safe_thread_pool.h"
#include "typing.h"

namespace vsag {

class Allocator;

/**
 * @file kmeans_cluster.h
 * @brief K-means clustering implementation.
 */

/**
 * @brief Initialization method for K-means clustering.
 */
enum class KMeansInitMethod {
    /// Random initialization of centroids.
    RANDOM,
    /// K-means++ initialization for better convergence.
    KMEANS_PLUS_PLUS,
};

/**
 * @brief K-means clustering algorithm implementation.
 *
 * KMeansCluster provides K-means clustering functionality for vector
 * quantization and partitioning. It supports both random and K-means++
 * initialization methods.
 */
class KMeansCluster {
public:
    /**
     * @brief Constructs a KMeansCluster with the given dimension.
     *
     * @param dim Vector dimension for clustering.
     * @param allocator Allocator for memory management.
     * @param thread_pool Optional thread pool for parallel computation.
     */
    explicit KMeansCluster(int32_t dim,
                           Allocator* allocator,
                           SafeThreadPoolPtr thread_pool = nullptr);

    /**
     * @brief Destructor that releases centroid memory.
     */
    ~KMeansCluster();

    /**
     * @brief Runs K-means clustering on the given data.
     *
     * @param k Number of clusters.
     * @param datas Pointer to data vectors (dim * count floats).
     * @param count Number of data points.
     * @param iter Maximum number of iterations.
     * @param err Optional output for final clustering error.
     * @param use_mse_for_convergence Whether to use MSE for convergence check.
     * @param threshold Convergence threshold for MSE change.
     * @param init_method Initialization method for centroids.
     * @return Vector of cluster labels for each data point.
     */
    Vector<int>
    Run(uint32_t k,
        const float* datas,
        uint64_t count,
        int iter = 25,
        double* err = nullptr,
        bool use_mse_for_convergence = false,
        float threshold = 1e-6F,
        KMeansInitMethod init_method = KMeansInitMethod::KMEANS_PLUS_PLUS);

public:
    /// Pointer to centroid vectors (k * dim floats).
    float* k_centroids_{nullptr};

private:
    /**
     * @brief Finds nearest centroids using BLAS optimization.
     *
     * @param query Query vectors (dim * query_count floats).
     * @param query_count Number of query vectors.
     * @param k Number of centroids.
     * @param y_sqr Pre-computed squared norms of centroids.
     * @param distances Output distances.
     * @param labels Output cluster labels.
     * @return Total assignment distance.
     */
    double
    find_nearest_one_with_blas(const float* query,
                               const uint64_t query_count,
                               const uint64_t k,
                               float* y_sqr,
                               float* distances,
                               Vector<int32_t>& labels);

    /**
     * @brief Finds nearest centroids using HGraph index.
     *
     * @param query Query vectors (dim * query_count floats).
     * @param query_count Number of query vectors.
     * @param k Number of centroids.
     * @param labels Output cluster labels.
     * @return Total assignment distance.
     */
    double
    find_nearest_one_with_hgraph(const float* query,
                                 const uint64_t query_count,
                                 const uint64_t k,
                                 Vector<int32_t>& labels);

    /**
     * @brief Selects initial centroids randomly.
     *
     * @param datas Data vectors (dim * count floats).
     * @param count Number of data points.
     * @param k Number of centroids to select.
     * @param gen Random number generator.
     */
    void
    select_initial_centroids_random(const float* datas,
                                    uint64_t count,
                                    uint32_t k,
                                    std::mt19937& gen);

    /**
     * @brief Selects initial centroids using K-means++ algorithm.
     *
     * @param datas Data vectors (dim * count floats).
     * @param count Number of data points.
     * @param k Number of centroids to select.
     * @param gen Random number generator.
     */
    void
    select_initial_centroids_kmeans_plus_plus(const float* datas,
                                              uint64_t count,
                                              uint32_t k,
                                              std::mt19937& gen);

private:
    /// Allocator for memory management.
    Allocator* const allocator_{nullptr};

    /// Thread pool for parallel computation.
    SafeThreadPoolPtr thread_pool_{nullptr};

    /// Vector dimension.
    const int32_t dim_{0};

    /// Threshold for switching to HGraph-based assignment.
    static constexpr uint64_t THRESHOLD_FOR_HGRAPH = 10000ULL;

    /// Batch size for query processing.
    static constexpr uint64_t QUERY_BS = 65536ULL;
};

}  // namespace vsag