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

#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "vsag/bitset.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"

namespace vsag {

/**
 * @brief Performs k-means clustering on the given data.
 *
 * This function applies the k-means clustering algorithm to partition the input
 * data into `k` clusters. The algorithm iteratively assigns data points to clusters
 * and updates the cluster centroids.
 *
 * @param d The dimensionality of the data points.
 * @param n The number of data points.
 * @param k The number of clusters.
 * @param x Pointer to the input data array of size `n * d`.
 * @param centroids Pre-allocated output array to store the centroid of each cluster, of size `k * d`.
 * @param dis_type The type of distance metric to use, specified as a string ("l2", "cosine", "ip").
 * @return float The final quantization error (sum of squared distances to the nearest centroid).
 */
float
kmeans_clustering(uint64_t d,
                  uint64_t n,
                  uint64_t k,
                  const float* x,
                  float* centroids,
                  const std::string& dis_type);

/**
 * @brief Filters a set of vectors based on a distance threshold.
 *
 * This function calculates the L2 distance between entries in the base and query sets and
 * filters out those pairs for which the distance exceeds a given threshold.
 *
 * @param dim The dimensionality of the vectors.
 * @param nb The number of base vectors.
 * @param base Pointer to the array containing base vectors of size `nb * dim`.
 * @param query Pointer to the query vector of size `dim`.
 * @param threshold The distance threshold for filtering.
 * @return BitsetPtr A pointer to a bitset indicating which base vectors have not been filtered out.
 */
BitsetPtr
l2_and_filtering(int64_t dim, int64_t nb, const float* base, const float* query, float threshold);

/**
 * @brief Computes the recall of k-nearest neighbors (k-NN) search results.
 *
 * This function computes how many of the retrieved nearest neighbors in the result set
 * match the true nearest neighbors from the given ground truth mapping.
 *
 * @param base Pointer to the array containing base vectors.
 * @param id_map Pointer to the ground truth id mapping.
 * @param base_num The number of base vectors.
 * @param query Pointer to the query vector.
 * @param data_dim The dimensionality of the vectors.
 * @param result_ids Pointer to the result id array from the k-NN search.
 * @param result_size The size of the result set, equal to K.
 * @return float The recall value of the k-NN search results.
 */
float
knn_search_recall(const float* base,
                  const int64_t* id_map,
                  int64_t base_num,
                  const float* query,
                  int64_t data_dim,
                  const int64_t* result_ids,
                  int64_t result_size);

/**
 * @brief Computes the recall of a range search based on a given threshold.
 *
 * This function assesses how many of the base vectors are correctly included or excluded
 * from the result set by comparing to a given set of result ids.
 *
 * @param base Pointer to the array containing base vectors.
 * @param base_ids Pointer to the base vector ids.
 * @param num_base The number of base vectors.
 * @param query Pointer to the query vector.
 * @param dim The dimensionality of the vectors.
 * @param result_ids Pointer to the result id array from the range search.
 * @param result_size The size of the result set.
 * @param threshold The distance threshold used in the range search.
 * @return float The recall value of the range search results.
 */
float
range_search_recall(const float* base,
                    const int64_t* base_ids,
                    int64_t num_base,
                    const float* query,
                    int64_t dim,
                    const int64_t* result_ids,
                    int64_t result_size,
                    float threshold);

/**
 * @brief Configuration for random projection tree (RPT) partitioning.
 */
struct RPTPartitionParams {
    /// Target bucket size L: every partition holds at most L vectors.
    uint64_t bucket_size{1000};

    /// Upper bound on the tree depth. When reached, the remaining vectors are split
    /// into equal-sized chunks by id order so that the size guarantees still hold.
    uint64_t max_depth{64};

    /// Random seed. The same input and seed always produce the same partitions.
    uint64_t seed{0};
};

/**
 * @brief Partitions vectors into near-uniform buckets using a random projection tree.
 *
 * The vectors are recursively split along random projection directions. Given `n`
 * vectors and a target bucket size `L = params.bucket_size`, the result satisfies:
 *   - the number of partitions equals `ceil(n / L)`;
 *   - every partition holds at most `L` vectors;
 *   - when `n >= L`, every partition holds at least `floor(L / 2)` vectors, and the
 *     ratio between the largest and the smallest partition never exceeds 2.
 *
 * @param d The dimensionality of the vectors, must be positive.
 * @param n The number of vectors. `n == 0` yields an empty result.
 * @param x Pointer to the input array of size `n * d`. It is owned by the caller and is
 *          neither modified nor retained after the call returns.
 * @param params Partitioning configuration, see RPTPartitionParams.
 * @return The vector ids (indices into `x`) of each partition, or an error when the
 *         arguments are invalid (`d == 0`, `bucket_size == 0`, or `x == nullptr` with
 *         `n > 0`).
 */
tl::expected<std::vector<std::vector<int64_t>>, Error>
rpt_partition(uint64_t d, uint64_t n, const float* x, const RPTPartitionParams& params);

}  // namespace vsag
