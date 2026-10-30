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

#include "typing.h"
#include "vsag/utils.h"

namespace vsag {
class Allocator;
struct RPTPartitionParams;

/**
 * @brief Random projection tree (RPT) partitioner.
 *
 * Splits a vector set into partitions whose sizes are bounded by the target bucket size.
 * Every node draws a random projection direction, is assigned the number of leaves
 * `k = ceil(n / L)` it must produce, and is cut at the projection rank that hands
 * `ceil(k / 2)` leaves to the left child. The cut is therefore generally not the projection
 * median, and no additional random threshold offset is used. Every leaf ends up with either
 * `floor(N / K)` or `ceil(N / K)` vectors, where `K = ceil(N / L)` is the total number of
 * partitions.
 *
 * The permutation of ids is partitioned in place and no vector data is copied. The input
 * pointer is owned by the caller and is not retained after `Run` returns.
 */
class RPTPartition {
public:
    explicit RPTPartition(uint64_t dim, Allocator* allocator);

    /**
     * @brief Partitions `count` vectors of `dim` dimensions.
     *
     * @param datas Pointer to at least `count * dim` contiguous float values.
     * @param count Number of vectors to partition.
     * @param dim Number of dimensions in each vector; must match the constructor argument.
     * @param params Partitioning parameters.
     *
     * @throws VsagException(INVALID_ARGUMENT) when `dim == 0`, `dim != dim_`,
     *         `params.bucket_size == 0`, or when `datas == nullptr` while `count > 0`.
     */
    std::vector<std::vector<int64_t>>
    Run(const float* datas, uint64_t count, uint64_t dim, const RPTPartitionParams& params);

    RPTPartitionResult
    RunWithAllocator(const float* datas,
                     uint64_t count,
                     uint64_t dim,
                     const RPTPartitionParams& params);

private:
    void
    split(uint64_t begin, uint64_t end, uint64_t leaves, uint64_t depth, uint64_t node_seed);

    void
    generate_direction(uint64_t node_seed, float* direction) const;

    void
    project(uint64_t begin, uint64_t end, const float* direction);

    void
    emit_leaf(uint64_t begin, uint64_t end);

    // When max_depth is reached, splits [begin, end) into `leaves` chunks of nearly equal size
    // in the current permutation order.
    void
    emit_chunks(uint64_t begin, uint64_t end, uint64_t leaves);

private:
    const uint64_t dim_{0};
    Allocator* const allocator_{nullptr};

    const float* datas_{nullptr};
    uint64_t max_depth_{0};

    Vector<InnerIdType> ids_;    // permutation of vector ids, partitioned in place
    Vector<float> projections_;  // projection value of each vector on the current node
    Vector<float> direction_;    // projection direction of the current node
    RPTPartitionResult partitions_;
};

}  // namespace vsag
