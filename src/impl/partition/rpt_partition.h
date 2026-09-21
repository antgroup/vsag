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

/**
 * @brief Random projection tree (RPT) partitioner.
 *
 * Splits a vector set into partitions whose sizes are bounded by the target bucket size.
 * Instead of stopping at the median once a node is small enough (which yields a power of
 * two leaves), every node is assigned the number of leaves `k = ceil(n / L)` it must
 * produce and is cut at the position that hands `ceil(k / 2)` leaves to the left child.
 * Every leaf therefore ends up with either `floor(N / K)` or `ceil(N / K)` vectors, where
 * `K = ceil(N / L)` is the total number of partitions.
 *
 * The permutation of ids is partitioned in place and no vector data is copied. The input
 * pointer is owned by the caller and is not retained after `Run` returns.
 */
class RPTPartition {
public:
    explicit RPTPartition(uint64_t dim, Allocator* allocator);

    /**
     * @brief Partitions `count` vectors of `dim_` dimensions.
     *
     * @throws VsagException(INVALID_ARGUMENT) when `params.bucket_size == 0`, or when
     *         `datas == nullptr` while `count > 0`.
     */
    std::vector<std::vector<int64_t>>
    Run(const float* datas, uint64_t count, const RPTPartitionParams& params);

private:
    void
    split(uint64_t begin, uint64_t end, uint64_t leaves, uint64_t depth, uint64_t node_seed);

    void
    generate_direction(uint64_t node_seed, float* direction) const;

    void
    project(uint64_t begin, uint64_t end, const float* direction);

    void
    emit_leaf(uint64_t begin, uint64_t end);

    // Splits [begin, end) into `leaves` chunks by id order, used when max_depth is reached.
    void
    emit_chunks(uint64_t begin, uint64_t end, uint64_t leaves);

private:
    const uint64_t dim_{0};
    Allocator* const allocator_{nullptr};

    const float* datas_{nullptr};
    RPTPartitionParams params_;

    Vector<InnerIdType> ids_;    // permutation of vector ids, partitioned in place
    Vector<float> projections_;  // projection value of each vector on the current node
    Vector<float> direction_;    // projection direction of the current node
    std::vector<std::vector<int64_t>> partitions_;
};

}  // namespace vsag
