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

#include "rpt_partition.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "impl/allocator/safe_allocator.h"
#include "simd/fp32_simd.h"
#include "vsag_exception.h"

namespace vsag {

namespace {

// splitmix64 finalizer, used to derive independent per-node seeds from the user seed.
// Deriving child seeds from the parent seed (instead of from a global generator) makes
// the projection direction of every node independent of the traversal order, which
// keeps the result reproducible once the build is parallelized.
uint64_t
mix_seed(uint64_t seed) {
    seed += 0x9E3779B97F4A7C15ULL;
    seed = (seed ^ (seed >> 30)) * 0xBF58476D1CE4E5B9ULL;
    seed = (seed ^ (seed >> 27)) * 0x94D049BB133111EBULL;
    return seed ^ (seed >> 31);
}

constexpr uint64_t LEFT_CHILD_SALT = 0x1ULL;
constexpr uint64_t RIGHT_CHILD_SALT = 0x2ULL;

}  // namespace

RPTPartition::RPTPartition(uint64_t dim, Allocator* allocator)
    : dim_(dim),
      allocator_(allocator),
      ids_(allocator),
      projections_(allocator),
      direction_(allocator) {
}

std::vector<std::vector<int64_t>>
RPTPartition::Run(const float* datas, uint64_t count, const RPTPartitionParams& params) {
    if (params.bucket_size == 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "bucket_size must be positive");
    }
    if (datas == nullptr && count > 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "datas cannot be null");
    }
    if (count > static_cast<uint64_t>(std::numeric_limits<InnerIdType>::max())) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "count exceeds the supported range");
    }

    datas_ = datas;
    params_ = params;
    partitions_.clear();
    if (count == 0) {
        return {};
    }

    ids_.resize(count);
    for (uint64_t i = 0; i < count; ++i) {
        ids_[i] = static_cast<InnerIdType>(i);
    }
    projections_.resize(count);
    direction_.resize(dim_);

    uint64_t leaves = (count + params.bucket_size - 1) / params.bucket_size;
    partitions_.reserve(leaves);
    this->split(0, count, leaves, 0, mix_seed(params.seed));

    ids_.clear();
    ids_.shrink_to_fit();
    projections_.clear();
    projections_.shrink_to_fit();
    direction_.clear();
    direction_.shrink_to_fit();
    datas_ = nullptr;
    return std::move(partitions_);
}

void
RPTPartition::split(
    uint64_t begin, uint64_t end, uint64_t leaves, uint64_t depth, uint64_t node_seed) {
    if (leaves <= 1) {
        this->emit_leaf(begin, end);
        return;
    }
    if (depth >= params_.max_depth) {
        this->emit_chunks(begin, end, leaves);
        return;
    }

    // Split n = q * leaves + r vectors so that the left child owns ceil(leaves / 2) leaves.
    // The left child receives q vectors per leaf plus at most one extra vector for each of
    // its leaves, which keeps every leaf at either q or q + 1 vectors.
    uint64_t count = end - begin;
    uint64_t quotient = count / leaves;
    uint64_t remainder = count % leaves;
    uint64_t left_leaves = (leaves + 1) / 2;
    uint64_t left_count = quotient * left_leaves + std::min(remainder, left_leaves);

    this->generate_direction(node_seed, direction_.data());
    this->project(begin, end, direction_.data());

    // Ties on the projection value are broken by id so that the cut position is always
    // reachable (e.g. for duplicated vectors) and the output is deterministic.
    auto comparator = [this](InnerIdType a, InnerIdType b) {
        if (projections_[a] != projections_[b]) {
            return projections_[a] < projections_[b];
        }
        return a < b;
    };
    std::nth_element(ids_.begin() + static_cast<int64_t>(begin),
                     ids_.begin() + static_cast<int64_t>(begin + left_count),
                     ids_.begin() + static_cast<int64_t>(end),
                     comparator);

    uint64_t mid = begin + left_count;
    this->split(begin, mid, left_leaves, depth + 1, mix_seed(node_seed ^ LEFT_CHILD_SALT));
    this->split(mid, end, leaves - left_leaves, depth + 1, mix_seed(node_seed ^ RIGHT_CHILD_SALT));
}

void
RPTPartition::generate_direction(uint64_t node_seed, float* direction) const {
    std::mt19937_64 generator(node_seed);
    std::normal_distribution<float> gaussian(0.0F, 1.0F);
    float norm = 0.0F;
    for (uint64_t i = 0; i < dim_; ++i) {
        direction[i] = gaussian(generator);
        norm += direction[i] * direction[i];
    }
    norm = std::sqrt(norm);
    if (norm > 0.0F) {
        for (uint64_t i = 0; i < dim_; ++i) {
            direction[i] /= norm;
        }
    }
}

void
RPTPartition::project(uint64_t begin, uint64_t end, const float* direction) {
    for (uint64_t i = begin; i < end; ++i) {
        InnerIdType id = ids_[i];
        float value = FP32ComputeIP(datas_ + static_cast<uint64_t>(id) * dim_, direction, dim_);
        // NaN breaks the strict weak ordering required by nth_element; rank such vectors last.
        if (std::isnan(value)) {
            value = std::numeric_limits<float>::infinity();
        }
        projections_[id] = value;
    }
}

void
RPTPartition::emit_leaf(uint64_t begin, uint64_t end) {
    std::vector<int64_t> partition;
    partition.reserve(end - begin);
    for (uint64_t i = begin; i < end; ++i) {
        partition.push_back(static_cast<int64_t>(ids_[i]));
    }
    partitions_.emplace_back(std::move(partition));
}

void
RPTPartition::emit_chunks(uint64_t begin, uint64_t end, uint64_t leaves) {
    uint64_t count = end - begin;
    uint64_t quotient = count / leaves;
    uint64_t remainder = count % leaves;
    uint64_t cursor = begin;
    for (uint64_t i = 0; i < leaves; ++i) {
        uint64_t chunk = quotient + (i < remainder ? 1 : 0);
        this->emit_leaf(cursor, cursor + chunk);
        cursor += chunk;
    }
}

tl::expected<std::vector<std::vector<int64_t>>, Error>
rpt_partition(uint64_t d, uint64_t n, const float* x, const RPTPartitionParams& params) {
    if (d == 0) {
        return tl::unexpected(Error(ErrorType::INVALID_ARGUMENT, "dim must be positive"));
    }
    try {
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        RPTPartition partitioner(d, allocator.get());
        return partitioner.Run(x, n, params);
    } catch (const VsagException& e) {
        return tl::unexpected(e.error_);
    } catch (const std::exception& e) {
        return tl::unexpected(Error(ErrorType::INTERNAL_ERROR, e.what()));
    }
}

}  // namespace vsag
