// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

#include "basic_types.h"
#include "metric_type.h"
#include "typing.h"
#include "vsag_exception.h"

namespace vsag {

inline constexpr uint32_t K_FUSED_DEFAULT_CLUSTER_COUNT = 16;
inline constexpr uint32_t K_FUSED_CODEC_VERSION = 3;
inline constexpr uint64_t K_FUSED_CODEC_HEADER_SIZE = 2 * sizeof(uint32_t) + sizeof(uint64_t);

// Validate dimension/count and size arithmetic before returning the serialized codec size.
inline uint64_t
CheckedFusedCodecSize(uint64_t dim, uint64_t count) {
    CHECK_ARGUMENT(dim > 0 and count > 0 and count <= std::numeric_limits<int32_t>::max(),
                   "invalid fused codec dimension or cluster count");
    CHECK_ARGUMENT(dim <= std::numeric_limits<uint64_t>::max() / (2 * sizeof(float)),
                   "fused codec dimension overflow");
    const uint64_t stride = dim * 2 * sizeof(float);
    CHECK_ARGUMENT(
        count <= (std::numeric_limits<uint64_t>::max() - K_FUSED_CODEC_HEADER_SIZE) / stride,
        "fused codec size overflow");
    return K_FUSED_CODEC_HEADER_SIZE + count * stride;
}

// A single query owns this cache across graph routing, traversal and reranking.
// Not thread-safe: these stages must access it sequentially. Concurrent queries/workers need
// separate caches; ready and computed_count intentionally require no atomic operations.
class RaBitQFusedQueryCache {
public:
    explicit RaBitQFusedQueryCache(Allocator* allocator)
        : add(allocator), error(allocator), ready(allocator) {
    }

    void
    Initialize(const float* query,
               const float* centers,
               const double* norms,
               uint64_t dim,
               uint32_t count,
               MetricType metric) {
        query_ = query;
        centers_ = centers;
        norms_ = norms;
        dim_ = dim;
        metric_ = metric;
        query_norm_ = 0.0;
        for (uint64_t d = 0; d < dim; ++d) {
            query_norm_ += static_cast<double>(query[d]) * query[d];
        }
        add.resize(count);
        error.resize(count);
        ready.assign(count, 0);
        computed_count = 0;
    }

    void
    Ensure(uint32_t id) {
        assert(id < ready.size());
        if (ready[id] != 0) {
            return;
        }
        const auto* center = centers_ + uint64_t{id} * dim_;
        double dot = 0.0;
        for (uint64_t d = 0; d < dim_; ++d) {
            dot += static_cast<double>(query_[d]) * center[d];
        }
        const double scale = query_norm_ + norms_[id];
        double squared = scale - 2.0 * dot;
        // Subtraction loses precision near the center. Recompute the residual directly there.
        if (squared <= 1e-6 * scale) {
            squared = 0.0;
            for (uint64_t d = 0; d < dim_; ++d) {
                const double residual = static_cast<double>(query_[d]) - center[d];
                squared += residual * residual;
            }
        }
        add[id] = static_cast<float>(metric_ == MetricType::METRIC_TYPE_IP ? -dot : squared);
        error[id] = static_cast<float>(std::sqrt(squared));
        ready[id] = 1;
        ++computed_count;
    }

    Vector<float> add;
    Vector<float> error;
    Vector<uint8_t> ready;
    uint64_t computed_count{0};

private:
    const float* query_{nullptr};
    const float* centers_{nullptr};
    const double* norms_{nullptr};
    uint64_t dim_{0};
    double query_norm_{0.0};
    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};
};

struct RaBitQFusedCodeView {
    const uint8_t* one_bit_code{nullptr};
    const uint8_t* supplement_code{nullptr};
    uint32_t cluster_id{0};
};

/**
 * Internal non-owning bridge between the RaBitQ model/query processor and a fused node slab.
 *
 * Fused codes use cluster-residual semantics. The legacy split 1+7 format stores HNSW-compatible
 * BinData/ExData records; the x=1..4 native formats retain the ordinary split bit-plane encoding.
 * Callers must retain the cluster id and use the fused distance methods selected by the codec.
 */
class RabitQFusedInterface {
public:
    virtual ~RabitQFusedInterface() = default;

    [[nodiscard]] virtual bool
    GetFusedCodeView(InnerIdType id, RaBitQFusedCodeView& view) const = 0;

    virtual void
    SetFusedCodes(InnerIdType id,
                  uint32_t cluster_id,
                  const uint8_t* one_bit_code,
                  const uint8_t* supplement_code) = 0;

    virtual void
    PrefetchFusedCodes(InnerIdType id, bool include_supplement) const = 0;

    [[nodiscard]] virtual uint64_t
    FusedOneBitCodeSize() const = 0;

    [[nodiscard]] virtual uint64_t
    FusedSupplementCodeSize() const = 0;
};

}  // namespace vsag
