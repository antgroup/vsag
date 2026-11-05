
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

#include "multi_vector_computer.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "simd/fp32_simd.h"
#include "vsag_exception.h"

namespace vsag {

MultiVectorComputer::MultiVectorComputer(uint32_t dim, MetricType metric, Allocator* allocator)
    : dim_(dim), metric_(metric), allocator_(allocator), query_tokens_(allocator) {
}

void
MultiVectorComputer::SetQuery(const float* query_tokens, uint32_t query_token_count) {
    query_token_count_ = query_token_count;
    if (query_token_count == 0) {
        query_tokens_.clear();
        return;
    }
    const uint64_t total_floats = static_cast<uint64_t>(query_token_count) * dim_;
    try {
        query_tokens_.resize(total_floats);
    } catch (const std::bad_alloc&) {
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                            "bad alloc when resizing multi-vector query buffer");
    }
    std::memcpy(query_tokens_.data(), query_tokens, total_floats * sizeof(float));
}

void
MultiVectorComputer::ComputeDist(const uint8_t* codes, uint32_t token_count, float* dist) const {
    const auto* doc_tokens = reinterpret_cast<const float*>(codes);
    float total = 0.0F;

    if (metric_ == MetricType::METRIC_TYPE_IP) {
        for (uint32_t q = 0; q < query_token_count_; ++q) {
            const float* q_tok = query_tokens_.data() + static_cast<uint64_t>(q) * dim_;
            float min_dist = std::numeric_limits<float>::max();

            for (uint32_t d = 0; d < token_count; ++d) {
                const float* d_tok = doc_tokens + static_cast<uint64_t>(d) * dim_;
                float dist_val = 1.0F - FP32ComputeIP(q_tok, d_tok, dim_);
                if (dist_val < min_dist) {
                    min_dist = dist_val;
                }
            }

            total += min_dist;
        }
    } else if (metric_ == MetricType::METRIC_TYPE_L2SQR) {
        for (uint32_t q = 0; q < query_token_count_; ++q) {
            const float* q_tok = query_tokens_.data() + static_cast<uint64_t>(q) * dim_;
            float min_dist = std::numeric_limits<float>::max();

            for (uint32_t d = 0; d < token_count; ++d) {
                const float* d_tok = doc_tokens + static_cast<uint64_t>(d) * dim_;
                float dist_val = FP32ComputeL2Sqr(q_tok, d_tok, dim_);
                if (dist_val < min_dist) {
                    min_dist = dist_val;
                }
            }

            total += min_dist;
        }
    } else {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "unsupported metric type for MultiVectorComputer");
    }

    *dist = total;
}

}  // namespace vsag
