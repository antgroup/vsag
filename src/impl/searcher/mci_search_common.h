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

#include <cmath>
#include <cstring>
#include <vector>

#include "metric_type.h"
#include "simd/fp32_simd.h"
#include "typing.h"

namespace vsag {

struct MCIEpochMarks {
    std::vector<uint16_t> marks;
    uint16_t tag{1};

    void
    Reset(uint64_t size) {
        if (marks.size() < size) {
            marks.assign(size, 0);
            tag = 1;
            return;
        }
        ++tag;
        if (tag == 0) {
            std::memset(marks.data(), 0, marks.size() * sizeof(uint16_t));
            tag = 1;
        }
    }

    [[nodiscard]] bool
    Get(InnerIdType id) const {
        return marks[id] == tag;
    }

    void
    Set(InnerIdType id) {
        marks[id] = tag;
    }
};

inline float
CalcMCICosineQueryInvNorm(const float* query, uint64_t dim) {
    const auto norm = FP32ComputeIP(query, query, dim);
    if (norm <= 0.0F) {
        return 0.0F;
    }
    return 1.0F / std::sqrt(norm);
}

inline float
MCIRawFloatDistance(const float* query,
                    const float* vector,
                    uint64_t dim,
                    MetricType metric,
                    float cosine_query_inv_norm,
                    bool cosine_hold_mold) {
    if (metric == MetricType::METRIC_TYPE_L2SQR) {
        return FP32ComputeL2Sqr(query, vector, dim);
    }

    auto similarity = FP32ComputeIP(query, vector, dim);
    if (metric == MetricType::METRIC_TYPE_COSINE) {
        similarity *= cosine_query_inv_norm;
        if (cosine_hold_mold) {
            if (vector[dim] <= 0.0F) {
                return 1.0F;
            }
            similarity /= vector[dim];
        }
    }
    return 1.0F - similarity;
}

}  // namespace vsag
