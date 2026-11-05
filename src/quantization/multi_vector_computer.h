
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

#include "computer.h"
#include "metric_type.h"
#include "typing.h"
#include "utils/pointer_define.h"
#include "vsag/allocator.h"

namespace vsag {

DEFINE_POINTER(MultiVectorComputer)

/**
 * @brief Computer that scores a multi-vector (ColBERT-style) doc against a multi-vector query
 *        using the MaxSim distance:
 *
 *            MaxSim(query, doc) = sum_{q_tok in query} min_{d_tok in doc} dist(q_tok, d_tok)
 *
 *        Smaller is more similar (every per-token term is itself a distance):
 *          - IP   : 1 - <q_tok, d_tok>
 *          - L2   : ||q_tok - d_tok||^2
 *          - COSINE will be added in a follow-up PR (normalize then reuse IP).
 *
 *        The class is intentionally storage-agnostic: ComputeDist accepts a flat
 *        token-codes buffer (no length prefix) plus an explicit token_count. The
 *        caller (typically MultiVectorDataCell) is responsible for stripping any
 *        on-disk prefix before passing the pointer in.
 *
 *        First version supports FP32 token codes only (codes are reinterpreted as
 *        const float*).
 */
class MultiVectorComputer : public ComputerInterface {
public:
    MultiVectorComputer(uint32_t dim, MetricType metric, Allocator* allocator);

    ~MultiVectorComputer() override = default;

    /**
     * @brief Bind a query (a flat array of `query_token_count * dim` floats).
     *
     * The query is copied into an internal buffer so the caller may release the
     * input afterwards. May be called multiple times to rebind a different query.
     */
    void
    SetQuery(const float* query_tokens, uint32_t query_token_count);

    /**
     * @brief Compute MaxSim(query, doc).
     *
     * @param codes        pure token codes for the doc, reinterpretable as
     *                     `const float*` of `token_count * dim` floats. Must NOT
     *                     contain a token-count prefix.
     * @param token_count  number of tokens in the doc (>= 1).
     * @param dist         output distance (smaller is more similar).
     */
    void
    ComputeDist(const uint8_t* codes, uint32_t token_count, float* dist) const;

    [[nodiscard]] uint32_t
    GetDim() const {
        return dim_;
    }

    [[nodiscard]] uint32_t
    GetQueryTokenCount() const {
        return query_token_count_;
    }

    [[nodiscard]] MetricType
    GetMetric() const {
        return metric_;
    }

private:
    uint32_t dim_{0};
    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};
    Allocator* const allocator_{nullptr};

    Vector<float> query_tokens_;
    uint32_t query_token_count_{0};
};

}  // namespace vsag
