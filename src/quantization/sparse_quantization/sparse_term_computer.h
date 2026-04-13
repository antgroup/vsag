
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

/**
 * @file sparse_term_computer.h
 * @brief Sparse term computer for sparse vector distance computation.
 */

#pragma once

#include <cstdint>
#include <memory>

#include "algorithm/sindi/sindi_parameter.h"
#include "metric_type.h"
#include "utils/pointer_define.h"
#include "utils/sparse_vector_transform.h"

namespace vsag {

/**
 * @brief Parameters for quantization with min/max bounds.
 */
struct QuantizationParams {
    float min_val = 0.0f;  ///< Minimum value for quantization.
    float max_val = 0.0f;  ///< Maximum value for quantization.
    float diff = 1.0f;     ///< Difference between max and min values.
};

static constexpr int INVALID_TERM = -1;  ///< Invalid term marker.

DEFINE_POINTER(SparseTermComputer)

/**
 * @brief Sparse term computer for efficient sparse vector distance computation.
 *
 * Manages query pruning and term-based distance accumulation for sparse vectors.
 */
class SparseTermComputer {
public:
    ~SparseTermComputer() = default;

    /**
     * @brief Constructs a sparse term computer from query vector.
     * @param sparse_query Sparse query vector.
     * @param search_param SINDI search parameters for pruning.
     * @param allocator Memory allocator.
     */
    explicit SparseTermComputer(const SparseVector& sparse_query,
                                const SINDISearchParameter& search_param,
                                Allocator* allocator = nullptr)
        : sorted_query_(allocator),
          query_retain_ratio_(1.0F - search_param.query_prune_ratio),
          term_retain_ratio_(1.0F - search_param.term_prune_ratio),
          raw_query_(sparse_query) {
        SetQuery(sparse_query);
    }

    /**
     * @brief Sets the query vector for computation.
     * @param sparse_query Sparse query vector.
     */
    void
    SetQuery(const SparseVector& sparse_query) {
        sort_sparse_vector(sparse_query, sorted_query_);

        pruned_len_ = (uint32_t)(query_retain_ratio_ * sparse_query.len_);
        if (pruned_len_ == 0) {
            if (sorted_query_.size() != 0) {
                pruned_len_ = 1;
            }
        }

        for (auto i = 0; i < sorted_query_.size(); i++) {
            sorted_query_[i].second *= -1;  // note that: dist_ip = -1 * query * base
        }
    }

    /**
     * @brief Accumulates distance contributions from term data.
     * @tparam T Data type for term values.
     * @param term_iterator Iterator index for current term.
     * @param term_ids Array of term IDs.
     * @param term_datas Array of term values.
     * @param term_count Number of terms.
     * @param global_dists Global distance accumulator array.
     */
    template <class T>
    void
    ScanForAccumulate(uint32_t term_iterator,
                      const uint16_t* term_ids,
                      const T* term_datas,
                      uint32_t term_count,
                      float* global_dists) {
        float query_val = sorted_query_[term_iterator].second;

        // TODO(ZXY): add prefetch to decrease cache miss like:
        //  __builtin_prefetch(term_ids + term_count / 2, 0, 3);
        //  __builtin_prefetch(term_datas + term_count / 2, 0, 3);
        //  __builtin_prefetch(global_dists + term_ids[term_count / 2], 0, 3);

        for (auto i = 0; i < term_count; i++) {
            global_dists[term_ids[i]] += query_val * term_datas[i];
        }
    }

    /**
     * @brief Calculates distance for a specific target ID.
     * @param term_iterator Iterator index for current term.
     * @param term_ids Array of term IDs.
     * @param term_datas Array of term values.
     * @param term_count Number of terms.
     * @param target_id Target ID to find.
     * @param dist Output distance accumulator.
     */
    inline void
    ScanForCalculateDist(uint32_t term_iterator,
                         const uint16_t* term_ids,
                         const float* term_datas,
                         uint32_t term_count,
                         uint16_t target_id,
                         float* dist) {
        float query_val = sorted_query_[term_iterator].second;

        for (auto i = 0; i < term_count; i++) {
            if (term_ids[i] == target_id) {
                *dist += query_val * term_datas[i];
                break;
            }
        }
    }

    /**
     * @brief Checks if there are more terms to process.
     * @return True if more terms available.
     */
    inline bool
    HasNextTerm() {
        return term_iterator_ < pruned_len_;
    }

    /**
     * @brief Gets the next term iterator and advances.
     * @return Current term iterator value.
     */
    inline uint32_t
    NextTermIter() {
        return term_iterator_++;
    }

    /**
     * @brief Resets the term iterator to beginning.
     */
    inline void
    ResetTerm() {
        term_iterator_ = 0;
    }

    /**
     * @brief Gets the term ID at given iterator position.
     * @param term_iterator Iterator position.
     * @return Term ID.
     */
    uint32_t
    GetTerm(uint32_t term_iterator) {
        return sorted_query_[term_iterator].first;
    }

public:
    Vector<std::pair<uint32_t, float>> sorted_query_;  ///< Sorted query (id, value) pairs.

    const SparseVector& raw_query_;  ///< Reference to original query.

    float query_retain_ratio_{0.0F};  ///< Ratio of query terms to retain.

    float term_retain_ratio_{0.0F};  ///< Ratio of base terms to retain.

    uint32_t pruned_len_{0};  ///< Length after pruning.

    uint32_t term_iterator_{0};  ///< Current term iterator.

    Allocator* const allocator_{nullptr};  ///< Memory allocator.
};
}  // namespace vsag
