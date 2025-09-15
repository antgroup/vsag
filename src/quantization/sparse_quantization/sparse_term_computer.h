
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
#include <memory>

#include "algorithm/sindi/sindi_parameter.h"
#include "metric_type.h"
#include "utils/pointer_define.h"
#include "utils/sparse_vector_transform.h"
namespace vsag {

static constexpr int INVALID_TERM = -1;
DEFINE_POINTER(SparseTermComputer)
class SparseTermComputer {
public:
    ~SparseTermComputer() = default;

    explicit SparseTermComputer(const SparseVector& sparse_query,
                                const SINDISearchParameter& search_param,
                                Allocator* allocator = nullptr)
        : sorted_query_(allocator),
          query_retain_ratio_(1.0F - search_param.query_prune_ratio),
          term_retain_ratio_(1.0F - search_param.term_prune_ratio),
          raw_query_(sparse_query) {
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

    inline void
    ScanForAccumulate(uint32_t term_iterator,
                      const uint32_t* term_ids,
                      const float* term_datas,
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

    inline void
    ScanForCalculateDist(uint32_t term_iterator,
                         const uint32_t* term_ids,
                         const float* term_datas,
                         uint32_t term_count,
                         uint32_t target_id,
                         float* dist) {
        float query_val = sorted_query_[term_iterator].second;

        for (auto i = 0; i < term_count; i++) {
            if (term_ids[i] == target_id) {
                *dist += query_val * term_datas[i];
                break;
            }
        }
    }

    inline bool
    HasNextTerm() {
        return term_iterator_ < pruned_len_;
    }

    inline uint32_t
    NextTermIter() {
        return term_iterator_++;
    }

    inline void
    ResetTerm() {
        term_iterator_ = 0;
    }

    uint32_t
    GetTerm(uint32_t term_iterator) {
        return sorted_query_[term_iterator].first;
    }

public:
    Vector<std::pair<uint32_t, float>> sorted_query_;

    const SparseVector& raw_query_;

    float query_retain_ratio_{0.0F};

    float term_retain_ratio_{0.0F};

    uint32_t pruned_len_{0};

    uint32_t term_iterator_{0};

    Allocator* const allocator_{nullptr};
};
}  // namespace vsag
