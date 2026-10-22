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

#include <algorithm>
#include <cstdint>
#include <limits>

#include "common.h"
#include "impl/inner_search_param.h"
#include "impl/searcher/searcher_utils.h"
#include "query_context.h"
#include "typing.h"
#include "vsag/allocator.h"

namespace vsag {

// Per-traversal state of the distance early stop. The disabled specialization is empty, so a
// traversal without early stop carries no extra state.
template <bool enabled>
class DistanceEarlyStop {
public:
    DistanceEarlyStop(Allocator* /*allocator*/,
                      const InnerSearchParam& /*param*/,
                      const QueryContext* /*ctx*/) {
    }

    void
    Record(float /*score*/) {
    }
};

// Keeps the k smallest finite result scores in a max-heap; the rule needs no result IDs. The stop
// threshold alpha * max(0, k-th score) is updated when the k-th score changes, so the check per
// expansion is one comparison. A k-th score of zero or below means k copies of the query are
// already held (cosine rounding can make it slightly negative); then any pending candidate with a
// score of zero or more stops the search.
template <>
class DistanceEarlyStop<true> {
public:
    DistanceEarlyStop(Allocator* allocator, const InnerSearchParam& param, const QueryContext* ctx)
        : best_(allocator) {
        CHECK_ARGUMENT(ctx != nullptr and ctx->distance_early_stop_alpha >= 1.0F,
                       "distance early stop requires an alpha of at least 1");
        CHECK_ARGUMENT(param.rerank_topk > 0,
                       "distance early stop requires a positive result count");
        CHECK_ARGUMENT(param.distance_batch_func == nullptr,
                       "distance early stop requires built-in distance");
        alpha_ = static_cast<double>(ctx->distance_early_stop_alpha);
        k_ = static_cast<uint64_t>(param.rerank_topk);
        best_.reserve(k_);
    }

    void
    Record(float score) {
        if (not is_finite_distance(score)) {
            return;
        }
        if (best_.size() < k_) {
            best_.push_back(score);
            std::push_heap(best_.begin(), best_.end());
            if (best_.size() == k_) {
                update_threshold();
            }
        } else if (score < best_.front()) {
            std::pop_heap(best_.begin(), best_.end());
            best_.back() = score;
            std::push_heap(best_.begin(), best_.end());
            update_threshold();
        }
    }

    // True once k results are held and the nearest pending distance reaches alpha times the
    // current k-th distance.
    [[nodiscard]] bool
    ShouldStop(float next_distance) const {
        return is_finite_distance(next_distance) and
               static_cast<double>(next_distance) >= threshold_;
    }

private:
    void
    update_threshold() {
        threshold_ = alpha_ * static_cast<double>(std::max(0.0F, best_.front()));
    }

    Vector<float> best_;
    uint64_t k_{0};
    double alpha_{0.0};
    double threshold_{std::numeric_limits<double>::infinity()};
};

}  // namespace vsag
