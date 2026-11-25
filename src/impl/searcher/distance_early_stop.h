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

#include "impl/searcher/searcher_utils.h"
#include "typing.h"
#include "vsag/allocator.h"

namespace vsag {

// Per-traversal state of the distance early stop. Keeps the k smallest finite result scores in a
// max-heap and caches the stop threshold alpha * max(0, k-th score), so the check per expansion is
// one comparison. A k-th score of zero or below (k copies of the query; cosine rounding can make
// it slightly negative) sets the threshold to zero.
class DistanceEarlyStop {
public:
    DistanceEarlyStop(Allocator* allocator, int64_t k, float alpha)
        : best_(allocator), k_(static_cast<uint64_t>(k)), alpha_(alpha) {
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

    // True once k results are held and the nearest pending distance reaches the threshold.
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
    uint64_t k_;
    double alpha_;
    double threshold_{std::numeric_limits<double>::infinity()};
};

}  // namespace vsag
