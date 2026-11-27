// Copyright 2026-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>

#include "container_types.h"
#include "impl/heap/distance_heap.h"
#include "utils/float_utils.h"

namespace vsag {

// Return the nearest k distinct live predecessors in the ID order expected by MCI.
// The caller owns the heap copy, so pruning HGraph edges can proceed independently.
template <typename IsRemoved>
bool
SelectMciAddCandidates(Vector<DistanceHeap::DistanceRecord>& scores,
                       InnerIdType row,
                       uint64_t k,
                       IsRemoved is_removed,
                       Vector<InnerIdType>& ids) {
    ids.clear();
    if (k == 0 or scores.size() < k) {
        return false;
    }
    scores.erase(std::remove_if(scores.begin(),
                                scores.end(),
                                [&](const auto& p) {
                                    return p.second >= row or not IsFiniteFloatBits(p.first) or
                                           is_removed(p.second);
                                }),
                 scores.end());
    std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
        return a.second == b.second ? a.first < b.first : a.second < b.second;
    });
    scores.erase(std::unique(scores.begin(),
                             scores.end(),
                             [](const auto& a, const auto& b) { return a.second == b.second; }),
                 scores.end());
    if (scores.size() < k) {
        return false;
    }
    if (scores.size() > k) {
        std::nth_element(scores.begin(), scores.begin() + k, scores.end());
    }
    for (uint64_t i = 0; i < k; ++i) {
        ids.push_back(scores[i].second);
    }
    std::sort(ids.begin(), ids.end());
    return true;
}

}  // namespace vsag
