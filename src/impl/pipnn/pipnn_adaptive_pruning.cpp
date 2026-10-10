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

#include "pipnn_adaptive_pruning.h"

#include <algorithm>
#include <numeric>

#include "parameter.h"
#include "utils/float_utils.h"

namespace vsag {

void
validate_pipnn_adaptive_pruning(float alpha, float adjust_step) {
    const bool valid = IsFiniteFloatBits(alpha) and IsFiniteFloatBits(adjust_step) and
                       adjust_step >= 0 and alpha - 2 * adjust_step > 0 and
                       IsFiniteFloatBits(alpha + 3 * adjust_step);
    CHECK_ARGUMENT(valid,
                   "PiPNN adaptive pruning requires finite alpha/step, step >= 0 and "
                   "alpha - 2*step > 0");
}

Vector<uint64_t>
select_pipnn_edges_adaptive(const Vector<std::pair<float, uint32_t>>& ordered,
                            uint64_t target_degree,
                            float alpha,
                            float adjust_step,
                            const std::function<float(uint32_t, uint32_t)>& distance,
                            Allocator* allocator,
                            PiPNNAdaptivePruningStats* stats) {
    validate_pipnn_adaptive_pruning(alpha, adjust_step);
    PiPNNAdaptivePruningStats result;
    result.second_alpha = alpha;
    Vector<uint64_t> accepted(allocator);
    if (target_degree == 0 or ordered.empty()) {
        if (stats != nullptr) {
            *stats = result;
        }
        return accepted;
    }
    for (const auto& candidate : ordered) {
        const bool valid_candidate = IsFiniteFloatBits(candidate.first) and candidate.first >= 0;
        CHECK_ARGUMENT(valid_candidate,
                       "PiPNN adaptive pruning requires finite non-negative L2 distances");
    }
    accepted.reserve(std::min<uint64_t>(target_degree, ordered.size()));
    Vector<uint64_t> positions(ordered.size(), allocator);
    std::iota(positions.begin(), positions.end(), uint64_t{0});
    Vector<uint64_t> rejected(allocator);
    auto scan = [&](const Vector<uint64_t>& input, float threshold, Vector<uint64_t>& failures) {
        for (const auto position : input) {
            if (accepted.size() == target_degree) {
                break;
            }
            const auto& candidate = ordered[position];
            bool keep = true;
            for (const auto selected : accepted) {
                const float pair_distance = distance(ordered[selected].second, candidate.second);
                ++result.distance_calls;
                const bool valid_pair = IsFiniteFloatBits(pair_distance) and pair_distance >= 0;
                CHECK_ARGUMENT(
                    valid_pair,
                    "PiPNN adaptive pruning requires finite non-negative pairwise L2 distances");
                if (threshold * pair_distance < candidate.first) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                accepted.emplace_back(position);
            } else {
                failures.emplace_back(position);
            }
        }
    };

    scan(positions, alpha, rejected);
    result.initial_accepted = accepted.size();
    result.initial_rejected = rejected.size();
    if (adjust_step == 0) {
        // PiPNN already prunes short reservoirs and never fills rejected neighbors.
        // Keeping the same candidate order makes this the original single-pass result.
    } else if (not accepted.empty() and accepted.size() < target_degree) {
        result.branch = PiPNNAdaptivePruningBranch::RELAX;
        const double ratio =
            static_cast<double>(target_degree) / static_cast<double>(accepted.size());
        const int steps = ratio > 3 ? 3 : (ratio > 1.5 ? 2 : 1);
        result.second_alpha = alpha + static_cast<float>(steps) * adjust_step;
        Vector<uint64_t> remaining(allocator);
        scan(rejected, result.second_alpha, remaining);
    } else if (accepted.size() == target_degree) {
        result.branch = PiPNNAdaptivePruningBranch::TIGHTEN;
        const double ratio =
            static_cast<double>(rejected.size()) / static_cast<double>(target_degree);
        const int steps = ratio >= 5 ? 0 : (ratio >= 2.5 ? 1 : 2);
        result.second_alpha = alpha - static_cast<float>(steps) * adjust_step;
        accepted.clear();
        auto tightened_rejects = std::move(rejected);
        tightened_rejects.clear();
        // Revisit the complete original reservoir, including the unscanned first-pass tail.
        scan(positions, result.second_alpha, tightened_rejects);
        if (accepted.size() < target_degree) {
            Vector<uint64_t> remaining(allocator);
            scan(tightened_rejects, alpha, remaining);
        }
    }
    std::sort(accepted.begin(), accepted.end());
    if (stats != nullptr) {
        *stats = result;
    }
    return accepted;
}

}  // namespace vsag
