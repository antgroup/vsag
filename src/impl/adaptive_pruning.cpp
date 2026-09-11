
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

#include "adaptive_pruning.h"

#include <algorithm>
#include <cmath>

#include "hash_types.h"

namespace vsag {

void
AdaptivePruningParameter::FromJson(const JsonType& json) {
    CHECK_ARGUMENT(json.IsObject(), "index parameters must be an object");
    const bool has_removed_fill =
        json.Contains("fill_rejected") || json.Contains("adaptive_pruning_fill_rejected");
    CHECK_ARGUMENT(not has_removed_fill,
                   "fill_rejected has been removed; rejected neighbors are never filled");
    *this = AdaptivePruningParameter{};
    if (json.Contains("adaptive_pruning")) {
        enabled = json["adaptive_pruning"].GetBool();
    }
    if (json.Contains("adaptive_pruning_adjust_step")) {
        adjust_step = json["adaptive_pruning_adjust_step"].GetFloat();
    }
    if (json.Contains("adaptive_pruning_apply_to_reverse")) {
        apply_to_reverse = json["adaptive_pruning_apply_to_reverse"].GetBool();
    }
    if (json.Contains("adaptive_pruning_apply_to_upper")) {
        apply_to_upper = json["adaptive_pruning_apply_to_upper"].GetBool();
    }
}

JsonType
AdaptivePruningParameter::ToJson() const {
    JsonType json;
    json["adaptive_pruning"].SetBool(enabled);
    json["adaptive_pruning_adjust_step"].SetFloat(adjust_step);
    json["adaptive_pruning_apply_to_reverse"].SetBool(apply_to_reverse);
    json["adaptive_pruning_apply_to_upper"].SetBool(apply_to_upper);
    return json;
}

void
AdaptivePruningParameter::Validate(float alpha) const {
    if (not enabled) {
        return;
    }
    const bool valid_thresholds = std::isfinite(alpha) && std::isfinite(adjust_step) &&
                                  adjust_step >= 0 && alpha - 2 * adjust_step > 0 &&
                                  std::isfinite(alpha + 3 * adjust_step);
    CHECK_ARGUMENT(valid_thresholds,
                   "adaptive_pruning requires finite alpha/step, step >= 0 and alpha - 2*step > 0");
    CHECK_ARGUMENT(not apply_to_upper,
                   "adaptive_pruning currently supports only bottom-layer selection");
}

Vector<PruningCandidate>
select_edges_adaptive(Vector<PruningCandidate>& candidates,
                      InnerIdType center,
                      uint64_t target_degree,
                      float alpha,
                      const AdaptivePruningParameter& parameter,
                      const std::function<float(InnerIdType, InnerIdType)>& distance,
                      Allocator* allocator,
                      AdaptivePruningStats* stats) {
    CHECK_ARGUMENT(parameter.enabled, "select_edges_adaptive requires enabled adaptive_pruning");
    parameter.Validate(alpha);
    AdaptivePruningStats result;
    result.second_alpha = alpha;
    Vector<PruningCandidate> accepted(allocator);
    if (target_degree == 0 || candidates.empty()) {
        if (stats != nullptr) {
            *stats = result;
        }
        return accepted;
    }
    for (const auto& candidate : candidates) {
        const bool valid_distance = std::isfinite(candidate.first) && candidate.first >= 0;
        CHECK_ARGUMENT(valid_distance,
                       "adaptive_pruning requires finite non-negative L2 distances");
    }
    std::sort(candidates.begin(), candidates.end());
    UnorderedSet<InnerIdType> seen(allocator);
    candidates.erase(
        std::remove_if(
            candidates.begin(),
            candidates.end(),
            [&](const auto& c) { return c.second == center || not seen.insert(c.second).second; }),
        candidates.end());
    accepted.reserve(std::min<uint64_t>(target_degree, candidates.size()));
    Vector<PruningCandidate> rejected(allocator);
    auto scan = [&](const Vector<PruningCandidate>& input,
                    float threshold,
                    Vector<PruningCandidate>& failures) {
        for (const auto& candidate : input) {
            if (accepted.size() == target_degree) {
                break;
            }
            bool good = true;
            for (const auto& selected : accepted) {
                float pair_distance = distance(selected.second, candidate.second);
                ++result.distance_calls;
                const bool valid_distance = std::isfinite(pair_distance) && pair_distance >= 0;
                CHECK_ARGUMENT(
                    valid_distance,
                    "adaptive_pruning requires finite non-negative pairwise L2 distances");
                if (threshold * pair_distance < candidate.first) {
                    good = false;
                    break;
                }
            }
            if (good) {
                accepted.push_back(candidate);
            } else {
                failures.push_back(candidate);
            }
        }
    };
    scan(candidates, alpha, rejected);
    result.initial_accepted = accepted.size();
    result.initial_rejected = rejected.size();
    if (parameter.adjust_step == 0) {
        // A zero step keeps the single-pass result, including any unused capacity.
    } else if (not accepted.empty() && accepted.size() < target_degree) {
        result.branch = AdaptivePruningBranch::RELAX;
        const double ratio =
            static_cast<double>(target_degree) / static_cast<double>(accepted.size());
        int steps = 1;
        if (ratio > 3) {
            steps = 3;
        } else if (ratio > 1.5) {
            steps = 2;
        }
        result.second_alpha = alpha + static_cast<float>(steps) * parameter.adjust_step;
        Vector<PruningCandidate> remaining(allocator);
        scan(rejected, result.second_alpha, remaining);
    } else if (accepted.size() == target_degree) {
        result.branch = AdaptivePruningBranch::TIGHTEN;
        const double ratio =
            static_cast<double>(rejected.size()) / static_cast<double>(target_degree);
        int steps = 2;
        if (ratio >= 5) {
            steps = 0;
        } else if (ratio >= 2.5) {
            steps = 1;
        }
        result.second_alpha = alpha - static_cast<float>(steps) * parameter.adjust_step;
        accepted.clear();
        rejected.clear();
        // The normalized original list is exactly sorted(A+B) followed by unscanned tail T.
        scan(candidates, result.second_alpha, rejected);
        if (accepted.size() < target_degree) {
            Vector<PruningCandidate> remaining(allocator);
            scan(rejected, alpha, remaining);
        }
    }
    std::sort(accepted.begin(), accepted.end());
    if (stats != nullptr) {
        *stats = result;
    }
    return accepted;
}

}  // namespace vsag
