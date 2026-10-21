// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>

namespace vsag {

// Floyd sampling: uniform positions without replacement, independent of vector geometry.
inline std::vector<uint64_t>
mci_sample_positions(uint64_t population, uint64_t requested, uint64_t seed) {
    const auto count = std::min(population, requested);
    std::unordered_set<uint64_t> chosen;
    chosen.reserve(count * 2);
    std::mt19937_64 generator(seed);
    for (uint64_t j = population - count; j < population; ++j) {
        const auto draw = std::uniform_int_distribution<uint64_t>(0, j)(generator);
        if (not chosen.insert(draw).second) {
            chosen.insert(j);
        }
    }
    std::vector<uint64_t> result(chosen.begin(), chosen.end());
    std::sort(result.begin(), result.end());
    return result;
}

struct MCIDistanceSampleResult {
    float radius{0};
    uint64_t global_inside{0};
    uint64_t valid_inside{0};
    double enrichment{0};
    double lower{0};
    double upper{0};
    const char* prediction{"unknown"};
};

inline std::pair<double, double>
mci_wilson_interval(uint64_t successes, uint64_t count, double z) {
    // No observations constrain the probability; preserve the full possible interval.
    if (count == 0) {
        return {0.0, 1.0};
    }
    const auto n = static_cast<double>(count);
    const auto p = static_cast<double>(successes) / n;
    const auto z2 = z * z;
    const auto center = p + z2 / (2 * n);
    const auto width = z * std::sqrt(p * (1 - p) / n + z2 / (4 * n * n));
    const auto divisor = 1 + z2 / n;
    return {std::max(0.0, (center - width) / divisor), std::min(1.0, (center + width) / divisor)};
}

// The empirical radius and overlapping finite populations make these intervals a conservative
// decision heuristic, not a formal joint confidence guarantee for an adaptive quantile.
inline MCIDistanceSampleResult
mci_classify_distance_samples(const std::vector<float>& global,
                              const std::vector<float>& valid,
                              double quantile,
                              double negative_ratio,
                              double positive_ratio,
                              double z) {
    MCIDistanceSampleResult result;
    if (global.empty() or valid.empty()) {
        return result;
    }
    auto ordered = global;
    const auto rank = std::min<uint64_t>(
        global.size(),
        std::max<uint64_t>(
            1, static_cast<uint64_t>(std::ceil(quantile * static_cast<double>(global.size())))));
    std::nth_element(
        ordered.begin(), ordered.begin() + static_cast<int64_t>(rank) - 1, ordered.end());
    result.radius = ordered[rank - 1];
    result.global_inside =
        std::count_if(global.begin(), global.end(), [&](float d) { return d <= result.radius; });
    result.valid_inside =
        std::count_if(valid.begin(), valid.end(), [&](float d) { return d <= result.radius; });
    const auto a = static_cast<double>(result.global_inside) / static_cast<double>(global.size());
    const auto b = static_cast<double>(result.valid_inside) / static_cast<double>(valid.size());
    result.enrichment = b / a;
    const auto ai = mci_wilson_interval(result.global_inside, global.size(), z);
    const auto bi = mci_wilson_interval(result.valid_inside, valid.size(), z);
    result.lower = bi.first / ai.second;
    result.upper = ai.first > 0 ? bi.second / ai.first : INFINITY;
    result.prediction = result.upper <= negative_ratio   ? "negative"
                        : result.lower >= positive_ratio ? "positive"
                                                         : "neutral";
    return result;
}

}  // namespace vsag
