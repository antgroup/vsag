
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
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "common.h"

namespace vsag {

inline constexpr uint64_t DEFAULT_CAQ_ROUNDS = 6;

// Return the norm of the centered scalar codes. Callers retain their own record layout and
// metric-specific correction factors. Wider code types support variable-bit SAQ segments.
template <typename CodeType>
float
CodeAdjustmentQuantize(const float* data,
                       uint64_t dim,
                       uint64_t bits,
                       CodeType* code,
                       uint64_t rounds = DEFAULT_CAQ_ROUNDS) {
    static_assert(std::is_same_v<CodeType, uint8_t> or std::is_same_v<CodeType, uint16_t>);
    CHECK_ARGUMENT(bits > 0 and bits <= std::numeric_limits<CodeType>::digits,
                   "CAQ bit width exceeds the scalar code type");
    // CAQ starts from an LVQ grid and improves cosine alignment with coordinate adjustment.
    // Each coordinate moves by at most one level per round, keeping the complexity O(rounds * D).
    constexpr double adjustment_epsilon = 1e-8;
    const uint32_t code_max = (1U << bits) - 1U;
    const double center = 0.5 * static_cast<double>(code_max);

    double max_abs = 0.0;
    for (uint64_t d = 0; d < dim; ++d) {
        max_abs = std::max(max_abs, std::fabs(static_cast<double>(data[d])));
    }

    if (max_abs <= 0.0) {
        std::fill_n(code, dim, static_cast<CodeType>(code_max / 2U));
        return 1.0F;
    }

    const double delta = 2.0 * max_abs / static_cast<double>(code_max + 1U);
    const double inv_delta = 1.0 / delta;
    double ip = 0.0;
    double norm_sqr = 0.0;
    for (uint64_t d = 0; d < dim; ++d) {
        const double scaled = (static_cast<double>(data[d]) + max_abs) * inv_delta;
        auto quantized = static_cast<int64_t>(std::floor(scaled));
        quantized = std::clamp<int64_t>(quantized, 0, code_max);
        code[d] = static_cast<CodeType>(quantized);

        const double centered = static_cast<double>(quantized) - center;
        ip += static_cast<double>(data[d]) * centered;
        norm_sqr += centered * centered;
    }

    auto alignment_score = [](double inner_product, double squared_norm) {
        return squared_norm > 0.0 ? inner_product * inner_product / squared_norm : 0.0;
    };

    for (uint64_t round = 0; round < rounds; ++round) {
        bool adjusted = false;
        for (uint64_t d = 0; d < dim; ++d) {
            const int32_t current_code = code[d];
            const double current_value = static_cast<double>(current_code) - center;
            int32_t best_code = current_code;
            double best_ip = ip;
            double best_norm_sqr = norm_sqr;
            double best_score = alignment_score(ip, norm_sqr);

            for (int32_t direction = -1; direction <= 1; direction += 2) {
                const int32_t candidate_code = current_code + direction;
                if (candidate_code < 0 or candidate_code > static_cast<int32_t>(code_max)) {
                    continue;
                }

                const double candidate_ip =
                    ip + static_cast<double>(direction) * static_cast<double>(data[d]);
                if (candidate_ip < 0.0) {
                    continue;
                }
                const double candidate_norm_sqr =
                    norm_sqr + 2.0 * current_value * static_cast<double>(direction) + 1.0;
                const double candidate_score = alignment_score(candidate_ip, candidate_norm_sqr);
                const double tolerance = adjustment_epsilon * std::max(1.0, std::fabs(best_score));
                if (candidate_score > best_score + tolerance) {
                    best_code = candidate_code;
                    best_ip = candidate_ip;
                    best_norm_sqr = candidate_norm_sqr;
                    best_score = candidate_score;
                }
            }

            if (best_code != current_code) {
                code[d] = static_cast<CodeType>(best_code);
                ip = best_ip;
                norm_sqr = best_norm_sqr;
                adjusted = true;
            }
        }

        if (not adjusted) {
            break;
        }
    }

    norm_sqr = 0.0;
    for (uint64_t d = 0; d < dim; ++d) {
        const double centered = static_cast<double>(code[d]) - center;
        norm_sqr += centered * centered;
    }

    float y_norm = static_cast<float>(std::sqrt(norm_sqr));
    if (not std::isfinite(y_norm) or y_norm <= 0.0F) {
        y_norm = 1.0F;
    }
    return y_norm;
}

}  // namespace vsag
