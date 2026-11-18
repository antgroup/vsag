// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "lite/rabitq_constants.h"

// FHT/Kac transform primitives and the RaBitQ Model used by encoding and
// decoding. Pure computation: no IO, threads, or allocator dependency.

namespace vsag::lite::detail::rabitq {

inline void
codec_require(bool value, const char* message) {
    if (not value) {
        throw std::runtime_error(message);
    }
}

inline void
fht(float* values, uint64_t dim) {
    for (uint64_t step = 1; step < dim; step *= 2) {
        for (uint64_t block = 0; block < dim; block += step * 2) {
            for (uint64_t lane = 0; lane < step; ++lane) {
                const float left = values[block + lane];
                const float right = values[block + step + lane];
                values[block + lane] = left + right;
                values[block + step + lane] = left - right;
            }
        }
    }
}

inline void
kacs_walk(std::vector<float>& values) {
    const uint64_t half = values.size() / 2;
    const uint64_t base = values.size() % 2;
    const uint64_t offset = base + half;
    for (uint64_t i = 0; i < half; ++i) {
        const float left = values[i];
        const float right = values[i + offset];
        values[i] = left + right;
        values[i + offset] = left - right;
    }
    if (base != 0) {
        values[half] *= std::sqrt(2.0F);
    }
}

inline uint64_t
floor_power_of_two(uint64_t value) {
    uint64_t result = 1;
    while (result <= value / 2) {
        result *= 2;
    }
    return result;
}

struct Model {
    uint64_t dim{};
    std::vector<float> centroid;
    std::vector<uint8_t> flips;

    void
    Transform(std::vector<float>& values) const {
        codec_require(values.size() == dim, "transform dimension mismatch");
        const uint64_t bytes = (dim + 7) / 8;
        const uint64_t truncated_dim = floor_power_of_two(dim);
        const float scale = 1.0F / std::sqrt(static_cast<float>(truncated_dim));
        for (uint32_t round = 0; round < K_ROUNDS; ++round) {
            for (uint64_t d = 0; d < dim; ++d) {
                if ((flips[round * bytes + d / 8] & (1U << (d % 8))) != 0U) {
                    values[d] = -values[d];
                }
            }
            float* block = round % 2 == 0 ? values.data() : values.data() + dim - truncated_dim;
            fht(block, truncated_dim);
            for (uint64_t d = 0; d < truncated_dim; ++d) {
                block[d] *= scale;
            }
            if (truncated_dim != dim) {
                kacs_walk(values);
            }
        }
        if (truncated_dim != dim) {
            for (float& value : values) {
                value *= 0.25F;
            }
        }
    }

    void
    InverseTransform(std::vector<float>& values) const {
        codec_require(
            dim > 0 and values.size() == dim and flips.size() == K_ROUNDS * ((dim + 7) / 8),
            "inverse transform model or dimension mismatch");
        const uint64_t bytes = (dim + 7) / 8;
        const uint64_t truncated_dim = floor_power_of_two(dim);
        const float scale = 1.0F / std::sqrt(static_cast<float>(truncated_dim));
        if (truncated_dim != dim) {
            for (float& value : values) {
                value *= 4.0F;
            }
        }
        for (uint32_t remaining = K_ROUNDS; remaining > 0; --remaining) {
            const uint32_t round = remaining - 1;
            if (truncated_dim != dim) {
                const uint64_t half = dim / 2;
                const uint64_t offset = dim % 2 + half;
                for (uint64_t i = 0; i < half; ++i) {
                    const float left = values[i];
                    const float right = values[i + offset];
                    values[i] = (left + right) * 0.5F;
                    values[i + offset] = (left - right) * 0.5F;
                }
                if (dim % 2 != 0) {
                    values[half] /= std::sqrt(2.0F);
                }
            }
            float* block = round % 2 == 0 ? values.data() : values.data() + dim - truncated_dim;
            fht(block, truncated_dim);
            for (uint64_t d = 0; d < truncated_dim; ++d) {
                block[d] *= scale;
            }
            for (uint64_t d = 0; d < dim; ++d) {
                if ((flips[round * bytes + d / 8] & (1U << (d % 8))) != 0U) {
                    values[d] = -values[d];
                }
            }
        }
    }
};

}  // namespace vsag::lite::detail::rabitq
