// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "lite/rabitq_constants.h"
#include "lite/rabitq_transform.h"

// RaBitQ model training: centroid + random sign flips.

namespace vsag::lite::detail::rabitq {

template <typename RowAt>
inline Model
train_rows(uint64_t count, uint64_t dim, uint32_t seed, RowAt row_at) {
    if (count == 0 or dim == 0) {
        throw std::runtime_error("invalid training dimensions");
    }
    Model model{
        dim, std::vector<float>(dim, 0.0F), std::vector<uint8_t>(K_ROUNDS * ((dim + 7) / 8))};
    for (uint64_t i = 0; i < count; ++i) {
        const float* row = row_at(i);
        for (uint64_t d = 0; d < dim; ++d) {
            if (not std::isfinite(row[d])) {
                throw std::runtime_error("non-finite training value");
            }
        }
        for (uint64_t d = 0; d < dim; ++d) {
            model.centroid[d] += row[d];
        }
    }
    for (float& value : model.centroid) {
        value /= static_cast<float>(count);
    }
    std::mt19937 generator(seed);
    std::uniform_int_distribution<uint32_t> bytes(0, 255);
    for (uint8_t& value : model.flips) {
        value = static_cast<uint8_t>(bytes(generator));
    }
    for (float value : model.centroid) {
        if (not std::isfinite(value)) {
            throw std::runtime_error("training centroid overflow");
        }
    }
    model.Transform(model.centroid);
    for (float value : model.centroid) {
        if (not std::isfinite(value)) {
            throw std::runtime_error("training transform overflow");
        }
    }
    return model;
}

inline Model
train(const std::vector<float>& base, uint64_t count, uint64_t dim, uint32_t seed) {
    if (count == 0 or dim == 0 or count > base.max_size() / dim or base.size() != count * dim) {
        throw std::runtime_error("invalid training matrix");
    }
    return train_rows(count, dim, seed, [&](uint64_t row) { return base.data() + row * dim; });
}

}  // namespace vsag::lite::detail::rabitq
