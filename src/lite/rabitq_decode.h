// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "lite/rabitq_constants.h"
#include "lite/rabitq_storage.h"
#include "lite/rabitq_transform.h"

// RaBitQ decoding: bit-plane reconstruction, decode to approximate original
// space, and scalar filter inner-product computation.

namespace vsag::lite::detail::rabitq {

inline uint32_t
read_plane_code(const uint8_t* planes,
                uint64_t plane_bytes,
                uint64_t d,
                uint32_t bits,
                bool most_significant_first) {
    const auto mask = static_cast<uint8_t>(1U << (d & 7U));
    const uint64_t byte = d >> 3U;
    uint32_t code = 0;
    for (uint32_t bit = 0; bit < bits; ++bit) {
        if ((planes[bit * plane_bytes + byte] & mask) != 0U) {
            code += most_significant_first ? 1U << (bits - bit - 1U) : 1U << bit;
        }
    }
    return code;
}

// Reconstruct an approximate original-space vector, not the original FP32
// input or the quantizer's distance estimate. Persistence must retain the model
// and encoded planes directly instead of decoding and re-encoding this value.
inline const float*
decode(const Model& model, const EncodedView& code, std::vector<float>& scratch) {
    if (model.dim == 0 or model.centroid.size() != model.dim or code.filter == nullptr or
        code.supplement == nullptr or not std::isfinite(code.metadata.norm) or
        code.metadata.norm < 0.0F or not std::isfinite(code.metadata.code_norm) or
        code.metadata.code_norm <= 0.0F) {
        throw std::runtime_error("invalid decoding model or metadata");
    }
    scratch.resize(model.dim);
    const uint64_t plane_bytes = (model.dim + 7) / 8;
    for (uint64_t d = 0; d < model.dim; ++d) {
        const uint32_t high = read_plane_code(code.filter, plane_bytes, d, K_FILTER_BITS, true);
        const uint32_t low =
            read_plane_code(code.supplement, plane_bytes, d, K_SUPPLEMENT_BITS, false);
        const float centered = static_cast<float>((high << K_SUPPLEMENT_BITS) | low) - 127.5F;
        scratch[d] = centered * code.metadata.norm / code.metadata.code_norm + model.centroid[d];
    }
    model.InverseTransform(scratch);
    for (float value : scratch) {
        if (not std::isfinite(value)) {
            throw std::runtime_error("non-finite decoded vector");
        }
    }
    return scratch.data();
}

inline float
scalar_filter_centered_ip(const std::vector<float>& query, const uint8_t* filter) {
    const uint64_t plane_bytes = (query.size() + 7) / 8;
    float result = 0.0F;
    for (uint64_t d = 0; d < query.size(); ++d) {
        const auto code = read_plane_code(filter, plane_bytes, d, K_FILTER_BITS, true);
        result += query[d] * (static_cast<float>(code) - 3.5F);
    }
    return result;
}

}  // namespace vsag::lite::detail::rabitq
