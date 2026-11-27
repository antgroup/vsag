// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "lite/rabitq_constants.h"
#include "lite/rabitq_storage.h"
#include "lite/rabitq_transform.h"
#include "simd/kernels/rabitq_pack.h"

// RaBitQ encoding: fast_encode (coordinate-descent scalar-code optimisation),
// normalization, encode_normalized, and the public encode/prepare_encoding
// entry points.

namespace vsag::lite::detail::rabitq {

inline std::vector<uint8_t>
fast_encode(const std::vector<float>& normalized, float& code_norm) {
    constexpr uint32_t code_max = 255;
    constexpr double center = 127.5;
    double max_abs = 0.0;
    for (float value : normalized) {
        max_abs = std::max(max_abs, std::fabs(static_cast<double>(value)));
    }
    std::vector<uint8_t> codes(normalized.size(), 127);
    if (max_abs == 0.0) {
        code_norm = 1.0F;
        return codes;
    }
    const double inverse_delta = 128.0 / max_abs;
    double ip = 0.0;
    double norm_sqr = 0.0;
    for (uint64_t d = 0; d < normalized.size(); ++d) {
        auto code = static_cast<int64_t>(std::floor((normalized[d] + max_abs) * inverse_delta));
        code = std::clamp<int64_t>(code, 0, code_max);
        codes[d] = static_cast<uint8_t>(code);
        const double centered = static_cast<double>(code) - center;
        ip += normalized[d] * centered;
        norm_sqr += centered * centered;
    }
    const auto score = [](double inner, double norm) {
        return norm > 0.0 ? inner * inner / norm : 0.0;
    };
    for (uint32_t round = 0; round < K_ENCODE_ROUNDS; ++round) {
        bool changed = false;
        for (uint64_t d = 0; d < normalized.size(); ++d) {
            const int32_t current = codes[d];
            const double current_value = current - center;
            int32_t best = current;
            double best_ip = ip;
            double best_norm = norm_sqr;
            double best_score = score(ip, norm_sqr);
            for (int32_t direction : {-1, 1}) {
                const int32_t candidate = current + direction;
                if (candidate < 0 or candidate > static_cast<int32_t>(code_max)) {
                    continue;
                }
                const double next_ip =
                    ip + static_cast<double>(direction) * static_cast<double>(normalized[d]);
                if (next_ip < 0.0) {
                    continue;
                }
                const double next_norm = norm_sqr + 2.0 * current_value * direction + 1.0;
                const double next_score = score(next_ip, next_norm);
                if (next_score > best_score + 1e-8 * std::max(1.0, std::fabs(best_score))) {
                    best = candidate;
                    best_ip = next_ip;
                    best_norm = next_norm;
                    best_score = next_score;
                }
            }
            if (best != current) {
                codes[d] = static_cast<uint8_t>(best);
                ip = best_ip;
                norm_sqr = best_norm;
                changed = true;
            }
        }
        if (not changed) {
            break;
        }
    }
    norm_sqr = 0.0;
    for (uint8_t code : codes) {
        const double value = code - center;
        norm_sqr += value * value;
    }
    code_norm = static_cast<float>(std::sqrt(norm_sqr));
    if (not std::isfinite(code_norm) or code_norm <= 0.0F) {
        code_norm = 1.0F;
    }
    return codes;
}

inline std::vector<float>
normalize(const Model& model, const float* input, float& norm) {
    std::vector<float> values(input, input + model.dim);
    for (float value : values) {
        if (not std::isfinite(value)) {
            throw std::runtime_error("non-finite vector");
        }
    }
    model.Transform(values);
    double squared = 0.0;
    for (uint64_t d = 0; d < model.dim; ++d) {
        values[d] -= model.centroid[d];
        if (not std::isfinite(values[d])) {
            throw std::runtime_error("normalization transform overflow");
        }
        squared += static_cast<double>(values[d]) * values[d];
    }
    norm = static_cast<float>(std::sqrt(squared));
    if (not std::isfinite(norm) or norm < 0.0F) {
        throw std::runtime_error("normalization norm overflow");
    }
    const float divisor = norm == 0.0F ? 1.0F : norm;
    for (float& value : values) {
        value /= divisor;
    }
    return values;
}

inline Encoded
encode_normalized(const Model& model, const std::vector<float>& normalized, float norm) {
    if (normalized.size() != model.dim or not std::isfinite(norm) or norm < 0.0F) {
        throw std::runtime_error("invalid normalized encoding input");
    }
    Encoded result;
    result.norm = norm;
    result.scalar = fast_encode(normalized, result.code_norm);
    const uint64_t plane_bytes = (model.dim + 7) / 8;
    result.filter.assign(plane_bytes * K_FILTER_BITS, 0);
    result.supplement.assign(plane_bytes * K_SUPPLEMENT_BITS, 0);
    vsag::simd::RaBitQPackScalarToSplitPlanesTail(result.scalar.data(),
                                                  result.filter.data(),
                                                  result.supplement.data(),
                                                  model.dim,
                                                  K_TOTAL_BITS,
                                                  K_FILTER_BITS,
                                                  0);
    double full_ip = 0.0;
    double full_norm_sqr = 0.0;
    double filter_ip = 0.0;
    double filter_norm_sqr = 0.0;
    double query_sum = 0.0;
    for (uint64_t d = 0; d < model.dim; ++d) {
        const float query = normalized[d];
        const float code = result.scalar[d];
        const auto filter_code = static_cast<float>(result.scalar[d] >> K_SUPPLEMENT_BITS);
        full_ip += query * code;
        full_norm_sqr += (code - 127.5F) * (code - 127.5F);
        filter_ip += query * filter_code;
        filter_norm_sqr += (filter_code - 3.5F) * (filter_code - 3.5F);
        query_sum += query;
    }
    result.code_norm = static_cast<float>(std::sqrt(full_norm_sqr));
    result.filter_norm = static_cast<float>(std::sqrt(filter_norm_sqr));
    result.error = static_cast<float>((full_ip - 127.5 * query_sum) / result.code_norm);
    result.filter_error =
        std::fabs(static_cast<float>((filter_ip - 3.5 * query_sum) / result.filter_norm));
    result.filter_error = std::clamp(result.filter_error, 1e-5F, 1.0F);
    result.lower_bound_error =
        std::sqrt(std::max(0.0F, 1.0F - result.filter_error * result.filter_error) /
                  std::max(1.0F, static_cast<float>(model.dim - 1)));
    if (not std::isfinite(result.error) or not std::isfinite(result.filter_error) or
        not std::isfinite(result.lower_bound_error)) {
        throw std::runtime_error("non-finite encoding metadata");
    }
    return result;
}

struct PreparedEncoding {
    Encoded code;
    std::vector<float> query;
};

inline PreparedEncoding
prepare_encoding(const Model& model, const float* input) {
    if (input == nullptr) {
        throw std::runtime_error("null encoding input");
    }
    float norm = 0.0F;
    auto query = normalize(model, input, norm);
    auto code = encode_normalized(model, query, norm);
    return {std::move(code), std::move(query)};
}

inline Encoded
encode(const Model& model, const float* input) {
    return prepare_encoding(model, input).code;
}

}  // namespace vsag::lite::detail::rabitq
