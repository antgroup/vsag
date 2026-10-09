// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "simd/kernels/rabitq_pack.h"

// Internal 3+5 L2 codec extracted from the measured Lite probe. Layout and math
// follow quantization/rabitq_quantization/rabitq_quantizer.cpp and the official
// FHT/Kac transformer; no Full allocator, IO or graph dependencies are imported.
// This module does not expose a public RaBitQ backend or a snapshot format.
namespace vsag::lite::detail::rabitq {

constexpr uint32_t K_TOTAL_BITS = 8;
constexpr uint32_t K_FILTER_BITS = 3;
constexpr uint32_t K_SUPPLEMENT_BITS = 5;
constexpr uint32_t K_ROUNDS = 4;
constexpr uint32_t K_ENCODE_ROUNDS = 6;
constexpr float K_ERROR_RATE = 1.9F;

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

    // Reverse the exact sequence used by Transform, including the truncated
    // dimension Kac steps. The caller owns the buffer; no mutable model scratch.
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

inline Model
train(const std::vector<float>& base, uint64_t count, uint64_t dim, uint32_t seed) {
    codec_require(count > 0 and dim > 0 and base.size() == count * dim, "invalid training matrix");
    Model model{
        dim, std::vector<float>(dim, 0.0F), std::vector<uint8_t>(K_ROUNDS * ((dim + 7) / 8))};
    for (float value : base) {
        codec_require(std::isfinite(value), "non-finite training value");
    }
    for (uint64_t i = 0; i < count; ++i) {
        for (uint64_t d = 0; d < dim; ++d) {
            model.centroid[d] += base[i * dim + d];
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
        codec_require(std::isfinite(value), "training centroid overflow");
    }
    model.Transform(model.centroid);
    for (float value : model.centroid) {
        codec_require(std::isfinite(value), "training transform overflow");
    }
    return model;
}

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

struct Encoded {
    std::vector<uint8_t> filter, supplement, scalar;
    float norm{}, code_norm{}, error{}, filter_norm{}, filter_error{}, lower_bound_error{};
};

struct EncodedMetadata {
    float norm{}, code_norm{}, error{}, filter_norm{}, filter_error{}, lower_bound_error{};
};

struct EncodedView {
    const uint8_t* filter;
    const uint8_t* supplement;
    EncodedMetadata metadata;
};

struct EncodedRecords {
    explicit EncodedRecords(uint64_t input_dim) : dim(input_dim) {
    }

    [[nodiscard]] uint64_t
    FilterBytes() const {
        return ((dim + 7) / 8) * K_FILTER_BITS;
    }

    [[nodiscard]] uint64_t
    SupplementBytes() const {
        return ((dim + 7) / 8) * K_SUPPLEMENT_BITS;
    }

    [[nodiscard]] uint64_t
    Size() const {
        return metadata.size();
    }

    void
    Reserve(uint64_t count) {
        metadata.reserve(count);
        filters.reserve(count * FilterBytes());
        supplements.reserve(count * SupplementBytes());
    }

    void
    Resize(uint64_t count) {
        metadata.resize(count);
        filters.resize(count * FilterBytes());
        supplements.resize(count * SupplementBytes());
    }

    void
    Append(Encoded code) {
        codec_require(
            code.filter.size() == FilterBytes() and code.supplement.size() == SupplementBytes(),
            "invalid encoded record");
        metadata.push_back({code.norm,
                            code.code_norm,
                            code.error,
                            code.filter_norm,
                            code.filter_error,
                            code.lower_bound_error});
        filters.insert(filters.end(), code.filter.begin(), code.filter.end());
        supplements.insert(supplements.end(), code.supplement.begin(), code.supplement.end());
    }

    void
    Replace(uint64_t slot, Encoded code) {
        codec_require(slot < Size() and code.filter.size() == FilterBytes() and
                          code.supplement.size() == SupplementBytes(),
                      "invalid encoded replacement");
        metadata[slot] = {code.norm,
                          code.code_norm,
                          code.error,
                          code.filter_norm,
                          code.filter_error,
                          code.lower_bound_error};
        std::copy(code.filter.begin(), code.filter.end(), filters.data() + slot * FilterBytes());
        std::copy(code.supplement.begin(),
                  code.supplement.end(),
                  supplements.data() + slot * SupplementBytes());
    }

    void
    RemoveSwap(uint64_t slot) {
        codec_require(slot < Size(), "encoded removal outside storage");
        const uint64_t last = Size() - 1;
        if (slot != last) {
            metadata[slot] = metadata[last];
            std::copy_n(filters.data() + last * FilterBytes(),
                        FilterBytes(),
                        filters.data() + slot * FilterBytes());
            std::copy_n(supplements.data() + last * SupplementBytes(),
                        SupplementBytes(),
                        supplements.data() + slot * SupplementBytes());
        }
        metadata.pop_back();
        filters.resize(last * FilterBytes());
        supplements.resize(last * SupplementBytes());
    }

    [[nodiscard]] EncodedView
    At(uint64_t id) const {
        codec_require(id < Size(), "encoded record outside storage");
        return {filters.data() + id * FilterBytes(),
                supplements.data() + id * SupplementBytes(),
                metadata[id]};
    }

    uint64_t dim;
    std::vector<EncodedMetadata> metadata;
    std::vector<uint8_t> filters;
    std::vector<uint8_t> supplements;
};

inline std::vector<float>
normalize(const Model& model, const float* input, float& norm) {
    std::vector<float> values(input, input + model.dim);
    for (float value : values) {
        codec_require(std::isfinite(value), "non-finite vector");
    }
    model.Transform(values);
    double squared = 0.0;
    for (uint64_t d = 0; d < model.dim; ++d) {
        values[d] -= model.centroid[d];
        codec_require(std::isfinite(values[d]), "normalization transform overflow");
        const float term = values[d] * values[d];
        squared += std::isfinite(term) ? static_cast<double>(term)
                                       : static_cast<double>(values[d]) * values[d];
    }
    norm = squared < 1e-5 ? 1.0F : static_cast<float>(std::sqrt(squared));
    codec_require(std::isfinite(norm) and norm > 0.0F, "normalization norm overflow");
    for (float& value : values) {
        value /= norm;
    }
    return values;
}

inline Encoded
encode_normalized(const Model& model, const std::vector<float>& normalized, float norm) {
    codec_require(normalized.size() == model.dim and std::isfinite(norm) and norm > 0.0F,
                  "invalid normalized encoding input");
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
    codec_require(std::isfinite(result.error) and std::isfinite(result.filter_error) and
                      std::isfinite(result.lower_bound_error),
                  "non-finite encoding metadata");
    return result;
}

struct PreparedEncoding {
    Encoded code;
    std::vector<float> query;
};

// Prepare once against the fixed model, retaining query scratch for mutation.
inline PreparedEncoding
prepare_encoding(const Model& model, const float* input) {
    codec_require(input != nullptr, "null encoding input");
    float norm = 0.0F;
    auto query = normalize(model, input, norm);
    auto code = encode_normalized(model, query, norm);
    return {std::move(code), std::move(query)};
}

inline Encoded
encode(const Model& model, const float* input) {
    return prepare_encoding(model, input).code;
}

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
    codec_require(model.dim > 0 and model.centroid.size() == model.dim and
                      code.filter != nullptr and code.supplement != nullptr and
                      std::isfinite(code.metadata.norm) and code.metadata.norm > 0.0F and
                      std::isfinite(code.metadata.code_norm) and code.metadata.code_norm > 0.0F,
                  "invalid decoding model or metadata");
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
        codec_require(std::isfinite(value), "non-finite decoded vector");
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
