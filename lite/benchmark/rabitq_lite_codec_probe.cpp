// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "simd/kernels/rabitq_pack.h"

namespace {
constexpr uint32_t K_TOTAL_BITS = 8;
constexpr uint32_t K_FILTER_BITS = 3;
constexpr uint32_t K_SUPPLEMENT_BITS = 5;
constexpr uint32_t K_ROUNDS = 4;
constexpr uint32_t K_ENCODE_ROUNDS = 6;

void
require(bool value, const char* message) {
    if (not value) {
        throw std::runtime_error(message);
    }
}

void
fht(std::vector<float>& values) {
    for (uint64_t step = 1; step < values.size(); step *= 2) {
        for (uint64_t block = 0; block < values.size(); block += step * 2) {
            for (uint64_t lane = 0; lane < step; ++lane) {
                const float left = values[block + lane];
                const float right = values[block + step + lane];
                values[block + lane] = left + right;
                values[block + step + lane] = left - right;
            }
        }
    }
}

struct Model {
    uint64_t dim{};
    std::vector<float> centroid;
    std::vector<uint8_t> flips;

    void
    Transform(std::vector<float>& values) const {
        const uint64_t bytes = (dim + 7) / 8;
        const float scale = 1.0F / std::sqrt(static_cast<float>(dim));
        for (uint32_t round = 0; round < K_ROUNDS; ++round) {
            for (uint64_t d = 0; d < dim; ++d) {
                if ((flips[round * bytes + d / 8] & (1U << (d % 8))) != 0U) {
                    values[d] = -values[d];
                }
            }
            fht(values);
            for (float& value : values) {
                value *= scale;
            }
        }
    }
};

Model
train(const std::vector<float>& base, uint64_t count, uint64_t dim, uint32_t seed) {
    require(count > 0 and dim > 0 and base.size() == count * dim, "invalid training matrix");
    require((dim & (dim - 1)) == 0, "probe FHT dimension must be a power of two");
    Model model{
        dim, std::vector<float>(dim, 0.0F), std::vector<uint8_t>(K_ROUNDS * ((dim + 7) / 8))};
    for (float value : base) {
        require(std::isfinite(value), "non-finite training value");
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
    model.Transform(model.centroid);
    return model;
}

std::vector<uint8_t>
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
    float norm{}, code_norm{}, error{};
};

std::vector<float>
normalize(const Model& model, const float* input, float& norm) {
    std::vector<float> values(input, input + model.dim);
    for (float value : values) {
        require(std::isfinite(value), "non-finite vector");
    }
    model.Transform(values);
    double squared = 0.0;
    for (uint64_t d = 0; d < model.dim; ++d) {
        values[d] -= model.centroid[d];
        squared += values[d] * values[d];
    }
    norm = squared < 1e-5 ? 1.0F : static_cast<float>(std::sqrt(squared));
    for (float& value : values) {
        value /= norm;
    }
    return values;
}

Encoded
encode(const Model& model, const float* input) {
    Encoded result;
    auto normalized = normalize(model, input, result.norm);
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
    double ip = 0.0;
    double sum = 0.0;
    for (uint64_t d = 0; d < model.dim; ++d) {
        ip += normalized[d] * static_cast<float>(result.scalar[d]);
        sum += normalized[d];
    }
    result.error = static_cast<float>((ip - 127.5 * sum) / result.code_norm);
    require(std::isfinite(result.error), "non-finite encoding metadata");
    return result;
}

void
self_test() {
    constexpr uint64_t dim = 128;
    constexpr uint64_t count = 64;
    std::mt19937 generator(91);
    std::normal_distribution<float> distribution;
    std::vector<float> base(count * dim);
    for (float& value : base) {
        value = distribution(generator);
    }
    const auto first = train(base, count, dim, 47);
    const auto second = train(base, count, dim, 47);
    require(first.centroid == second.centroid and first.flips == second.flips,
            "model is not deterministic");
    const auto encoded = encode(first, base.data());
    const auto repeated = encode(first, base.data());
    require(encoded.scalar == repeated.scalar and encoded.filter == repeated.filter and
                encoded.supplement == repeated.supplement,
            "encoding is not deterministic");
    require(encoded.filter.size() == 48 and encoded.supplement.size() == 80,
            "unexpected 128D layout");
    float direct = 0.0F;
    float split = 0.0F;
    auto query = normalize(first, base.data() + dim, direct);
    direct = 0.0F;
    for (uint64_t d = 0; d < dim; ++d) {
        direct += query[d] * static_cast<float>(encoded.scalar[d]);
    }
    const uint64_t plane_bytes = 16;
    for (uint64_t d = 0; d < dim; ++d) {
        const auto mask = static_cast<uint8_t>(1U << (d & 7U));
        const uint64_t byte = d >> 3U;
        uint32_t code = 0;
        for (uint32_t bit = 0; bit < K_FILTER_BITS; ++bit) {
            if ((encoded.filter[bit * plane_bytes + byte] & mask) != 0U) {
                code += 1U << (K_SUPPLEMENT_BITS + K_FILTER_BITS - bit - 1U);
            }
        }
        for (uint32_t bit = 0; bit < K_SUPPLEMENT_BITS; ++bit) {
            if ((encoded.supplement[bit * plane_bytes + byte] & mask) != 0U) {
                code += 1U << bit;
            }
        }
        split += query[d] * static_cast<float>(code);
    }
    require(std::fabs(direct - split) <= 1e-4F * std::max(1.0F, std::fabs(direct)),
            "codec split mismatch");
}
}  // namespace

int
main() {
    try {
        self_test();
        std::cout << "rabitq_lite_codec_probe: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
