// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "simd/kernels/rabitq_pack.h"

namespace {

constexpr uint32_t K_TOTAL_BITS = 8;
constexpr uint32_t K_FILTER_BITS = 3;
constexpr uint32_t K_SUPPLEMENT_BITS = K_TOTAL_BITS - K_FILTER_BITS;

float
three_bit_centered_ip(const float* query, const uint8_t* planes, uint64_t dim) {
    const uint64_t plane_bytes = (dim + 7) / 8;
    float result = 0.0F;
    for (uint64_t d = 0; d < dim; ++d) {
        const auto mask = static_cast<uint8_t>(1U << (d & 7U));
        const uint64_t byte = d >> 3U;
        uint32_t code = 0;
        for (uint32_t bit = 0; bit < K_FILTER_BITS; ++bit) {
            code += ((planes[bit * plane_bytes + byte] & mask) != 0U)
                        ? (1U << (K_FILTER_BITS - bit - 1U))
                        : 0U;
        }
        result += query[d] * (static_cast<float>(code) - 3.5F);
    }
    return result;
}

float
supplement_ip(const float* query, const uint8_t* planes, uint64_t dim) {
    const uint64_t plane_bytes = (dim + 7) / 8;
    float result = 0.0F;
    for (uint64_t d = 0; d < dim; ++d) {
        const auto mask = static_cast<uint8_t>(1U << (d & 7U));
        const uint64_t byte = d >> 3U;
        uint32_t code = 0;
        for (uint32_t bit = 0; bit < K_SUPPLEMENT_BITS; ++bit) {
            code += ((planes[bit * plane_bytes + byte] & mask) != 0U) ? (1U << bit) : 0U;
        }
        result += query[d] * static_cast<float>(code);
    }
    return result;
}

void
require(bool condition, const char* message) {
    if (not condition) {
        throw std::runtime_error(message);
    }
}

void
check_dimension(uint64_t dim) {
    const uint64_t plane_bytes = (dim + 7) / 8;
    std::mt19937 generator(static_cast<uint32_t>(47 + dim));
    std::uniform_int_distribution<uint32_t> code_distribution(0, 255);
    std::uniform_real_distribution<float> query_distribution(-1.0F, 1.0F);

    std::vector<uint8_t> scalar(dim);
    std::vector<float> query(dim);
    float query_sum = 0.0F;
    float expected = 0.0F;
    for (uint64_t i = 0; i < dim; ++i) {
        scalar[i] = static_cast<uint8_t>(code_distribution(generator));
        query[i] = query_distribution(generator);
        query_sum += query[i];
        expected += query[i] * static_cast<float>(scalar[i]);
    }

    std::vector<uint8_t> filter(plane_bytes * K_FILTER_BITS, 0);
    std::vector<uint8_t> supplement(plane_bytes * K_SUPPLEMENT_BITS, 0);
    vsag::simd::RaBitQPackScalarToSplitPlanesTail(
        scalar.data(), filter.data(), supplement.data(), dim, K_TOTAL_BITS, K_FILTER_BITS, 0);

    const float filter_centered = three_bit_centered_ip(query.data(), filter.data(), dim);
    const float filter_ip = filter_centered + 3.5F * query_sum;
    const float residual_ip = supplement_ip(query.data(), supplement.data(), dim);
    const float actual = 32.0F * filter_ip + residual_ip;
    const float tolerance = 1e-4F * std::max(1.0F, std::fabs(expected));
    require(std::fabs(actual - expected) <= tolerance, "split inner product mismatch");

    for (uint64_t i = dim; i < plane_bytes * 8; ++i) {
        const auto mask = static_cast<uint8_t>(1U << (i & 7U));
        const uint64_t byte = i >> 3U;
        for (uint32_t bit = 0; bit < K_FILTER_BITS; ++bit) {
            require((filter[bit * plane_bytes + byte] & mask) == 0, "filter tail bit is not zero");
        }
        for (uint32_t bit = 0; bit < K_SUPPLEMENT_BITS; ++bit) {
            require((supplement[bit * plane_bytes + byte] & mask) == 0,
                    "supplement tail bit is not zero");
        }
    }
}

}  // namespace

int
main() {
    try {
        for (uint64_t dim : {1U, 7U, 8U, 9U, 31U, 128U, 129U}) {
            check_dimension(dim);
        }
        std::cout << "rabitq_lite_layout_probe: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
