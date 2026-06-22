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

#include "sindi_sparse_dmq.h"

#include <algorithm>
#include <cmath>

namespace vsag::sindi_dmq {
namespace {

constexpr float K_MIN_DENOMINATOR = 1e-12F;

[[nodiscard]] float
safe_rescale(float numerator, float denominator) {
    if (std::abs(denominator) <= K_MIN_DENOMINATOR) {
        return 0.0F;
    }
    return numerator / denominator;
}

}  // namespace

void
BuildDirectCodebook(float* values, uint32_t length, DirectDmqCodebook* codebook) {
    codebook->thresholds.fill(0.0F);
    codebook->values.fill(0.0F);
    if (length == 0) {
        return;
    }

    std::sort(values, values + length);

    double value_sum = 0.0;
    double square_sum = 0.0;
    for (uint32_t value_index = 0; value_index < length; ++value_index) {
        value_sum += values[value_index];
        square_sum += static_cast<double>(values[value_index]) * values[value_index];
    }

    double total_weight = 0.0;
    for (uint32_t value_index = 0; value_index < length; ++value_index) {
        double value = values[value_index];
        total_weight += value * value * length + square_sum - 2.0 * value * value_sum;
    }
    if (total_weight <= K_MIN_DENOMINATOR) {
        codebook->thresholds.fill(values[0]);
        codebook->values.fill(values[0]);
        return;
    }

    uint32_t partition = 1;
    double current_weight = 0.0;
    for (uint32_t value_index = 0; value_index < length && partition < 512; ++value_index) {
        double value = values[value_index];
        double weight = value * value * length + square_sum - 2.0 * value * value_sum;
        current_weight += weight;
        while (current_weight * 512.0 + 1e-7 >= total_weight * partition) {
            if (((partition - 1) & 1U) != 0U) {
                codebook->thresholds[(partition - 1) / 2] = values[value_index];
            } else {
                codebook->values[(partition - 1) / 2] = values[value_index];
            }
            ++partition;
            if (partition == 512) {
                break;
            }
        }
    }

    for (; partition < 512; ++partition) {
        if (((partition - 1) & 1U) != 0U) {
            codebook->thresholds[(partition - 1) / 2] = values[length - 1];
        } else {
            codebook->values[(partition - 1) / 2] = values[length - 1];
        }
    }
}

uint8_t
EncodeDirectResidual(float residual, const DirectDmqCodebook& codebook) {
    return static_cast<uint8_t>(
        std::lower_bound(codebook.thresholds.begin(), codebook.thresholds.end(), residual) -
        codebook.thresholds.begin());
}

float
DecodeDirectValue(const DirectDmqVectorFactors& factors,
                  const DirectDmqCodebook& codebook,
                  uint8_t code) {
    return factors.mean + factors.alpha * codebook.values[code];
}

void
EncodeDirectValues(const float* values,
                   const DirectDmqCodebook* codebooks,
                   uint32_t length,
                   uint8_t* codes,
                   DirectDmqVectorFactors* factors) {
    *factors = DirectDmqVectorFactors{};
    if (length == 0) {
        return;
    }

    double value_sum = 0.0;
    for (uint32_t value_index = 0; value_index < length; ++value_index) {
        value_sum += values[value_index];
    }
    factors->mean = static_cast<float>(value_sum / length);

    double numerator = 0.0;
    double denominator = 0.0;
    for (uint32_t value_index = 0; value_index < length; ++value_index) {
        float residual = values[value_index] - factors->mean;
        uint8_t code = EncodeDirectResidual(residual, codebooks[value_index]);
        codes[value_index] = code;
        float qualifier = codebooks[value_index].values[code];
        numerator += static_cast<double>(residual) * residual;
        denominator += static_cast<double>(qualifier) * residual;
    }
    factors->alpha = safe_rescale(static_cast<float>(numerator), static_cast<float>(denominator));
}

float
ComputeExactInnerProduct(const uint32_t* base_ids,
                         const float* base_values,
                         uint32_t base_length,
                         const uint32_t* query_ids,
                         const float* query_values,
                         uint32_t query_length) {
    float inner_product = 0.0F;
    uint32_t query_offset = 0;
    uint32_t base_offset = 0;
    while (query_offset < query_length && base_offset < base_length) {
        if (query_ids[query_offset] < base_ids[base_offset]) {
            ++query_offset;
        } else if (query_ids[query_offset] > base_ids[base_offset]) {
            ++base_offset;
        } else {
            inner_product += query_values[query_offset] * base_values[base_offset];
            ++query_offset;
            ++base_offset;
        }
    }
    return inner_product;
}

float
ComputeDirectApproximateInnerProduct(const uint32_t* base_ids,
                                     const uint8_t* codes,
                                     const DirectDmqCodebook* codebooks,
                                     uint32_t base_length,
                                     const uint32_t* query_ids,
                                     const float* query_values,
                                     uint32_t query_length,
                                     const DirectDmqVectorFactors& factors) {
    float inner_product = 0.0F;
    uint32_t query_offset = 0;
    uint32_t base_offset = 0;
    while (query_offset < query_length && base_offset < base_length) {
        if (query_ids[query_offset] < base_ids[base_offset]) {
            ++query_offset;
        } else if (query_ids[query_offset] > base_ids[base_offset]) {
            ++base_offset;
        } else {
            inner_product += query_values[query_offset] *
                             DecodeDirectValue(factors, codebooks[base_offset], codes[base_offset]);
            ++query_offset;
            ++base_offset;
        }
    }
    return inner_product;
}

}  // namespace vsag::sindi_dmq
