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

#include <array>
#include <cstdint>

namespace vsag::sindi_dmq {

constexpr uint32_t kDirectDmqBits = 8;
constexpr uint32_t kDirectDmqCodebookSize = 1U << kDirectDmqBits;
constexpr uint32_t kDirectDmqThresholdCount = kDirectDmqCodebookSize - 1;

struct DirectDmqVectorFactors {
    float mean{0.0F};
    float alpha{0.0F};
};

struct DirectDmqCodebook {
    std::array<float, kDirectDmqThresholdCount> thresholds{};
    std::array<float, kDirectDmqCodebookSize> values{};
};

void
BuildDirectCodebook(float* values, uint32_t length, DirectDmqCodebook* codebook);

void
EncodeDirectValues(const float* values,
                   const DirectDmqCodebook* codebooks,
                   uint32_t length,
                   uint8_t* codes,
                   DirectDmqVectorFactors* factors);

[[nodiscard]] uint8_t
EncodeDirectResidual(float residual, const DirectDmqCodebook& codebook);

[[nodiscard]] float
DecodeDirectValue(const DirectDmqVectorFactors& factors,
                  const DirectDmqCodebook& codebook,
                  uint8_t code);

[[nodiscard]] float
ComputeExactInnerProduct(const uint32_t* base_ids,
                         const float* base_values,
                         uint32_t base_length,
                         const uint32_t* query_ids,
                         const float* query_values,
                         uint32_t query_length);

[[nodiscard]] float
ComputeDirectApproximateInnerProduct(const uint32_t* base_ids,
                                     const uint8_t* codes,
                                     const DirectDmqCodebook* codebooks,
                                     uint32_t base_length,
                                     const uint32_t* query_ids,
                                     const float* query_values,
                                     uint32_t query_length,
                                     const DirectDmqVectorFactors& factors);

}  // namespace vsag::sindi_dmq