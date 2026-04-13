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

/**
 * @file linear_congruential_generator.h
 * @brief Linear Congruential Generator (LCG) for pseudo-random number generation.
 */

#pragma once

#include <cstdint>

namespace vsag {
/**
 * @brief Simple Linear Congruential Generator for pseudo-random number generation.
 *
 * Implements the LCG formula: X(n+1) = (A * X(n) + C) mod M
 * with parameters A=1664525, C=1013904223, M=2^32-1.
 * This is a fast but not cryptographically secure random number generator.
 */
class LinearCongruentialGenerator {
public:
    /**
     * @brief Construct a new Linear Congruential Generator with default seed.
     */
    LinearCongruentialGenerator();

    /**
     * @brief Generate the next random floating-point number.
     *
     * @return float A random float in the range [0.0, 1.0).
     */
    float
    NextFloat();

private:
    unsigned int current_;                     ///< Current state value
    static constexpr uint32_t A = 1664525;     ///< Multiplier constant
    static constexpr uint32_t C = 1013904223;  ///< Increment constant
    static constexpr uint32_t M = 4294967295;  ///< Modulus (2^32 - 1)
};
}  // namespace vsag