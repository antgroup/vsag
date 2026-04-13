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

namespace vsag {

/**
 * @file number.h
 * @brief Numeric wrapper with range checking utilities.
 */

/**
 * @struct Number
 * @brief Template wrapper for numeric types with range checking.
 *
 * This struct wraps a numeric value and provides utility methods
 * for range checking operations.
 *
 * @tparam T The underlying numeric type.
 */
template <typename T>
struct Number {
    /**
     * @brief Constructs a Number with the given value.
     * @param n The numeric value to wrap.
     */
    explicit Number(T n) : num(n) {
    }

    /**
     * @brief Checks if the number is within a range [lower, upper].
     * @param lower Lower bound (inclusive).
     * @param upper Upper bound (inclusive).
     * @return True if lower <= num <= upper, false otherwise.
     *
     * @note Uses unsigned arithmetic trick for efficiency.
     */
    bool
    in_range(T lower, T upper) {
        return ((unsigned)(num - lower) <= (upper - lower));
    }

    /// The wrapped numeric value
    T num;
};
}  // namespace vsag