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
 * @file types.h
 * @brief Common types for test fixtures.
 */

#pragma once

#include <cstdint>
#include <iostream>

namespace fixtures {

/**
 * @struct comparable_float_t
 * @brief Float wrapper that compares values with tolerance for relative error.
 * Useful for comparing distances, recall values, and timing measurements in tests.
 */
struct comparable_float_t {
    /**
     * @brief Constructs a comparable_float_t from a float value.
     * @param val The float value to wrap.
     */
    comparable_float_t(float val);

    /**
     * @brief Compares two comparable_float_t values with tolerance.
     * @param d The other value to compare against.
     * @return True if values are within epsilon tolerance.
     */
    bool
    operator==(const comparable_float_t& d) const;

    /**
     * @brief Outputs the value to a stream.
     * @param os The output stream.
     * @param obj The comparable_float_t to output.
     * @return Reference to the output stream.
     */
    friend std::ostream&
    operator<<(std::ostream& os, const comparable_float_t& obj);

    float value;                  // The wrapped float value.
    const double epsilon = 2e-6;  // Tolerance for comparison.
};

using dist_t = comparable_float_t;
using time_t = comparable_float_t;
using recall_t = comparable_float_t;

/**
 * @struct IOItem
 * @brief Represents a single IO operation for testing IO behavior.
 */
struct IOItem {
    uint64_t start_;   // Starting offset of the IO operation.
    uint64_t length_;  // Length of data to read/write.
    uint8_t* data_;    // Pointer to the data buffer.

    /**
     * @brief Destructor that frees the data buffer.
     */
    ~IOItem();
};

}  // namespace fixtures
