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
#include <chrono>

namespace vsag {

/**
 * @file timer.h
 * @brief Simple timer utility for measuring elapsed time.
 */

/**
 * @class Timer
 * @brief RAII-style timer for measuring elapsed time.
 *
 * This class provides a convenient way to measure elapsed time with automatic
 * recording on destruction. It supports both reference and pointer-based output.
 */
class Timer {
public:
    /**
     * @brief Constructs a timer that records elapsed time to a reference.
     * @param ref Reference to store elapsed time in seconds.
     */
    explicit Timer(double& ref);

    /**
     * @brief Constructs a timer that records elapsed time to a pointer.
     * @param ref Pointer to store elapsed time in seconds.
     */
    explicit Timer(double* ref);

    /**
     * @brief Constructs a timer without automatic recording.
     *
     * Use Record() to manually retrieve elapsed time.
     */
    explicit Timer();

    /**
     * @brief Destructor that records elapsed time if a reference was provided.
     */
    ~Timer();

    /**
     * @brief Records and returns the elapsed time.
     * @return Elapsed time in seconds since construction.
     */
    double
    Record();

    /**
     * @brief Sets the overtime threshold for CheckOvertime().
     * @param threshold Threshold value in seconds.
     */
    void
    SetThreshold(double threshold);

    /**
     * @brief Checks if elapsed time exceeds the threshold.
     * @return True if elapsed time exceeds the threshold, false otherwise.
     */
    bool
    CheckOvertime();

private:
    /// Pointer to store elapsed time (may be null)
    double* ref_{nullptr};

    /// Threshold for overtime check in seconds
    double threshold_{std::numeric_limits<double>::max()};

    /// Start time point
    std::chrono::steady_clock::time_point start;
};
}  // namespace vsag