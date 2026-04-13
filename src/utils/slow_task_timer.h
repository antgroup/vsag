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
#include <string>

namespace vsag {

/**
 * @file slow_task_timer.h
 * @brief Timer for logging slow tasks.
 */

/**
 * @class SlowTaskTimer
 * @brief RAII-style timer for logging slow task execution.
 *
 * This class measures task execution time and logs a warning if the
 * duration exceeds the specified threshold. Useful for performance
 * monitoring and identifying bottlenecks.
 */
class SlowTaskTimer {
public:
    /**
     * @brief Constructs a slow task timer with optional threshold.
     * @param name Task name for logging identification.
     * @param log_threshold_ms Threshold in milliseconds for slow task warning.
     *                         Default is 0 (always log).
     */
    explicit SlowTaskTimer(std::string name, int64_t log_threshold_ms = 0);

    /**
     * @brief Destructor that checks and logs if task was slow.
     */
    ~SlowTaskTimer();

public:
    /// Task name for logging
    std::string name;

    /// Threshold in milliseconds for slow task warning
    int64_t threshold;

    /// Start time point
    std::chrono::steady_clock::time_point start;
};

}  // namespace vsag