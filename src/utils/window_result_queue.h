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

#include <string>
#include <vector>

namespace vsag {

/**
 * @file window_result_queue.h
 * @brief Sliding window queue for computing average results.
 */

/**
 * @class WindowResultQueue
 * @brief Fixed-size queue for computing sliding window averages.
 *
 * This class maintains a window of float values and provides
 * average computation over the window. Useful for smoothing
 * performance metrics or result quality measurements.
 */
class WindowResultQueue {
public:
    /**
     * @brief Constructs an empty window result queue.
     */
    WindowResultQueue();

    /**
     * @brief Pushes a new value into the queue.
     * @param value Value to add to the queue.
     */
    void
    Push(float value);

    /**
     * @brief Computes the average of all values in the queue.
     * @return Average value, or 0 if queue is empty.
     */
    [[nodiscard]] float
    GetAvgResult() const;

private:
    /// Number of elements in the queue
    uint64_t count_ = 0;

    /// Storage for queue values
    std::vector<float> queue_;
};
}  // namespace vsag