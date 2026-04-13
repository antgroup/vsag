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

/// @file runtime_parameter.h
/// @brief Runtime parameter for index optimization and configuration.

#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <variant>
#include <vector>

namespace vsag {

/// @brief Runtime parameter with min/max range and step for iteration.
///
/// This struct represents a tunable parameter that can be iterated through
/// a range of values, used primarily for index optimization and parameter search.
struct RuntimeParameter {
public:
    /// @brief Constructs a runtime parameter with the specified range.
    /// @param name Parameter name.
    /// @param min Minimum value.
    /// @param max Maximum value.
    /// @param step Step size for iteration (default 0 means auto-calculated as 1/10 of range).
    RuntimeParameter(const std::string& name, float min, float max, float step = 0)
        : name_(name), min_(min), cur_(min), max_(max), step_(step) {
        is_end_ = false;
        if (std::abs(step_) <= 1e-5) {
            step_ = (max_ - min_) / 10.0;
        }
    }

    /// @brief Gets the next value in the iteration sequence.
    /// @return Current value before advancing to next.
    float
    Next() {
        if (is_end_) {
            Reset();
        }
        float prev = cur_;
        cur_ += step_;
        if (cur_ > max_) {
            cur_ = min_;
            is_end_ = true;
        }
        return prev;
    }

    /// @brief Gets the current value without advancing.
    /// @return Current parameter value.
    float
    Cur() const {
        return cur_;
    }

    /// @brief Resets the parameter to its minimum value.
    void
    Reset() {
        cur_ = min_;
        is_end_ = false;
    }

    /// @brief Checks if the iteration has completed a full cycle.
    /// @return True if the iteration has reached the end.
    bool
    IsEnd() const {
        return is_end_;
    }

public:
    /// Parameter name.
    std::string name_;

private:
    /// Minimum value of the parameter range.
    float min_{0};
    /// Maximum value of the parameter range.
    float max_{0};
    /// Step size for iteration.
    float step_{0};
    /// Current value in the iteration.
    float cur_{0};
    /// Flag indicating if iteration has completed a full cycle.
    bool is_end_{false};
};

}  // namespace vsag