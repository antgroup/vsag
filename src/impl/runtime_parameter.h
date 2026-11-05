
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

struct RuntimeParameter {
public:
    RuntimeParameter(const std::string& name, float min, float max, float step = 0)
        : name_(name), min_(min), cur_(min), max_(max), step_(step) {
        is_end_ = false;
        if (std::abs(step_) <= 1e-5) {
            step_ = (max_ - min_) / 10.0;
        }
    }

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

    float
    Cur() const {
        return cur_;
    }

    void
    Reset() {
        cur_ = min_;
        is_end_ = false;
    }

    bool
    IsEnd() const {
        return is_end_;
    }

public:
    std::string name_;

private:
    float min_{0};
    float max_{0};
    float step_{0};
    float cur_{0};
    bool is_end_{false};
};

}  // namespace vsag
