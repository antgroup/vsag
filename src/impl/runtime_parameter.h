
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
using ParamValue = std::variant<int, float, std::string>;

struct RuntimeParameter {
public:
    RuntimeParameter(const std::string& name) : name_(name) {
    }
    virtual ~RuntimeParameter() = default;

    virtual ParamValue
    Next() = 0;

    virtual ParamValue
    Cur() const = 0;

    virtual void
    Reset() = 0;

    virtual bool
    IsEnd() const = 0;

public:
    std::string name_;
};

struct IntRuntimeParameter : RuntimeParameter {
public:
    IntRuntimeParameter(const std::string& name, int min, int max, int step = -1.0)
        : RuntimeParameter(name), min_(min), cur_(min), max_(max) {
        is_end_ = false;
        if (step < 0) {
            step_ = (max_ - min_) / 10.0;
        }
        if (step_ == 0) {
            step_ = 1;
        }
    }

    ParamValue
    Next() override {
        cur_ += step_;
        is_end_ = (cur_ > max_);
        if (cur_ > max_) {
            cur_ -= (max_ - min_);
        }
        return cur_;
    }

    ParamValue
    Cur() const override {
        return cur_;
    }

    void
    Reset() override {
        cur_ = min_;
        is_end_ = (cur_ > max_);
    }

    bool
    IsEnd() const override {
        return is_end_;
    }

private:
    int min_{0};
    int max_{0};
    int step_{0};
    int cur_{0};
    bool is_end_{false};
};
}  // namespace vsag
