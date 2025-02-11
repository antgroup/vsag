
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
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "allocator_wrapper.h"
#include "index/index_common_param.h"
#include "logger.h"
#include "spdlog/spdlog.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"

namespace vsag {

struct SlowTaskTimer {
    explicit SlowTaskTimer(std::string name, int64_t log_threshold_ms = 0);
    ~SlowTaskTimer();

    std::string name;
    int64_t threshold;
    std::chrono::steady_clock::time_point start;
};

struct Timer {
    explicit Timer(double& ref);
    ~Timer();

    double& ref_;
    std::chrono::steady_clock::time_point start;
};

class WindowResultQueue {
public:
    WindowResultQueue();

    void
    Push(float value);

    [[nodiscard]] float
    GetAvgResult() const;

private:
    size_t count_ = 0;
    std::vector<float> queue_;
};

template <typename T>
struct Number {
    explicit Number(T n) : num(n) {
    }

    bool
    in_range(T lower, T upper) {
        return ((unsigned)(num - lower) <= (upper - lower));
    }

    T num;
};

template <typename IndexOpParameters>
tl::expected<IndexOpParameters, Error>
try_parse_parameters(const std::string& json_string) {
    try {
        return IndexOpParameters::FromJson(json_string);
    } catch (const std::exception& e) {
        return tl::unexpected<Error>(ErrorType::INVALID_ARGUMENT, e.what());
    }
}

template <typename IndexOpParameters>
tl::expected<IndexOpParameters, Error>
try_parse_parameters(JsonType& param_obj, IndexCommonParam index_common_param) {
    try {
        return IndexOpParameters::FromJson(param_obj, index_common_param);
    } catch (const std::exception& e) {
        return tl::unexpected<Error>(ErrorType::INVALID_ARGUMENT, e.what());
    }
}

std::string
format_map(const std::string& str, const std::unordered_map<std::string, std::string>& mappings);

class LinearCongruentialGenerator {
public:
    LinearCongruentialGenerator() {
        auto now = std::chrono::steady_clock::now();
        auto timestamp =
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        current_ = static_cast<unsigned int>(timestamp);
    }

    float
    NextFloat() {
        current_ = (A * current_ + C) % M;
        return static_cast<float>(current_) / static_cast<float>(M);
    }

private:
    unsigned int current_;
    static const uint32_t A = 1664525;
    static const uint32_t C = 1013904223;
    static const uint32_t M = 4294967295;  // 2^32 - 1
};

}  // namespace vsag
