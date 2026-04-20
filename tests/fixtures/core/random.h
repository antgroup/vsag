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
 * @file random.h
 * @brief Random value generation utilities for tests.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace fixtures {

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
RandomValue(const T& min, const T& max, int seed = -1) {
    if (seed >= 0) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<T> dis(min, max);
        return dis(gen);
    }
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(min, max);
    return dis(gen);
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
RandomValue(const T& min, const T& max, int seed = -1) {
    if (seed >= 0) {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<T> dis(min, max);
        return dis(gen);
    }
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    std::uniform_int_distribution<T> dis(min, max);
    return dis(gen);
}

template <typename T>
std::vector<T>
RandomSelect(const std::vector<T>& vec, int64_t count = 1) {
    std::vector<T> selected;
    count = std::min(count, static_cast<int64_t>(vec.size()));
    std::sample(vec.begin(),
                vec.end(),
                std::back_inserter(selected),
                count,
                std::mt19937(RandomValue(0, 10000)));
    return selected;
}

}  // namespace fixtures
