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
#include <utility>

namespace vsag::eval {

template <typename Clock = std::chrono::steady_clock, typename Search>
auto
MeasureSearch(Search&& search) {
    const auto start = Clock::now();
    auto result = std::forward<Search>(search)();
    const auto end = Clock::now();
    const auto latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return std::make_pair(std::move(result), latency_ms);
}

}  // namespace vsag::eval
