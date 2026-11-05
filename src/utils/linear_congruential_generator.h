
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

#include <cstdint>

namespace vsag {
class LinearCongruentialGenerator {
public:
    LinearCongruentialGenerator();

    float
    NextFloat();

private:
    unsigned int current_;
    static constexpr uint32_t A = 1664525;
    static constexpr uint32_t C = 1013904223;
    static constexpr uint32_t M = 4294967295;  // 2^32 - 1
};
}  // namespace vsag
