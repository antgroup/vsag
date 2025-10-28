
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

class Allocator;
struct NonContinuousArea {
public:
    uint64_t offset{0};
    uint64_t size{0};

    NonContinuousArea() = default;
    explicit NonContinuousArea(uint64_t offset, uint64_t size) : offset(offset), size(size){};
};

class NonContinuousAllocator {
public:
    explicit NonContinuousAllocator(Allocator* allocator) : allocator_(allocator) {
    }

    ~NonContinuousAllocator() = default;

    [[nodiscard]] inline NonContinuousArea
    Require(uint64_t size) {
        // 4k align
        size = (size + ALOGN_SIZE - 1) & ~(ALOGN_SIZE - 1);
        NonContinuousArea area{last_offset_, size};
        last_offset_ += size;
        return area;
    }

private:
    Allocator* const allocator_{nullptr};

    uint64_t last_offset_{0};

    static constexpr uint64_t ALOGN_SIZE = 4096;
};

}  // namespace vsag
