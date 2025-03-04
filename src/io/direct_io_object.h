
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
#include <string>

namespace vsag {

class DirectIOObject {
public:
    DirectIOObject() = default;

    DirectIOObject(uint64_t size, uint64_t offset) {
        this->Set(size, offset);
    }

    void
    Set(uint64_t size1, uint64_t offset1) {
        this->size = size1;
        this->offset = offset1;
        if (align_data) {
            free(align_data);
        }
        auto new_offset = (offset >> ALIGN_BIT) << ALIGN_BIT;
        auto inner_offset = offset & ALIGN_MASK;
        auto new_size = (((size + inner_offset) + ALIGN_MASK) >> ALIGN_BIT) << ALIGN_BIT;
        this->align_data = static_cast<uint8_t*>(std::aligned_alloc(ALIGN_SIZE, new_size));
        this->data = align_data + inner_offset;
        this->size = new_size;
        this->offset = new_offset;
    }

    void
    Release() {
        free(this->align_data);
        this->align_data = nullptr;
        this->data = nullptr;
    }

public:
    uint8_t* data{nullptr};
    uint64_t size;
    uint64_t offset;
    uint8_t* align_data{nullptr};

    static constexpr int64_t ALIGN_BIT = 9;

    static constexpr int64_t ALIGN_SIZE = 1 << ALIGN_BIT;

    static constexpr int64_t ALIGN_MASK = (1 << ALIGN_BIT) - 1;
};
}  // namespace vsag
