
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
        this->align_bit = Options::Instance().direct_IO_object_align_bit();
        this->align_size = 1 << align_bit;
        this->align_mask = (1 << align_bit) - 1;
        this->size = size1;
        this->offset = offset1;
        if (align_data) {
            free(align_data);
        }
        auto new_offset = (offset >> align_bit) << align_bit;
        auto inner_offset = offset & align_mask;
        auto new_size = (((size + inner_offset) + align_mask) >> align_bit) << align_bit;
        this->align_data = static_cast<uint8_t*>(std::aligned_alloc(align_size, new_size));
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

    int64_t align_bit;

    int64_t align_size;

    int64_t align_mask;
};
}  // namespace vsag
