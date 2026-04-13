
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
#include <cstdlib>

#include "vsag/options.h"

namespace vsag {

/**
 * @brief Helper object for performing aligned direct IO operations.
 *
 * This class manages aligned memory buffers required for direct IO (O_DIRECT),
 * which bypasses the kernel page cache. It calculates aligned offsets and sizes
 * based on the configured alignment bit value.
 */
class DirectIOObject {
public:
    /**
     * @brief Constructs a DirectIOObject with alignment from Options.
     *
     * Initializes alignment parameters from global Options configuration.
     */
    DirectIOObject() {
        this->align_bit = Options::Instance().direct_IO_object_align_bit();
        this->align_size = 1 << align_bit;
        this->align_mask = (1 << align_bit) - 1;
    }

    /**
     * @brief Constructs a DirectIOObject with pre-set size and offset.
     *
     * @param size The requested size for the IO operation.
     * @param offset The requested offset for the IO operation.
     */
    DirectIOObject(uint64_t size, uint64_t offset) : DirectIOObject() {
        this->Set(size, offset);
    }

    /**
     * @brief Sets up aligned buffer for a given size and offset.
     *
     * Calculates the aligned offset and size, allocates aligned memory,
     * and sets up the data pointer with the correct inner offset.
     *
     * @param size1 The requested size for the IO operation.
     * @param offset1 The requested offset for the IO operation.
     */
    void
    Set(uint64_t size1, uint64_t offset1) {
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

    /**
     * @brief Releases the aligned memory buffer.
     */
    void
    Release() {
        free(this->align_data);
        this->align_data = nullptr;
        this->data = nullptr;
    }

public:
    /// Pointer to the usable data region within the aligned buffer.
    uint8_t* data{nullptr};

    /// Aligned size for the IO operation.
    uint64_t size;

    /// Aligned offset for the IO operation.
    uint64_t offset;

    /// Pointer to the aligned memory buffer (base of aligned allocation).
    uint8_t* align_data{nullptr};

    /// Bit count for alignment (e.g., 12 for 4KB alignment).
    int64_t align_bit;

    /// Alignment size in bytes (1 << align_bit).
    int64_t align_size;

    /// Mask for calculating inner offset within aligned block (align_size - 1).
    int64_t align_mask;
};
}  // namespace vsag
