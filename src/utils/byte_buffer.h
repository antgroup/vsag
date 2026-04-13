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

#include "vsag/allocator.h"

namespace vsag {

/**
 * @file byte_buffer.h
 * @brief Simple byte buffer with allocator-aware memory management.
 */

/**
 * @class ByteBuffer
 * @brief RAII wrapper for a byte buffer with custom allocator support.
 *
 * This class provides a simple byte buffer that allocates memory on construction
 * and deallocates on destruction using the provided allocator. Useful for
 * managing temporary byte storage with custom memory management.
 */
class ByteBuffer {
public:
    /**
     * @brief Constructs a byte buffer with the specified size.
     * @param size Size in bytes to allocate.
     * @param allocator Allocator for memory management.
     */
    ByteBuffer(uint64_t size, Allocator* allocator) : allocator(allocator) {
        data = static_cast<uint8_t*>(allocator->Allocate(size));
    }

    /**
     * @brief Destructor that deallocates the buffer.
     */
    ~ByteBuffer() {
        allocator->Deallocate(data);
    }

public:
    /// Pointer to the allocated byte data
    uint8_t* data;

    /// Allocator used for memory management
    Allocator* allocator;
};
}  // namespace vsag