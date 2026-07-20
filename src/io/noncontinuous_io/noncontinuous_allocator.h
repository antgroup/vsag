
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

/**
 * @brief Represents a non-continuous address area with offset and size.
 *
 * This structure defines a contiguous region within non-continuous storage,
 * identified by its physical offset and size.
 */
struct NonContinuousArea {
public:
    /// Physical offset of the area in the underlying storage.
    uint64_t offset{0};

    /// Size of the area in bytes.
    uint64_t size{0};

    /**
     * @brief Default constructor.
     */
    NonContinuousArea() = default;

    /**
     * @brief Constructs an area with specified offset and size.
     *
     * @param offset The physical offset of the area.
     * @param size The size of the area in bytes.
     */
    explicit NonContinuousArea(uint64_t offset, uint64_t size) : offset(offset), size(size){};
};

/**
 * @brief Allocator for managing non-continuous address space regions.
 *
 * This class allocates 4KB-aligned address areas for non-continuous storage.
 * It tracks the last allocated offset to ensure areas don't overlap.
 */
class NonContinuousAllocator {
public:
    /**
     * @brief Constructs a NonContinuousAllocator with a base allocator.
     *
     * @param allocator A pointer to the Allocator for memory management.
     */
    explicit NonContinuousAllocator(Allocator* allocator) : allocator_(allocator) {
    }

    /**
     * @brief Default destructor.
     */
    ~NonContinuousAllocator() = default;

    /**
     * @brief Requires a new area of the specified size.
     *
     * Allocates a 4KB-aligned area and returns its offset and size.
     * Updates the last_offset_ for subsequent allocations.
     *
     * @param size The requested size for the new area.
     * @return The allocated NonContinuousArea with offset and aligned size.
     */
    [[nodiscard]] inline NonContinuousArea
    Require(uint64_t size) {
        // 4k align
        size = (size + ALOGN_SIZE - 1) & ~(ALOGN_SIZE - 1);
        NonContinuousArea area{last_offset_, size};
        last_offset_ += size;
        return area;
    }

private:
    /// Base allocator for memory management (unused in this implementation).
    Allocator* const allocator_{nullptr};

    /// Last allocated offset for tracking non-overlapping areas.
    uint64_t last_offset_{0};

    /// Alignment size for area allocation (4KB).
    static constexpr uint64_t ALOGN_SIZE = 4096;
};

}  // namespace vsag
