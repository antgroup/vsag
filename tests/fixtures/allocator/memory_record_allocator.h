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
 * @file memory_record_allocator.h
 * @brief Memory allocator with tracking and peak memory recording.
 */

#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "vsag/allocator.h"

namespace fixtures {

/**
 * @class MemoryRecordAllocator
 * @brief Allocator that tracks memory allocations and records peak memory usage.
 * Useful for memory leak detection and memory usage analysis in tests.
 */
class MemoryRecordAllocator : public vsag::Allocator {
public:
    /**
     * @brief Returns the name of this allocator.
     * @return String identifier "MemoryRecordAllocator".
     */
    std::string
    Name() override;

    /**
     * @brief Constructs a new MemoryRecordAllocator instance.
     */
    MemoryRecordAllocator();

    /**
     * @brief Allocates memory of the specified size and tracks the allocation.
     * @param size The number of bytes to allocate.
     * @return Pointer to the allocated memory, or nullptr on failure.
     */
    void*
    Allocate(uint64_t size) override;

    /**
     * @brief Deallocates memory and removes it from tracking.
     * @param p Pointer to the memory to deallocate.
     */
    void
    Deallocate(void* p) override;

    /**
     * @brief Reallocates memory to a new size, updating tracking accordingly.
     * @param p Pointer to existing memory (may be nullptr).
     * @param size The new size in bytes.
     * @return Pointer to the reallocated memory, or nullptr on failure.
     */
    void*
    Reallocate(void* p, uint64_t size) override;

    /**
     * @brief Returns the peak memory usage recorded during allocations.
     * @return Maximum memory bytes allocated at any point.
     */
    uint64_t
    GetMemoryPeak() const;

    /**
     * @brief Returns the current total allocated memory.
     * @return Sum of bytes currently allocated and not deallocated.
     */
    uint64_t
    GetCurrentMemory() const;

private:
    uint64_t memory_bytes_{0};                       // Current total allocated bytes.
    uint64_t memory_peak_{0};                        // Peak memory usage recorded.
    std::unordered_map<void*, uint64_t> records_{};  // Map of pointers to their allocation sizes.
    std::mutex mutex_{};                             // Mutex for thread-safe operations.
};

}  // namespace fixtures
