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
 * @file default_allocator.h
 * @brief Default memory allocator implementation using standard malloc/free.
 */

#pragma once

#include <mutex>
#include <unordered_set>

#include "vsag/allocator.h"

namespace vsag {

/**
 * @brief Default memory allocator using standard C library functions.
 *
 * This allocator uses malloc, free, and realloc for memory management.
 * In debug mode, it tracks allocated pointers for memory leak detection.
 */
class DefaultAllocator : public Allocator {
public:
    /**
     * @brief Default constructor.
     */
    DefaultAllocator() = default;

    /**
     * @brief Destructor.
     */
    ~DefaultAllocator() override;

    DefaultAllocator(const DefaultAllocator&) = delete;
    DefaultAllocator(DefaultAllocator&&) = delete;

public:
    /**
     * @brief Gets the name of this allocator.
     *
     * @return The string "default".
     */
    std::string
    Name() override;

    /**
     * @brief Allocates memory of the specified size.
     *
     * @param size The number of bytes to allocate.
     * @return Pointer to the allocated memory, or nullptr on failure.
     */
    void*
    Allocate(uint64_t size) override;

    /**
     * @brief Deallocates previously allocated memory.
     *
     * @param p Pointer to the memory to deallocate.
     */
    void
    Deallocate(void* p) override;

    /**
     * @brief Reallocates memory to a new size.
     *
     * @param p Pointer to the memory to reallocate.
     * @param size The new size in bytes.
     * @return Pointer to the reallocated memory, or nullptr on failure.
     */
    void*
    Reallocate(void* p, uint64_t size) override;

private:
#ifndef NDEBUG
    /// Set of allocated pointers for debug tracking.
    std::unordered_set<void*> allocated_ptrs_;
    /// Mutex for thread-safe access to allocated_ptrs_.
    std::mutex set_mutex_;
#endif
};

}  // namespace vsag