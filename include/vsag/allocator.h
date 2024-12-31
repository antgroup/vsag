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

#include <string>

namespace vsag {

/**
 * @class Allocator
 * @brief An abstract base class for custom memory management.
 *
 * The `Allocator` class provides a standard interface for custom memory
 * allocation, deallocation, and reallocation, supporting user define control
 * over memory usage in applications.
 */
class Allocator {
public:
    /**
     * @brief Returns the name of the allocator.
     *
     * This pure virtual function should be overridden to provide the name
     * or identifier of the custom allocator implementation.
     *
     * @return std::string The name of the allocator.
     */
    virtual std::string
    Name() = 0;

    /**
     * @brief Allocates a block of memory.
     *
     * This pure virtual function should be overridden to allocate a block
     * of memory of at least the specified size.
     *
     * @param size The size of the memory block to allocate.
     * @return void* Pointer to the allocated memory block.
     */
    virtual void*
    Allocate(size_t size) = 0;

    /**
     * @brief Deallocates a previously allocated block of memory.
     *
     * This pure virtual function should be overridden to deallocate a block
     * of memory that was previously allocated by this allocator.
     *
     * @param p Pointer to the memory block to deallocate.
     */
    virtual void
    Deallocate(void* p) = 0;

    /**
     * @brief Reallocates a previously allocated block with a new size.
     *
     * This pure virtual function should be overridden to resize a block of
     * memory that was previously allocated by this allocator.
     *
     * @param p Pointer to the memory block to reallocate.
     * @param size The new size of the memory block.
     * @return void* Pointer to the reallocated memory block.
     */
    virtual void*
    Reallocate(void* p, size_t size) = 0;

    /**
     * @brief Constructs a new object of type T.
     *
     * This template function allocates memory for an object of type T and
     * constructs it using the provided arguments.
     *
     * @tparam T The type of the object to construct.
     * @tparam Args The types of the arguments for the constructor of T.
     * @param args The arguments to pass to the constructor of T.
     * @return T* Pointer to the newly constructed object.
     */
    template <typename T, typename... Args>
    T*
    New(Args&&... args) {
        void* p = Allocate(sizeof(T));
        try {
            return (T*)::new (p) T(std::forward<Args>(args)...);
        } catch (std::exception& e) {
            Deallocate(p);
            throw e;
        }
    }

    /**
     * @brief Destroys an object of type T and deallocates its memory.
     *
     * This template function calls the destructor of an object of type T
     * and deallocates the memory it occupies.
     *
     * @tparam T The type of the object to destroy.
     * @param p Pointer to the object to destroy.
     */
    template <typename T>
    void
    Delete(T* p) {
        if (p) {
            p->~T();
            Deallocate(static_cast<void*>(p));
        }
    }

public:
    virtual ~Allocator() = default;
};

}  // namespace vsag
