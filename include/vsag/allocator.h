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
 * @brief An abstract base class that defines a general interface for memory allocation.
 *
 * This class provides a set of pure virtual methods for memory management,
 * including allocation, deallocation, and reallocation of memory blocks.
 */
class Allocator {
public:
    /**
     * @brief Get the name of the allocator.
     * @return A string representing the name of the allocator.
     */
    virtual std::string
    Name() = 0;

    /**
     * @brief Allocate a block of memory of at least the specified size.
     * @param size The size of memory to allocate in bytes.
     * @return A pointer to the beginning of the allocated memory block.
     */
    virtual void*
    Allocate(size_t size) = 0;

    /**
     * @brief Deallocate a previously allocated block of memory.
     * @param p A pointer to the memory block to deallocate.
     */
    virtual void
    Deallocate(void* p) = 0;

    /**
     * @brief Reallocate the previously allocated block with a new size.
     * @param p A pointer to the previously allocated memory block.
     * @param size The new size for the memory block in bytes.
     * @return A pointer to the reallocated memory block.
     */
    virtual void*
    Reallocate(void* p, size_t size) = 0;

    /**
     * @brief Construct a new object and allocate memory for it.
     * @tparam T The type of the object to be created.
     * @tparam Args The types of the arguments for the constructor of T.
     * @param args The arguments to be forwarded to the constructor of T.
     * @return A pointer to the newly constructed object.
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
     * @brief Destroy an object and deallocate its memory.
     * @tparam T The type of the object to be destroyed.
     * @param p A pointer to the object to be destroyed.
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
    /**
     * @brief Virtual destructor to allow proper cleanup of derived classes.
     */
    virtual ~Allocator() = default;
};

}  // namespace vsag