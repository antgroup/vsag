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

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <string>
#include <utility>

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
    virtual ~Allocator() = default;

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
    Allocate(uint64_t size) = 0;

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
    Reallocate(void* p, uint64_t size) = 0;

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
            throw;
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
};

/**
 * @brief STL allocator adapter backed by a VSAG Allocator.
 *
 * The referenced Allocator must outlive every container that uses this adapter.
 */
template <typename T>
class AllocatorWrapper {
public:
    using value_type = T;
    using pointer = T*;
    using void_pointer = void*;
    using const_void_pointer = const void*;
    using uint64_type = uint64_t;
    using difference_type = std::ptrdiff_t;

    AllocatorWrapper(Allocator* allocator) : allocator_(allocator) {
    }

    template <typename U>
    AllocatorWrapper(const AllocatorWrapper<U>& other) : allocator_(other.allocator_) {
    }

    bool
    operator==(const AllocatorWrapper& other) const noexcept {
        return allocator_ == other.allocator_;
    }

    bool
    operator!=(const AllocatorWrapper& other) const noexcept {
        return allocator_ != other.allocator_;
    }

    pointer
    allocate(uint64_type n, const_void_pointer = nullptr) {
        auto* p = allocator_->Allocate(n * sizeof(value_type));
        if (p == nullptr) {
            throw std::bad_alloc();
        }
        return static_cast<pointer>(p);
    }

    void
    deallocate(pointer p, uint64_type) {
        allocator_->Deallocate(static_cast<void_pointer>(p));
    }

    template <typename U, typename... Args>
    void
    construct(U* p, Args&&... args) {
        ::new (static_cast<void_pointer>(p)) U(std::forward<Args>(args)...);
    }

    template <typename U>
    void
    destroy(U* p) {
        p->~U();
    }

    template <typename U>
    struct Rebind {
        using other = AllocatorWrapper<U>;
    };

    template <typename U>
    using rebind = Rebind<U>;

    Allocator* allocator_{nullptr};
};

}  // namespace vsag
