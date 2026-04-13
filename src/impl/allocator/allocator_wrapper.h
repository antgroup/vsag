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
 * @file allocator_wrapper.h
 * @brief STL-compatible allocator wrapper for vsag::Allocator.
 */

#pragma once

#include "vsag/allocator.h"

namespace vsag {

/**
 * @brief STL-compatible allocator wrapper that adapts vsag::Allocator for STL containers.
 *
 * This template class wraps a vsag::Allocator pointer and provides the interface
 * required by STL containers, enabling them to use custom memory allocation.
 *
 * @tparam T The type of objects to allocate.
 */
template <class T>
class AllocatorWrapper {
public:
    /// The value type being allocated.
    using value_type = T;
    /// Pointer type to the value type.
    using pointer = T*;
    /// Pointer type to void.
    using void_pointer = void*;
    /// Pointer type to const void.
    using const_void_pointer = const void*;
    /// Unsigned integer type for sizes.
    using uint64_type = uint64_t;
    /// Difference type for pointer arithmetic.
    using difference_type = std::ptrdiff_t;

    /**
     * @brief Constructs an AllocatorWrapper with a vsag allocator.
     *
     * @param allocator Pointer to the vsag allocator to wrap.
     */
    AllocatorWrapper(Allocator* allocator) {
        this->allocator_ = allocator;
    }

    /**
     * @brief Copy constructor from another AllocatorWrapper with different type.
     *
     * @tparam U The value type of the other allocator.
     * @param other The other AllocatorWrapper to copy from.
     */
    template <class U>
    AllocatorWrapper(const AllocatorWrapper<U>& other) : allocator_(other.allocator_) {
    }

    /**
     * @brief Equality comparison operator.
     *
     * @param other The other AllocatorWrapper to compare with.
     * @return true if both allocators are the same, false otherwise.
     */
    bool
    operator==(const AllocatorWrapper& other) const noexcept {
        return allocator_ == other.allocator_;
    }

    /**
     * @brief Inequality comparison operator.
     *
     * @param other The other AllocatorWrapper to compare with.
     * @return true if allocators are different, false otherwise.
     */
    bool
    operator!=(const AllocatorWrapper& other) const noexcept {
        return allocator_ != other.allocator_;
    }

    /**
     * @brief Allocates memory for n objects.
     *
     * @param n Number of objects to allocate space for.
     * @param hint Unused allocation hint (for compatibility with STL).
     * @return Pointer to the allocated memory.
     */
    inline pointer
    allocate(uint64_type n, const_void_pointer hint = 0) {
        return static_cast<pointer>(allocator_->Allocate(n * sizeof(value_type)));
    }

    /**
     * @brief Deallocates memory.
     *
     * @param p Pointer to the memory to deallocate.
     * @param n Number of objects (unused, for STL compatibility).
     */
    inline void
    deallocate(pointer p, uint64_type n) {
        allocator_->Deallocate(static_cast<void_pointer>(p));
    }

    /**
     * @brief Constructs an object in place.
     *
     * @tparam U The type of object to construct.
     * @tparam Args Constructor argument types.
     * @param p Pointer to the memory location.
     * @param args Constructor arguments.
     */
    template <class U, class... Args>
    inline void
    construct(U* p, Args&&... args) {
        ::new ((void_pointer)p) U(std::forward<Args>(args)...);
    }

    /**
     * @brief Destroys an object in place.
     *
     * @tparam U The type of object to destroy.
     * @param p Pointer to the object to destroy.
     */
    template <class U>
    inline void
    destroy(U* p) {
        p->~U();
    }

    /**
     * @brief Template for rebinding the allocator to a different type.
     *
     * @tparam U The new value type.
     */
    template <class U>
    struct rebind {
        /// The rebound allocator type.
        using other = AllocatorWrapper<U>;
    };

    /// Pointer to the underlying vsag allocator.
    Allocator* allocator_{};
};
}  // namespace vsag