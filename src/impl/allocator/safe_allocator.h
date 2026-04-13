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
 * @file safe_allocator.h
 * @brief Safe allocator wrapper that provides exception-safe memory allocation.
 */

#pragma once

#include <memory>
#include <new>

#include "default_allocator.h"
#include "vsag/allocator.h"

namespace vsag {

/**
 * @brief Safe allocator wrapper that provides exception-safe memory allocation.
 *
 * This class wraps an existing allocator and throws std::bad_alloc when
 * memory allocation fails, ensuring consistent error handling behavior.
 */
class SafeAllocator : public Allocator {
public:
    /**
     * @brief Creates a default safe allocator with a DefaultAllocator instance.
     *
     * @return A shared pointer to a new SafeAllocator wrapping DefaultAllocator.
     */
    static std::shared_ptr<Allocator>
    FactoryDefaultAllocator() {
        return std::make_shared<SafeAllocator>(new DefaultAllocator(), true);
    }

public:
    /**
     * @brief Constructs a SafeAllocator with a raw allocator pointer (non-owning).
     *
     * @param raw_allocator Pointer to the underlying allocator.
     */
    explicit SafeAllocator(Allocator* raw_allocator) : SafeAllocator(raw_allocator, false){};

    /**
     * @brief Constructs a SafeAllocator with a raw allocator pointer.
     *
     * @param raw_allocator Pointer to the underlying allocator.
     * @param owned Whether this wrapper owns the allocator and should delete it on destruction.
     */
    explicit SafeAllocator(Allocator* raw_allocator, bool owned)
        : raw_allocator_(raw_allocator), owned_(owned) {
    }

    /**
     * @brief Constructs a SafeAllocator with a shared allocator pointer.
     *
     * @param raw_allocator Shared pointer to the underlying allocator.
     */
    explicit SafeAllocator(const std::shared_ptr<Allocator>& raw_allocator)
        : raw_allocator_shared_(raw_allocator), raw_allocator_(raw_allocator.get()) {
    }

    /**
     * @brief Gets the name of this allocator.
     *
     * @return The allocator name with "_safewrapper" suffix.
     */
    std::string
    Name() override {
        return raw_allocator_->Name() + "_safewrapper";
    }

    /**
     * @brief Allocates memory of the specified size.
     *
     * @param size The number of bytes to allocate.
     * @return Pointer to the allocated memory.
     * @throws std::bad_alloc If allocation fails.
     */
    void*
    Allocate(uint64_t size) override {
        auto ret = raw_allocator_->Allocate(size);
        if (not ret) {
            throw std::bad_alloc();
        }
        return ret;
    }

    /**
     * @brief Deallocates previously allocated memory.
     *
     * @param p Pointer to the memory to deallocate.
     */
    void
    Deallocate(void* p) override {
        raw_allocator_->Deallocate(p);
    }

    /**
     * @brief Reallocates memory to a new size.
     *
     * @param p Pointer to the memory to reallocate.
     * @param size The new size in bytes.
     * @return Pointer to the reallocated memory.
     * @throws std::bad_alloc If reallocation fails.
     */
    void*
    Reallocate(void* p, uint64_t size) override {
        auto ret = raw_allocator_->Reallocate(p, size);
        if (not ret) {
            throw std::bad_alloc();
        }
        return ret;
    }

    /**
     * @brief Gets the raw underlying allocator.
     *
     * @return Pointer to the underlying allocator.
     */
    Allocator*
    GetRawAllocator() {
        return raw_allocator_;
    }

public:
    /**
     * @brief Destructs the SafeAllocator.
     *
     * Deletes the underlying allocator if owned.
     */
    ~SafeAllocator() override {
        if (owned_) {
            delete raw_allocator_;
        }
    }

private:
    /// Raw pointer to the underlying allocator.
    Allocator* const raw_allocator_{nullptr};

    /// Shared pointer to the underlying allocator (for shared ownership).
    std::shared_ptr<Allocator> const raw_allocator_shared_;

    /// Whether this wrapper owns the allocator.
    bool owned_{false};
};

}  // namespace vsag