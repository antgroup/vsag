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
 * @file lock_strategy.h
 * @brief Lock abstraction and RAII guard classes for thread synchronization.
 */

#pragma once

#include <shared_mutex>

#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER(MutexArray);

/**
 * @brief Abstract interface for mutex arrays supporting per-element locking.
 *
 * Provides an interface for managing multiple mutexes indexed by element ID,
 * supporting both exclusive and shared locking modes for read-write scenarios.
 */
class MutexArray {
public:
    virtual ~MutexArray() = default;

    /**
     * @brief Acquire an exclusive lock on the specified element.
     * @param i The element index to lock.
     */
    virtual void
    Lock(uint32_t i) = 0;

    /**
     * @brief Release an exclusive lock on the specified element.
     * @param i The element index to unlock.
     */
    virtual void
    Unlock(uint32_t i) = 0;

    /**
     * @brief Acquire a shared lock on the specified element.
     * @param i The element index to lock.
     */
    virtual void
    SharedLock(uint32_t i) = 0;

    /**
     * @brief Release a shared lock on the specified element.
     * @param i The element index to unlock.
     */
    virtual void
    SharedUnlock(uint32_t i) = 0;

    /**
     * @brief Resize the mutex array for a new number of elements.
     * @param new_element_num The new number of elements to support.
     */
    virtual void
    Resize(uint32_t new_element_num) = 0;

    /**
     * @brief Get the memory usage of this mutex array.
     * @return int64_t Memory usage in bytes.
     */
    virtual int64_t
    GetMemoryUsage() = 0;
};

/**
 * @brief Mutex array implementation using shared_mutex for each element.
 *
 * Provides per-element read-write mutexes for fine-grained concurrency control.
 */
class PointsMutex : public MutexArray {
public:
    /**
     * @brief Construct a PointsMutex with the specified number of elements.
     * @param element_num Number of elements (mutexes) to create.
     * @param allocator Allocator for memory allocation.
     */
    PointsMutex(uint32_t element_num, Allocator* allocator);

    void
    SharedLock(uint32_t i) override;

    void
    SharedUnlock(uint32_t i) override;

    void
    Lock(uint32_t i) override;

    void
    Unlock(uint32_t i) override;

    void
    Resize(uint32_t new_element_num) override;

    int64_t
    GetMemoryUsage() override;

private:
    Vector<std::shared_ptr<std::shared_mutex>> neighbors_mutex_;  ///< Per-element mutexes
    Allocator* const allocator_{nullptr};                         ///< Allocator for memory
    uint32_t element_num_{0};                                     ///< Number of elements (mutexes)
};

/**
 * @brief No-op mutex array implementation for single-threaded scenarios.
 *
 * Provides empty implementations of all lock operations for use when
 * thread safety is not required.
 */
class EmptyMutex : public MutexArray {
public:
    void
    SharedLock(uint32_t i) override {
    }

    void
    SharedUnlock(uint32_t i) override {
    }

    void
    Lock(uint32_t i) override {
    }

    void
    Unlock(uint32_t i) override {
    }

    void
    Resize(uint32_t new_element_num) override {
    }

    int64_t
    GetMemoryUsage() override {
        return 0;
    }
};

/**
 * @brief RAII guard for shared (read) locks.
 *
 * Acquires a shared lock on construction and releases it on destruction.
 * The mutex_impl parameter is passed by reference to reduce shared_ptr overhead.
 */
class SharedLock {
public:
    /**
     * @brief Construct a SharedLock and acquire a shared lock.
     * @param mutex_impl Pointer to the mutex array implementation.
     * @param locked_index The element index to lock.
     */
    SharedLock(const MutexArrayPtr& mutex_impl, uint32_t locked_index)
        : mutex_impl_(mutex_impl), locked_index_(locked_index) {
        mutex_impl_->SharedLock(locked_index_);
    }
    ~SharedLock() {
        mutex_impl_->SharedUnlock(locked_index_);
    }

private:
    uint32_t locked_index_;            ///< The locked element index
    const MutexArrayPtr& mutex_impl_;  ///< Reference to mutex array
};

/**
 * @brief RAII guard for exclusive (write) locks.
 *
 * Acquires an exclusive lock on construction and releases it on destruction.
 */
class LockGuard {
public:
    /**
     * @brief Construct a LockGuard and acquire an exclusive lock.
     * @param mutex_impl Pointer to the mutex array implementation.
     * @param locked_index The element index to lock.
     */
    LockGuard(MutexArrayPtr mutex_impl, uint32_t locked_index)
        : mutex_impl_(mutex_impl), locked_index_(locked_index) {
        mutex_impl_->Lock(locked_index_);
    }
    ~LockGuard() {
        mutex_impl_->Unlock(locked_index_);
    }

private:
    uint32_t locked_index_;     ///< The locked element index
    MutexArrayPtr mutex_impl_;  ///< Mutex array pointer
};

}  // namespace vsag