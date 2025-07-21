
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

#include <shared_mutex>

#include "typing.h"

namespace vsag {

class MutexArray {
public:
    virtual void
    Lock(uint32_t i) = 0;

    virtual void
    Unlock(uint32_t i) = 0;

    virtual void
    SharedLock(uint32_t i) = 0;

    virtual void
    SharedUnlock(uint32_t i) = 0;

    virtual void
    Resize(uint32_t new_element_num) = 0;
};

using MutexArrayPtr = std::shared_ptr<MutexArray>;

class PointsMutex : public MutexArray {
public:
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

private:
    Vector<std::shared_ptr<std::shared_mutex>> neighbors_mutex_;
    Allocator* const allocator_{nullptr};
    uint32_t element_num_{0};
};

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
};

// To reduce the overhead of the construction and destruction of share_ptr,
// the mutex_impl parameter is passed by reference rather than by value.
class SharedLock {
public:
    SharedLock(const MutexArrayPtr& mutex_impl, uint32_t locked_index)
        : mutex_impl_(mutex_impl), locked_index_(locked_index) {
        mutex_impl_->SharedLock(locked_index_);
    }
    ~SharedLock() {
        mutex_impl_->SharedUnlock(locked_index_);
    }

private:
    uint32_t locked_index_;
    const MutexArrayPtr& mutex_impl_;
};

class LockGuard {
public:
    LockGuard(MutexArrayPtr mutex_impl, uint32_t locked_index)
        : mutex_impl_(mutex_impl), locked_index_(locked_index) {
        mutex_impl_->Lock(locked_index_);
    }
    ~LockGuard() {
        mutex_impl_->Unlock(locked_index_);
    }

private:
    uint32_t locked_index_;
    MutexArrayPtr mutex_impl_;
};

}  // namespace vsag
