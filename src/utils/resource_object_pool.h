
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

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <type_traits>

#include "impl/allocator/safe_allocator.h"
#include "resource_object.h"
#include "typing.h"

namespace vsag {

template <typename T,
          typename = typename std::enable_if<std::is_base_of<ResourceObject, T>::value>::type>
class ResourceObjectPool {
public:
    using ConstructFuncType = std::function<std::shared_ptr<T>()>;
    static constexpr uint64_t kSubPoolCount = 16;

public:
    template <typename... Args>
    explicit ResourceObjectPool(uint64_t init_size, Allocator* allocator, Args... args)
        : allocator_(allocator), init_size_(init_size) {
        this->constructor_ = [=]() -> std::shared_ptr<T> { return std::make_shared<T>(args...); };
        if (allocator_ == nullptr) {
            this->owned_allocator_ = SafeAllocator::FactoryDefaultAllocator();
            this->allocator_ = owned_allocator_.get();
        }
        for (int i = 0; i < kSubPoolCount; ++i) {
            pool_[i] = std::make_unique<Deque<std::shared_ptr<T>>>(this->allocator_);
        }
        this->fill(init_size_);
    }

    ~ResourceObjectPool() {
        if (owned_allocator_ != nullptr) {
            for (int i = 0; i < kSubPoolCount; ++i) {
                pool_[i].reset();
            }
        }
    }

    std::shared_ptr<T>
    TakeOne() {
        while (true) {
            for (int i = 0; i < kSubPoolCount; ++i) {
                if (sub_pool_mutexes_[i].try_lock()) {
                    if (pool_[i]->empty()) {
                        sub_pool_mutexes_[i].unlock();
                        return this->constructor_();
                    }
                    std::shared_ptr<T> obj = pool_[i]->front();
                    pool_[i]->pop_front();
                    sub_pool_mutexes_[i].unlock();
                    obj->Reset();
                    return obj;
                }
            }
        }
    }

    void
    ReturnOne(std::shared_ptr<T>& obj) {
        while (true) {
            for (int i = 0; i < kSubPoolCount; ++i) {
                if (sub_pool_mutexes_[i].try_lock()) {
                    pool_[i]->emplace_back(obj);
                    sub_pool_mutexes_[i].unlock();
                    return;
                }
            }
        }
    }

private:
    inline void
    fill(uint64_t size) {
        for (uint64_t i = 0; i < size; ++i) {
            auto sub_pool_idx = i % kSubPoolCount;
            pool_[sub_pool_idx]->emplace_back(this->constructor_());
        }
    }

private:
    std::unique_ptr<Deque<std::shared_ptr<T>>> pool_[kSubPoolCount];
    std::mutex sub_pool_mutexes_[kSubPoolCount];
    uint64_t init_size_{0};

    ConstructFuncType constructor_{nullptr};
    Allocator* allocator_{nullptr};

    std::shared_ptr<Allocator> owned_allocator_{nullptr};
};
}  // namespace vsag
