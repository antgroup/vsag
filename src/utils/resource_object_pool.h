
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
#include <thread>
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
        : allocator_(allocator), init_size_(init_size), memory_usage_(0) {
        this->constructor_ = [=, pool = this]() -> std::shared_ptr<T> {
            auto ptr = std::make_shared<T>(args...);
            auto value = ptr->MemoryUsage();
            pool->memory_usage_.fetch_add(value, std::memory_order_relaxed);
            return ptr;
        };
        if (allocator_ == nullptr) {
            this->owned_allocator_ = SafeAllocator::FactoryDefaultAllocator();
            this->allocator_ = owned_allocator_.get();
        }
        for (int i = 0; i < kSubPoolCount; ++i) {
            pool_[i] = std::make_unique<Deque<std::shared_ptr<T>>>(this->allocator_);
        }
        this->fill(init_size_);
        memory_usage_ += kSubPoolCount * sizeof(Deque<std::shared_ptr<T>>);
        memory_usage_ += sizeof(ResourceObjectPool<T>);
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
        thread_local static uint64_t prefer_pool_id_{
            std::hash<std::thread::id>()(std::this_thread::get_id()) % kSubPoolCount};
        auto pool_id = prefer_pool_id_;
        while (true) {
            if (sub_pool_mutexes_[pool_id].try_lock()) {
                prefer_pool_id_ = pool_id;
                if (pool_[pool_id]->empty()) {
                    sub_pool_mutexes_[pool_id].unlock();
                    auto obj = this->constructor_();
                    obj->source_pool_id_ = pool_id;
                    return obj;
                }
                std::shared_ptr<T> obj = pool_[pool_id]->front();
                pool_[pool_id]->pop_front();
                sub_pool_mutexes_[pool_id].unlock();
                obj->source_pool_id_ = pool_id;
                obj->Reset();
                return obj;
            }
            ++pool_id;
            pool_id %= kSubPoolCount;
        }
    }

    void
    ReturnOne(std::shared_ptr<T>& obj) {
        auto pool_id = obj->source_pool_id_;
        std::lock_guard<std::mutex> lock(sub_pool_mutexes_[pool_id]);
        pool_[pool_id]->emplace_back(obj);
    }

    inline int64_t
    GetCurrentMemoryUsage() {
        return memory_usage_.load(std::memory_order_relaxed);
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

    std::atomic<int64_t> memory_usage_{0};

    ConstructFuncType constructor_{nullptr};
    Allocator* allocator_{nullptr};

    std::shared_ptr<Allocator> owned_allocator_{nullptr};
};
}  // namespace vsag
