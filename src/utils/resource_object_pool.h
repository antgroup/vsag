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
 * @file resource_object_pool.h
 * @brief Thread-safe object pool for reusable resource objects.
 */

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

/**
 * @brief Thread-safe pool for managing reusable resource objects.
 *
 * This template class provides a pool of pre-allocated ResourceObject instances
 * that can be efficiently borrowed and returned. It uses multiple sub-pools with
 * separate mutexes to reduce contention under concurrent access.
 *
 * @tparam T The resource type, must derive from ResourceObject.
 * @tparam Enable SFINAE constraint ensuring T derives from ResourceObject.
 */
template <typename T,
          typename = typename std::enable_if<std::is_base_of<ResourceObject, T>::value>::type>
class ResourceObjectPool {
public:
    /// Function type for constructing new resource objects.
    using ConstructFuncType = std::function<std::shared_ptr<T>()>;
    /// Number of sub-pools for reduced lock contention.
    static constexpr uint64_t kSubPoolCount = 16;

public:
    /**
     * @brief Construct a resource object pool with initial capacity.
     *
     * @tparam Args Constructor argument types for the resource object.
     * @param init_size Initial number of objects to pre-allocate.
     * @param allocator Allocator for memory allocation (can be nullptr for default).
     * @param args Arguments forwarded to the resource object constructor.
     */
    template <typename... Args>
    explicit ResourceObjectPool(uint64_t init_size, Allocator* allocator, Args... args)
        : allocator_(allocator), init_size_(init_size), memory_usage_(0) {
        this->constructor_ = [=, pool = this]() -> std::shared_ptr<T> {
            auto ptr = std::make_shared<T>(args...);
            auto value = ptr->GetMemoryUsage();
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

    /**
     * @brief Take a resource object from the pool.
     *
     * Retrieves an available object from the pool, or creates a new one if
     * the pool is empty. Uses thread-local preference to reduce contention.
     *
     * @return std::shared_ptr<T> A shared pointer to the resource object.
     */
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

    /**
     * @brief Return a resource object to the pool.
     *
     * Returns the object to the same sub-pool it was originally taken from,
     * as tracked by source_pool_id_.
     *
     * @param obj The resource object to return.
     */
    void
    ReturnOne(std::shared_ptr<T>& obj) {
        auto pool_id = obj->source_pool_id_;
        std::lock_guard<std::mutex> lock(sub_pool_mutexes_[pool_id]);
        pool_[pool_id]->emplace_back(obj);
    }

    /**
     * @brief Get the total memory usage of the pool.
     * @return int64_t Memory usage in bytes.
     */
    inline int64_t
    GetMemoryUsage() {
        return memory_usage_.load(std::memory_order_relaxed);
    }

private:
    /**
     * @brief Pre-fill the pool with initial objects.
     * @param size Number of objects to create.
     */
    inline void
    fill(uint64_t size) {
        for (uint64_t i = 0; i < size; ++i) {
            auto sub_pool_idx = i % kSubPoolCount;
            pool_[sub_pool_idx]->emplace_back(this->constructor_());
        }
    }

private:
    std::unique_ptr<Deque<std::shared_ptr<T>>> pool_[kSubPoolCount];  ///< Array of sub-pools
    std::mutex sub_pool_mutexes_[kSubPoolCount];                      ///< Mutexes for each sub-pool
    uint64_t init_size_{0};                                           ///< Initial pool size

    std::atomic<int64_t> memory_usage_{0};  ///< Total memory usage in bytes

    ConstructFuncType constructor_{nullptr};  ///< Object construction function
    Allocator* allocator_{nullptr};           ///< Allocator pointer (non-owning)

    std::shared_ptr<Allocator> owned_allocator_{nullptr};  ///< Owned allocator (if created)
};
}  // namespace vsag