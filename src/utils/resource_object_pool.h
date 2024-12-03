
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

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <type_traits>

#include "resource_object.h"
namespace vsag {

template <typename T,
          typename = typename std::enable_if<std::is_base_of<ResourceObject, T>::value>::type>
class ResourceObjectPool {
public:
    using ConstructFuncType = std::function<std::shared_ptr<T>()>;

public:
    template <typename... Args>
    explicit ResourceObjectPool(uint64_t init_size, Args... args) {
        this->pool_size_ = init_size;
        this->constructor_ = [=]() -> std::shared_ptr<T> { return std::make_shared<T>(args...); };
        this->resize(pool_size_);
    }

    void
    SetConstructor(ConstructFuncType func) {
        this->constructor_ = func;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            while (not pool_.empty()) {
                pool_.pop_back();
            }
        }
        this->resize(pool_size_);
    }

    std::shared_ptr<T>
    TakeOne() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (pool_.empty()) {
            return this->constructor_();
        }
        std::shared_ptr<T> obj = pool_.back();
        pool_.pop_back();
        pool_size_--;
        obj->Reset();
        return obj;
    }

    void
    ReturnOne(std::shared_ptr<T>& obj) {
        std::unique_lock<std::mutex> lock(mutex_);
        pool_.push_back(obj);
        pool_size_++;
    }

    [[nodiscard]] inline uint64_t
    GetSize() const {
        return this->pool_size_;
    }

private:
    inline void
    resize(uint64_t size) {
        std::unique_lock<std::mutex> lock(mutex_);
        int count = size - pool_.size();
        while (count > 0) {
            pool_.emplace_back(this->constructor_());
            --count;
        }
        while (count < 0) {
            pool_.pop_back();
            ++count;
        }
    }

    std::vector<std::shared_ptr<T>> pool_{};
    size_t pool_size_{0};
    ConstructFuncType constructor_{nullptr};
    std::mutex mutex_;
};

}  // namespace vsag