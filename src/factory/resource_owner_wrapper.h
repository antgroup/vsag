
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

#include "vsag/resource.h"

namespace vsag {

/**
 * @brief Wrapper that optionally takes ownership of a Resource object.
 *
 * This class wraps an existing Resource instance and can optionally take
 * ownership of it, deleting the wrapped resource when this wrapper is destroyed.
 * This is useful for managing resource lifetimes in contexts where ownership
 * semantics need to be controlled dynamically.
 */
class ResourceOwnerWrapper : public Resource {
public:
    /**
     * @brief Constructs a wrapper around a Resource.
     *
     * @param resource Pointer to the Resource to wrap.
     * @param owned If true, the wrapper takes ownership and deletes the resource
     *              on destruction. Defaults to false.
     */
    explicit ResourceOwnerWrapper(Resource* resource, bool owned = false)
        : Resource(nullptr, nullptr), resource_(resource), owned_(owned) {
    }

    /**
     * @brief Gets the allocator from the wrapped resource.
     *
     * @return The allocator instance from the wrapped resource.
     */
    std::shared_ptr<Allocator>
    GetAllocator() const override {
        return resource_->GetAllocator();
    }

    /**
     * @brief Gets the thread pool from the wrapped resource.
     *
     * @return The thread pool instance from the wrapped resource.
     */
    std::shared_ptr<ThreadPool>
    GetThreadPool() const override {
        return resource_->GetThreadPool();
    }

    /**
     * @brief Destructor that deletes the wrapped resource if owned.
     */
    ~ResourceOwnerWrapper() override {
        if (owned_) {
            delete resource_;
        }
    }

private:
    Resource* resource_{nullptr};  ///< Wrapped resource instance
    bool owned_{false};            ///< Ownership flag for the wrapped resource
};

}  // namespace vsag
