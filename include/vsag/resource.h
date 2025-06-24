
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

#include <memory>

#include "vsag/allocator.h"
#include "vsag/thread_pool.h"

namespace vsag {
/**
 * @class Resource
 * @brief A class for managing resources, primarily focused on memory allocation.
 *
 * The `Resource` class is designed to handle resources with a specific allocator
 */
class Resource {
public:
    /**
     * @brief Constructs a Resource with an allocator and a thread pool
     *
     * This constructor initializes a `Resource` with the given allocator. If no allocator
     * is provided, a default allocator will be created and owned by the Resource.
     * If no thread pool is provided, the Resource will not use a thread pool for its operations.
     *
     * @param allocator A pointer to an external `Allocator` object used for managing resource allocations.
     *                  If `nullptr`, a default allocator will be created and used.
     * @param thread_pool A pointer to a `ThreadPool` object used for executing multi-threaded tasks.
     *                    If `nullptr`, the Resource will not use a thread pool.
     */
    explicit Resource(Allocator* allocator, ThreadPool* thread_pool);

    /**
     * @brief Constructs a Resource with an allocator and a thread pool using shared pointers.
     *
     * This constructor initializes a `Resource` with the given allocator and thread pool via shared pointers.
     * If no allocator is provided (i.e., `allocator` is a null shared pointer), a default allocator will be created
     * and managed by the Resource. If no thread pool is provided (i.e., `thread_pool` is a null shared pointer),
     * the Resource will not use a thread pool for its operations.
     *
     * @param allocator A shared pointer to an external `Allocator` object. If null, a default allocator is created
     *                  and owned by the Resource.
     * @param thread_pool A shared pointer to a `ThreadPool` object. If null, the Resource will not use a thread pool.
     */
    explicit Resource(const std::shared_ptr<Allocator>& allocator,
                      const std::shared_ptr<ThreadPool>& thread_pool);

    /**
     * @brief Constructs a Resource without specifying an allocator.
     *
     * Default allocator will be created and owned.
     */
    explicit Resource();

    /// Virtual destructor for proper cleanup of derived classes.
    virtual ~Resource() = default;

    /**
     * @brief Retrieves the allocator associated with this resource.
     *
     * This function returns a shared pointer to the `Allocator` associated with this resource.
     *
     * @return std::shared_ptr<Allocator> A shared pointer to the allocator. If no allocator was provided,
     *                                    a default allocator will be returned.
     */
    virtual std::shared_ptr<Allocator>
    GetAllocator() const {
        return this->allocator;
    }

    /**
     * @brief Retrieves the thread pool associated with this resource.
     *
     * This function returns a shared pointer to the `ThreadPool` associated with this resource.
     *
     * @return std::shared_ptr<ThreadPool> A shared pointer to the thread pool. If no thread pool was provided,
     *                                     a null shared pointer will be returned.
     */
    virtual std::shared_ptr<ThreadPool>
    GetThreadPool() const {
        return this->thread_pool;
    }

private:
    ///< Shared pointer to the allocator associated with this resource.
    std::shared_ptr<Allocator> allocator;

    ///< Shared pointer to the thread pool associated with this resource.
    std::shared_ptr<ThreadPool> thread_pool;
};
}  // namespace vsag
