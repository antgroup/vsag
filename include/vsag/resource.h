
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

#include "allocator.h"

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
     * @brief Constructs a Resource with an optional allocator.
     *
     * This constructor initializes a `Resource` with a given allocator. If no allocator
     * is provided, default allocator will be created and owned by Resource
     *
     * @param allocator A outside pointer to an `Allocator` object used for
     * managing resource allocations.
     */
    explicit Resource(Allocator* allocator);

    /**
     * @brief Constructs a Resource without specifying an allocator.
     *
     * Default allocator will be created and owned.
     */
    Resource() : Resource(nullptr) {
    }

    /// Virtual destructor for proper cleanup of derived classes.
    virtual ~Resource() = default;

    /**
     * @brief Retrieves the allocator associated with this resource.
     *
     * This function returns a shared pointer to the `Allocator` associated with this resource,
     *
     * @return std::shared_ptr<Allocator> A shared pointer to the allocator.
     */
    virtual std::shared_ptr<Allocator>
    GetAllocator() {
        return this->allocator;
    }

public:
    ///< Shared pointer to the allocator associated with this resource.
    std::shared_ptr<Allocator> allocator;
};
}  // namespace vsag
