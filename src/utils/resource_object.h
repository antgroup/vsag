
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

namespace vsag {

class ResourceObject {
public:
    ResourceObject() = default;

    virtual ~ResourceObject() = default;

    /**
     * @brief Reset the resource to its initial state.
     *
     * This pure virtual function forces derived classes to provide an
     * implementation for resetting their specific resources. The reset
     * operation should revert the resource to a known, initial state,
     * freeing and reallocating memory if necessary, and ensuring that resources
     * are ready for reuse.
     */
    virtual void
    Reset() = 0;

    /**
     * @brief The ID of the sub-pool this resource object belongs to.
     *
     * This is used to ensure an object is returned to the same sub-pool it was
     * taken from, preventing pool imbalance under concurrent access.
     */
    int64_t source_pool_id_{0};

    /**
     * @brief Get the memory usage of the resource object.
     *
     * This pure virtual function forces derived classes to provide an
     * implementation for returning the memory usage of their specific resources.
     * The memory usage should include all dynamically allocated memory, whether
     * directly or indirectly, used by the resource.
     *
     * @return int64_t The memory usage of the resource object in bytes.
     */
    virtual int64_t
    GetMemoryUsage() const = 0;
};

}  // namespace vsag