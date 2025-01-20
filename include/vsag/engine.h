
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

#include "dataset.h"
#include "index.h"
#include "resource.h"

namespace vsag {
/**
 * @class Engine
 * @brief A class representing the core engine responsible for creating default resource
 * or using outside provided resources
 * During it's lifetime, user can use engine to create index safely.
 */
class Engine {
public:
    /**
     * @brief Constructs an Engine with default settings.
     *
     * Initializes an `Engine` instance without explicitly set resources.
     * Default Resource will be used and managed.
     * User can use the resource safely until the engine shutdown or destruct.
     */
    explicit Engine();

    /**
     * @brief Constructs an Engine with a provided outside resource.
     *
     * @param resource A pointer to the `Resource` object to be associated with this engine.
     * This `Resource` will never be managed by the engine, but can be used.
     * @see Resource
     */
    explicit Engine(Resource* resource);

    /**
     * @brief Gracefully shuts down the engine.
     *
     * Similar to a destructor, this function shuts down the `Engine`. It performs
     * checks and raises warnings if there are still external references to the resources
     * managed by the engine, ensuring caution against potential dangling references.
     */
    void
    Shutdown();

    /**
     * @brief Creates an index within the engine.
     *
     * This function attempts to create an index using the specified `name` and `parameters`.
     * It returns a result which may either contain a shared pointer to the created `Index`
     * or an `Error` object indicating failure conditions.
     *
     * @param name The name assigned to the index type, like "hnsw", "diskann", "hgraph" ...
     * @param parameters A JSON-like string containing various parameters required for index creation.
     * @return tl::expected<std::shared_ptr<Index>, Error> An expected value that contains either
     * a shared pointer to the successfully created `Index` or an `Error` detailing
     * why creation failed.
     * @see Index
     */
    tl::expected<std::shared_ptr<Index>, Error>
    CreateIndex(const std::string& name, const std::string& parameters);

private:
    std::shared_ptr<Resource> resource_;  ///< The resource used by this engine.
};
}  // namespace vsag
