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
#include "vsag/index.h"
#include "vsag/readerset.h"

namespace vsag {

class Factory {
public:
    /**
     * @brief Creates an index with the specified name and parameters.
     *
     * This function attempts to create an index using the provided `name` and `parameters`.
     * It returns a result which may either contain a shared pointer to the created `Index`
     * or an `Error` object indicating failure conditions.
     *
     * @param name The name assigned to the index type, like "hnsw", "diskann", "hgraph" ...
     * @param parameters A string containing configuration parameters for the index. For details on the parameters,
     *  please refer to the example codes in: https://github.com/antgroup/vsag/tree/main/examples/cpp
     * @param allocator An optional allocator for memory management. If not provided, a default allocator will be used.
     * @return tl::expected<std::shared_ptr<Index>, Error> A result containing either the created index or an error.
     */
    static tl::expected<std::shared_ptr<Index>, Error>
    CreateIndex(const std::string& name,
                const std::string& parameters,
                Allocator* allocator = nullptr);

    /**
     * @brief Creates a local file reader for the specified file.
     *
     * This function creates a reader that can read data from a local file,
     * starting from a specified base offset and reading a defined size.
     *
     * @param filename The path to the local file to be read.
     * @param base_offset The offset in the file from which to start reading.
     * @param size The number of bytes to read from the file.
     * @return std::shared_ptr<Reader> A shared pointer to the created local file reader.
     */
    static std::shared_ptr<Reader>
    CreateLocalFileReader(const std::string& filename, int64_t base_offset, int64_t size);

private:
    Factory() = default;
};

}  // namespace vsag
