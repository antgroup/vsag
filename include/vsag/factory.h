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
#include <string>

#include "vsag/allocator.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/readerset.h"

namespace vsag {

class Index;
class Reader;
using ReadFunc = std::function<void(uint64_t, uint64_t, void*)>;
using WriteFunc = std::function<void(uint64_t, uint64_t, const void*)>;
using ResizeFunc = std::function<void(uint64_t)>;

class Factory {
public:
    /**
     * @brief Creates an index with the specified name and parameters.
     *
     * This function attempts to create an index using the provided `name` and `parameters`.
     * It returns a result which may either contain a shared pointer to the created `Index`
     * or an `Error` object indicating failure conditions.
     *
     * @param name The name assigned to the index type, like "hgraph", "ivf", or "sindi".
     * @param parameters A string containing configuration parameters for the index. For details on the parameters,
     *  please refer to the example codes in: https://github.com/antgroup/vsag/tree/main/examples/cpp
     * @param allocator An optional allocator for memory management. If not provided, a default allocator will be used.
     * @return tl::expected<std::shared_ptr<Index>, Error> A result containing either the created index or an error.
     */
    static tl::expected<std::shared_ptr<Index>, Error>
    CreateIndex(const std::string& name,
                const std::string& parameters,
                Allocator* allocator = nullptr);

    static tl::expected<std::shared_ptr<Index>, Error>
    CreateIndex(const std::string& name,
                const std::string& parameters,
                const ExternalStorageSet& external_storages,
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

    static std::shared_ptr<Reader>
    CreateReadFuncReader(ReadFunc read_func, uint64_t size);

    static std::shared_ptr<Reader>
    CreateReadFuncReader(ReadFunc read_func, uint64_t base_offset, uint64_t size);

    /**
     * @brief Creates a paired Reader and Writer backed by synchronous callbacks.
     *
     * The adapter invokes read/write/resize storage callbacks while holding its internal mutex.
     * Those callbacks must not re-enter the returned Reader or Writer, including Reader::Size().
     * AsyncRead's completion instead runs synchronously on the calling thread after the mutex is
     * released and may re-enter the adapter. Completion exceptions propagate to that caller without
     * a second completion; worker-thread callers must catch them or use non-throwing completions.
     * A zero-length write beyond the current size grows storage through resize_func so the paired
     * Reader size and backing extent remain consistent. Growth contents are defined by resize_func;
     * the adapter does not guarantee zero-filled bytes. A successful resize callback must expose
     * exactly the requested logical extent (physical allocation may be larger). Callbacks must
     * throw on failure; the adapter publishes a new size only after the callback succeeds.
     */
    static ExternalStorage
    CreateExternalStorage(ReadFunc read_func,
                          WriteFunc write_func,
                          ResizeFunc resize_func,
                          uint64_t initial_size);

private:
    Factory() = default;
};

}  // namespace vsag
